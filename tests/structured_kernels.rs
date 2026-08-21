use malleus::{BufferBinding, ExecutableModule, Interpreter, OperandId, validate_module};
use quantitas::UnitRegistry;
use scientia::{
    DenseTensor, InputSourceRequirement, StructuredDerivativeContract, StructuredPointKernelBundle,
    TensorInputId, TensorInputRole, compile_semantics, derive_variational_form, factor_operator,
    infer_form_requirements, interpret_qfunction, lower_operator_kernels,
};
use std::collections::BTreeMap;

const POISSON: &str = r#"
module fc5.poisson;
model Poisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  property k = diffusivity(0);
  source f: VolumetricSource;
  equation balance on Omega { -div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(); }
}
"#;

fn poisson() -> (
    scientia::OperatorFactorization,
    scientia::StructuredOperatorKernels,
) {
    let compilation = compile_semantics(POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let kernels = lower_operator_kernels(&factorization).unwrap();
    (factorization, kernels)
}

fn execute(
    bundle: &StructuredPointKernelBundle,
    kernel_index: usize,
    values: &BTreeMap<OperandId, Vec<f64>>,
) -> Vec<Vec<f64>> {
    let module = ExecutableModule::reference(validate_module(bundle.module.clone()).unwrap());
    let executable = &module.kernels()[kernel_index];
    let kernel = executable.kernel().as_kernel();
    let mut buffers = kernel
        .operands
        .iter()
        .enumerate()
        .map(|(index, operand)| {
            values
                .get(&OperandId::new(index))
                .cloned()
                .unwrap_or_else(|| vec![0.0; operand.shape.iter().product()])
        })
        .collect::<Vec<_>>();
    let mut bindings = buffers
        .iter_mut()
        .enumerate()
        .map(|(index, values)| BufferBinding::new(OperandId::new(index), values))
        .collect::<Vec<_>>();
    Interpreter::run(executable, &mut bindings).unwrap();
    drop(bindings);
    buffers
}

fn primal_values(
    bundle: &StructuredPointKernelBundle,
    inputs: &BTreeMap<TensorInputId, Vec<f64>>,
) -> BTreeMap<OperandId, Vec<f64>> {
    bundle
        .primal_inputs
        .iter()
        .map(|binding| (binding.operand, inputs[&binding.input].clone()))
        .collect()
}

fn execute_derivative(
    bundle: &StructuredPointKernelBundle,
    contract: &StructuredDerivativeContract,
    inputs: &BTreeMap<TensorInputId, Vec<f64>>,
    directions: &BTreeMap<TensorInputId, Vec<f64>>,
    dependent_seed: Option<Vec<f64>>,
) -> Vec<Vec<f64>> {
    let mut values = primal_values(bundle, inputs);
    let input_by_operand = bundle
        .primal_inputs
        .iter()
        .map(|binding| (binding.operand, binding.input))
        .collect::<BTreeMap<_, _>>();
    for binding in &contract.independent_operands {
        if let Some(direction) = directions.get(&input_by_operand[&binding.primal]) {
            values.insert(binding.derivative, direction.clone());
        }
    }
    if let Some(seed) = dependent_seed {
        for binding in &contract.dependent_operands {
            values.insert(binding.derivative, seed.clone());
        }
    }
    execute(bundle, contract.kernel_index, &values)
}

fn output_of(buffers: &[Vec<f64>], contract: &StructuredDerivativeContract) -> Vec<f64> {
    buffers[contract.dependent_operands[0].derivative.index()].clone()
}

#[test]
fn fc5_lowers_complete_validated_malleus_bundles_and_matches_fc4_jvp() {
    let (factorization, kernels) = poisson();
    assert_eq!(
        kernels.source_factorization_digest,
        factorization.artifact_digest
    );
    assert_eq!(kernels.bundles.len(), 2);

    for bundle in &kernels.bundles {
        assert_eq!(bundle.module.kernels.len(), 4);
        validate_module(bundle.module.clone()).unwrap();
        assert_eq!(
            bundle.receipt.source_factorization_digest,
            factorization.artifact_digest
        );
        assert_eq!(bundle.receipt.integral_index, bundle.integral_index);
        assert_eq!(bundle.receipt.output_index, bundle.output_index);
        assert_eq!(bundle.receipt.numeric_policy.scalar_type, "f64");
        assert_eq!(bundle.jvp.mode, malleus::DerivativeMode::Jvp);
        assert_eq!(bundle.vjp.mode, malleus::DerivativeMode::Vjp);
        assert_eq!(bundle.parameter.mode, malleus::DerivativeMode::Jvp);
        assert_eq!(
            bundle.vjp.purpose,
            scientia::StructuredDerivativePurpose::StateAdjoint
        );
        assert!(
            bundle
                .receipt
                .derivative_evidence
                .contains(&scientia::StructuredDerivativeEvidence::StructuredChainRuleIdentity)
        );
    }

    let diffusion = kernels
        .bundles
        .iter()
        .find(|bundle| !bundle.receipt.active_inputs.is_empty())
        .unwrap();
    let integral = &factorization.integrals[diffusion.integral_index];
    let mut inputs = BTreeMap::new();
    let mut directions = BTreeMap::new();
    for input in &integral.primal.inputs {
        let value = if input.role == TensorInputRole::Active {
            vec![1.25, -0.75]
        } else {
            vec![2.5]
        };
        inputs.insert(input.id, value);
        if input.role == TensorInputRole::Active {
            directions.insert(input.id, vec![-0.5, 3.0]);
        }
    }
    let buffers = execute_derivative(diffusion, &diffusion.jvp, &inputs, &directions, None);
    let generated = output_of(&buffers, &diffusion.jvp);
    let symbolic_inputs = integral
        .jvp
        .inputs
        .iter()
        .map(|input| match input.role {
            TensorInputRole::Direction { primal } => {
                DenseTensor::new(input.shape.clone(), directions[&primal].clone()).unwrap()
            }
            _ => DenseTensor::new(input.shape.clone(), inputs[&input.id].clone()).unwrap(),
        })
        .collect::<Vec<_>>();
    let symbolic = interpret_qfunction(&integral.jvp, &symbolic_inputs).unwrap();
    assert_close(&generated, &symbolic[diffusion.output_index].data, 1.0e-14);
}

#[test]
fn complete_structured_kernel_bundle_round_trips_and_revalidates() {
    let (_, kernels) = poisson();
    let encoded = serde_json::to_vec(&kernels).unwrap();
    let decoded: scientia::StructuredOperatorKernels = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, kernels);
    for bundle in decoded.bundles {
        validate_module(bundle.module).unwrap();
    }
}

#[test]
fn poisson_malleus_kernels_match_analytic_tensors_on_multiple_geometries() {
    let (factorization, kernels) = poisson();
    let diffusion = kernels
        .bundles
        .iter()
        .find(|bundle| !bundle.receipt.active_inputs.is_empty())
        .unwrap();
    let source = kernels
        .bundles
        .iter()
        .find(|bundle| bundle.receipt.active_inputs.is_empty())
        .unwrap();
    let diffusion_integral = &factorization.integrals[diffusion.integral_index];
    let source_integral = &factorization.integrals[source.integral_index];
    let u = [1.0, 2.0, 4.0];
    let du = [2.0, -1.0, 3.0];
    let geometries = [
        (
            1.0,
            [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]],
            [[2.0, -1.0, -1.0], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]],
        ),
        (
            6.0,
            [[-0.5, -1.0 / 3.0], [0.5, 0.0], [0.0, 1.0 / 3.0]],
            [
                [13.0 / 6.0, -1.5, -2.0 / 3.0],
                [-1.5, 1.5, 0.0],
                [-2.0 / 3.0, 0.0, 2.0 / 3.0],
            ],
        ),
        (
            5.0,
            [[-0.4, -0.2], [0.6, -0.2], [-0.2, 0.4]],
            [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]],
        ),
    ];

    for (determinant, gradients, analytic_stiffness) in geometries {
        let gradient_u = contract(&gradients, &u);
        let gradient_du = contract(&gradients, &du);
        let diffusion_inputs = qfunction_inputs(diffusion_integral, gradient_u.to_vec(), 2.0);
        let source_inputs = qfunction_inputs(source_integral, Vec::new(), 5.0);
        let diffusion_primal = execute(
            diffusion,
            diffusion.primal_kernel_index,
            &primal_values(diffusion, &diffusion_inputs),
        );
        let source_primal = execute(
            source,
            source.primal_kernel_index,
            &primal_values(source, &source_inputs),
        );
        let flux = &diffusion_primal[diffusion.primal_output.index()];
        let source_value = source_primal[source.primal_output.index()][0];

        let mut residual = [0.0; 3];
        for row in 0..3 {
            residual[row] = 0.5
                * determinant
                * (gradients[row][0] * flux[0] + gradients[row][1] * flux[1] + source_value / 3.0);
        }
        let analytic =
            matvec(&analytic_stiffness, &u).map(|value| value - 5.0 * (0.5 * determinant) / 3.0);
        assert_close(&residual, &analytic, 1.0e-13);

        let active_input = diffusion.receipt.active_inputs[0];
        let directions = BTreeMap::from([(active_input, gradient_du.to_vec())]);
        let derivative = execute_derivative(
            diffusion,
            &diffusion.jvp,
            &diffusion_inputs,
            &directions,
            None,
        );
        let tangent_flux = output_of(&derivative, &diffusion.jvp);
        let tangent: [f64; 3] = std::array::from_fn(|row| {
            0.5 * determinant
                * (gradients[row][0] * tangent_flux[0] + gradients[row][1] * tangent_flux[1])
        });
        assert_close(&tangent, &matvec(&analytic_stiffness, &du), 1.0e-13);

        let epsilon = 1.0e-7;
        let perturbed = std::array::from_fn(|index| u[index] + epsilon * du[index]);
        let finite_difference = matvec(&analytic_stiffness, &perturbed)
            .iter()
            .zip(matvec(&analytic_stiffness, &u))
            .map(|(plus, base)| (plus - base) / epsilon)
            .collect::<Vec<_>>();
        assert_close(&tangent, &finite_difference, 2.0e-8);
    }
}

#[test]
fn generated_vjp_and_parameter_kernels_pass_adjoint_and_finite_difference_checks() {
    let (factorization, kernels) = poisson();
    let diffusion = kernels
        .bundles
        .iter()
        .find(|bundle| !bundle.receipt.active_inputs.is_empty())
        .unwrap();
    let source = kernels
        .bundles
        .iter()
        .find(|bundle| bundle.receipt.active_inputs.is_empty())
        .unwrap();
    let integral = &factorization.integrals[diffusion.integral_index];
    let inputs = qfunction_inputs(integral, vec![1.25, -0.75], 2.5);
    let active = diffusion.receipt.active_inputs[0];
    let direction = vec![-0.5, 3.0];
    let seed = vec![0.75, -1.25];

    let jvp = execute_derivative(
        diffusion,
        &diffusion.jvp,
        &inputs,
        &BTreeMap::from([(active, direction.clone())]),
        None,
    );
    let tangent = output_of(&jvp, &diffusion.jvp);
    let vjp = execute_derivative(
        diffusion,
        &diffusion.vjp,
        &inputs,
        &BTreeMap::new(),
        Some(seed.clone()),
    );
    let active_pair = diffusion.vjp.independent_operands[0];
    let cotangent = &vjp[active_pair.derivative.index()];
    let forward_dot = tangent.iter().zip(&seed).map(|(a, b)| a * b).sum::<f64>();
    let reverse_dot = direction
        .iter()
        .zip(cotangent)
        .map(|(a, b)| a * b)
        .sum::<f64>();
    assert!((forward_dot - reverse_dot).abs() < 1.0e-13);

    let parameter_input = diffusion.receipt.parameter_inputs[0];
    let parameter_direction = 0.4;
    let parameter = execute_derivative(
        diffusion,
        &diffusion.parameter,
        &inputs,
        &BTreeMap::from([(parameter_input, vec![parameter_direction])]),
        None,
    );
    let parameter_tangent = output_of(&parameter, &diffusion.parameter);
    let epsilon = 1.0e-7;
    let mut plus = inputs.clone();
    plus.get_mut(&parameter_input).unwrap()[0] += epsilon * parameter_direction;
    let base = execute(
        diffusion,
        diffusion.primal_kernel_index,
        &primal_values(diffusion, &inputs),
    );
    let perturbed = execute(
        diffusion,
        diffusion.primal_kernel_index,
        &primal_values(diffusion, &plus),
    );
    let finite_difference = perturbed[diffusion.primal_output.index()]
        .iter()
        .zip(&base[diffusion.primal_output.index()])
        .map(|(plus, base)| (plus - base) / epsilon)
        .collect::<Vec<_>>();
    assert_close(&parameter_tangent, &finite_difference, 1.0e-8);

    let source_integral = &factorization.integrals[source.integral_index];
    let source_inputs = qfunction_inputs(source_integral, Vec::new(), 5.0);
    let source_parameter = source.receipt.parameter_inputs[0];
    let result = execute_derivative(
        source,
        &source.parameter,
        &source_inputs,
        &BTreeMap::from([(source_parameter, vec![2.0])]),
        None,
    );
    assert_close(&output_of(&result, &source.parameter), &[-2.0], 0.0);
}

#[test]
fn malformed_fc4_shapes_and_derivative_receipts_are_refused() {
    let (mut factorization, _) = poisson();
    let diffusion = factorization
        .integrals
        .iter_mut()
        .find(|integral| {
            integral
                .primal
                .inputs
                .iter()
                .any(|input| input.role == TensorInputRole::Active)
        })
        .unwrap();
    diffusion
        .primal
        .inputs
        .iter_mut()
        .find(|input| input.role == TensorInputRole::Active)
        .unwrap()
        .shape = vec![3];
    assert!(matches!(
        lower_operator_kernels(&factorization),
        Err(scientia::StructuredLoweringError::Shape(message))
            if message.contains("has extent 3")
    ));

    let (mut factorization, _) = poisson();
    factorization.integrals[0].jvp.derivative_receipt = None;
    assert!(matches!(
        lower_operator_kernels(&factorization),
        Err(scientia::StructuredLoweringError::SourceJvp(message))
            if message.contains("has no derivative receipt")
    ));

    let (mut factorization, _) = poisson();
    let diffusion = factorization
        .integrals
        .iter_mut()
        .find(|integral| {
            integral
                .primal
                .inputs
                .iter()
                .any(|input| input.role == TensorInputRole::Active)
        })
        .unwrap();
    let reduced = diffusion.primal.outputs[0].expression.clone();
    diffusion.primal.outputs[0].expression = scientia::TensorScalarExpr::Binary {
        op: scientia::TensorBinaryOp::Add,
        lhs: Box::new(reduced),
        rhs: Box::new(scientia::TensorScalarExpr::Constant { value: 1.0 }),
    };
    assert!(matches!(
        lower_operator_kernels(&factorization),
        Err(scientia::StructuredLoweringError::Axis(message))
            if message.contains("enclosing nest")
    ));

    let (mut factorization, _) = poisson();
    let diffusion = factorization
        .integrals
        .iter_mut()
        .find(|integral| {
            integral
                .primal
                .inputs
                .iter()
                .any(|input| input.role == TensorInputRole::Active)
        })
        .unwrap();
    let active = diffusion
        .primal
        .inputs
        .iter()
        .find(|input| input.role == TensorInputRole::Active)
        .unwrap()
        .id;
    let free_axis = diffusion.primal.outputs[0].free_axes[0].id;
    let scientia::TensorScalarExpr::Reduction { expression, .. } =
        &mut diffusion.primal.outputs[0].expression
    else {
        panic!("Poisson diffusion output must remain a reduction fixture");
    };
    let existing = std::mem::replace(
        expression,
        Box::new(scientia::TensorScalarExpr::Constant { value: 0.0 }),
    );
    **expression = scientia::TensorScalarExpr::Binary {
        op: scientia::TensorBinaryOp::Add,
        lhs: existing,
        rhs: Box::new(scientia::TensorScalarExpr::Input {
            input: active,
            indices: vec![free_axis],
        }),
    };
    let integral_index = diffusion.integral_index;
    let kernels = lower_operator_kernels(&factorization).unwrap();
    let bundle = kernels
        .bundles
        .iter()
        .find(|bundle| bundle.integral_index == integral_index)
        .unwrap();
    assert_eq!(
        bundle
            .primal_inputs
            .iter()
            .filter(|binding| binding.input == active)
            .count(),
        2,
        "one logical tensor input must bind every distinct affine access map"
    );

    let (factorization, _) = poisson();
    let f32_policy = malleus::NumericPolicy {
        scalar_type: malleus::ScalarType::F32,
        ..malleus::NumericPolicy::default()
    };
    assert!(matches!(
        scientia::lower_operator_kernels_with_policy(&factorization, f32_policy),
        Err(scientia::StructuredLoweringError::ScalarPolicy(message))
            if message.contains("requires real64")
    ));
}

fn qfunction_inputs(
    integral: &scientia::IntegralOperatorFactorization,
    active_value: Vec<f64>,
    external_value: f64,
) -> BTreeMap<TensorInputId, Vec<f64>> {
    integral
        .primal
        .inputs
        .iter()
        .map(|input| {
            let values = if input.role == TensorInputRole::Active {
                active_value.clone()
            } else {
                assert!(input.source != InputSourceRequirement::Basis);
                vec![external_value]
            };
            (input.id, values)
        })
        .collect()
}

fn contract(gradients: &[[f64; 2]; 3], values: &[f64; 3]) -> [f64; 2] {
    std::array::from_fn(|component| {
        (0..3)
            .map(|index| gradients[index][component] * values[index])
            .sum()
    })
}

fn matvec(matrix: &[[f64; 3]; 3], values: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        (0..3)
            .map(|column| matrix[row][column] * values[column])
            .sum()
    })
}

fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected} within {tolerance}"
        );
    }
}
