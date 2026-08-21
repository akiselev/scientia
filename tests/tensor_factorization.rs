use quantitas::UnitRegistry;
use scientia::{
    BasisAdjoint, DenseTensor, DerivativeMode, ElementExecutionContext, FormComplexConvention,
    GeometryPreprocessingRequirement, InputSourceRequirement, OperatorAction, OperatorStage,
    QFunctionConstruction, TensorAxisRole, TensorBinaryOp, TensorCompileError, TensorInputRole,
    TensorInterpretError, TensorReductionOp, TensorScalarExpr, compile_semantics,
    compile_variational_form, derive_variational_form, factor_operator, infer_form_requirements,
    interpret_element_operator,
};

const POISSON: &str = r#"
module fc4.poisson;
model Poisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  property k = diffusivity(0);
  source f: VolumetricSource;
  equation balance on Omega { -div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(); }
}
"#;

const VALUE_AND_GRADIENT: &str = r#"
module fc4.value_gradient;
model ValueGradient {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: trial scalar H1(order=1) on Omega;
  field v: test scalar H1(order=1) on Omega;
  form residual {
    cell(Omega): u * v + dot(grad(u), grad(v));
  }
}
"#;

fn poisson_factorization() -> (
    scientia::SemanticCompilation,
    scientia::OperatorFactorization,
) {
    let compilation = compile_semantics(POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    (compilation, factorization)
}

fn contains_reduction(expression: &TensorScalarExpr) -> bool {
    match expression {
        TensorScalarExpr::Reduction {
            op: TensorReductionOp::Sum,
            axis,
            ..
        } => axis.role == TensorAxisRole::Reduction,
        TensorScalarExpr::Unary { arg, .. } => contains_reduction(arg),
        TensorScalarExpr::Binary { lhs, rhs, .. } => {
            contains_reduction(lhs) || contains_reduction(rhs)
        }
        _ => false,
    }
}

fn contains_input(expression: &TensorScalarExpr, input: scientia::TensorInputId) -> bool {
    match expression {
        TensorScalarExpr::Input {
            input: candidate, ..
        } => *candidate == input,
        TensorScalarExpr::Unary { arg, .. } => contains_input(arg, input),
        TensorScalarExpr::Binary { lhs, rhs, .. } => {
            contains_input(lhs, input) || contains_input(rhs, input)
        }
        TensorScalarExpr::Reduction { expression, .. } => contains_input(expression, input),
        _ => false,
    }
}

#[test]
fn fc4_emits_explicit_indexed_qfunctions_and_operator_stages() {
    let (_, factorization) = poisson_factorization();
    assert_eq!(factorization.integrals.len(), 2);
    assert_eq!(
        factorization.receipt.complex_convention,
        FormComplexConvention::ExplicitConjugationOnly
    );

    let diffusion = factorization
        .integrals
        .iter()
        .find(|integral| {
            integral
                .primal
                .outputs
                .iter()
                .any(|output| output.shape == [2])
        })
        .unwrap();
    assert_eq!(
        diffusion.tensor_program.scalar_semantics,
        scientia::TensorScalarSemantics::Real64
    );
    assert_eq!(diffusion.tensor_program.output.shape, Vec::<usize>::new());
    assert!(
        diffusion
            .tensor_program
            .inputs
            .iter()
            .any(|input| input.role == scientia::TensorProgramInputRole::Test)
    );
    assert_eq!(
        diffusion.primal.receipt.source_tensor_program_digest,
        diffusion.tensor_program.artifact_digest
    );
    let flux = diffusion
        .primal
        .outputs
        .iter()
        .find(|output| output.shape == [2])
        .unwrap();
    assert_eq!(flux.free_axes.len(), 1);
    assert_eq!(flux.free_axes[0].extent, 2);
    assert_eq!(flux.side, scientia::TensorSide::Cell);
    assert_eq!(flux.basis_adjoint, BasisAdjoint::Transpose);
    assert!(contains_reduction(&flux.expression));
    assert!(diffusion.stages.iter().any(|stage| matches!(
        stage,
        OperatorStage::Restriction {
            direction: scientia::RestrictionDirection::Gather,
            ..
        }
    )));
    assert!(diffusion.stages.iter().any(|stage| matches!(
        stage,
        OperatorStage::Geometry {
            requirement: GeometryPreprocessingRequirement::InverseJacobian
        }
    )));
    assert!(diffusion.stages.iter().any(|stage| matches!(
        stage,
        OperatorStage::QFunction { primal, jvp }
            if *primal == diffusion.primal.artifact_digest && *jvp == diffusion.jvp.artifact_digest
    )));
    assert!(diffusion.stages.iter().any(|stage| matches!(
        stage,
        OperatorStage::BasisAdjoint {
            action: BasisAdjoint::Transpose,
            ..
        }
    )));
    assert!(diffusion.stages.iter().any(|stage| matches!(
        stage,
        OperatorStage::Restriction {
            direction: scientia::RestrictionDirection::Scatter,
            ..
        }
    )));
    assert!(diffusion.stages.iter().any(|stage| matches!(
        stage,
        OperatorStage::EssentialConstraints { constraints } if constraints.len() == 1
    )));

    let derivative = diffusion.jvp.derivative_receipt.as_ref().unwrap();
    assert_eq!(derivative.mode, DerivativeMode::Jvp);
    assert_eq!(
        derivative.primal_artifact_digest,
        diffusion.primal.artifact_digest
    );
    assert_eq!(
        diffusion.jvp.receipt.construction,
        QFunctionConstruction::SymbolicJvp
    );
    assert_eq!(derivative.active_inputs.len(), 1);
    assert!(!derivative.frozen_inputs.is_empty());
    let direction = diffusion
        .jvp
        .inputs
        .iter()
        .find(|input| matches!(input.role, TensorInputRole::Direction { .. }))
        .unwrap();
    assert!(
        diffusion
            .jvp
            .outputs
            .iter()
            .any(|output| contains_input(&output.expression, direction.id))
    );

    let wire = serde_json::to_string(&factorization).unwrap();
    let round_trip: scientia::OperatorFactorization = serde_json::from_str(&wire).unwrap();
    assert_eq!(round_trip, factorization);
}

fn element_context(
    compilation: &scientia::SemanticCompilation,
    factorization: &scientia::OperatorFactorization,
    unknown_values: [f64; 3],
    direction_values: [f64; 3],
) -> ElementExecutionContext {
    let model = &compilation.semantic.models[0];
    let symbol = |name: &str| {
        model
            .symbols
            .iter()
            .find(|symbol| symbol.name == name)
            .unwrap()
            .id
    };
    let unknown = symbol("u");
    let property = symbol("k");
    let source = symbol("f");

    let mut context = ElementExecutionContext::default();
    context.element_dofs.insert(
        unknown,
        DenseTensor::new(vec![3], unknown_values.to_vec()).unwrap(),
    );
    context.direction_dofs.insert(
        unknown,
        DenseTensor::new(vec![3], direction_values.to_vec()).unwrap(),
    );
    context.quadrature_weights = vec![0.5];
    context.geometry.insert(
        GeometryPreprocessingRequirement::JacobianDeterminant,
        DenseTensor::scalar(1.0),
    );

    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                context
                    .basis
                    .entry(input.binding.clone())
                    .or_insert_with(|| match input.binding.evaluation.derivative {
                        scientia::DerivativeEvaluation::Value => {
                            DenseTensor::new(vec![1, 3], vec![1.0 / 3.0; 3]).unwrap()
                        }
                        scientia::DerivativeEvaluation::Gradient => {
                            DenseTensor::new(vec![1, 3, 2], vec![-1.0, -1.0, 1.0, 0.0, 0.0, 1.0])
                                .unwrap()
                        }
                        other => panic!("unexpected Poisson evaluation {other:?}"),
                    });
            } else if input.binding.symbol == property {
                context
                    .point_values
                    .insert(input.binding.clone(), DenseTensor::scalar(2.0));
            } else if input.binding.symbol == source {
                context
                    .point_values
                    .insert(input.binding.clone(), DenseTensor::scalar(5.0));
            }
        }
        for output in &integral.primal.outputs {
            context
                .basis
                .entry(output.binding.clone())
                .or_insert_with(|| match output.binding.evaluation.derivative {
                    scientia::DerivativeEvaluation::Value => {
                        DenseTensor::new(vec![1, 3], vec![1.0 / 3.0; 3]).unwrap()
                    }
                    scientia::DerivativeEvaluation::Gradient => {
                        DenseTensor::new(vec![1, 3, 2], vec![-1.0, -1.0, 1.0, 0.0, 0.0, 1.0])
                            .unwrap()
                    }
                    other => panic!("unexpected Poisson test evaluation {other:?}"),
                });
        }
    }
    context
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

#[test]
fn independent_p1_triangle_fixture_validates_poisson_residual_and_jvp() {
    let (compilation, factorization) = poisson_factorization();
    let test_symbol = factorization.integrals[0].primal.outputs[0].binding.symbol;
    let context = element_context(
        &compilation,
        &factorization,
        [1.0, 2.0, 4.0],
        [2.0, -1.0, 3.0],
    );

    // Independent analytic P1 triangle result:
    // K = [[2,-1,-1],[-1,1,0],[-1,0,1]], load = 5/6 [1,1,1].
    let residual =
        interpret_element_operator(&factorization, OperatorAction::Residual, &context).unwrap();
    assert_close(
        &residual[&test_symbol].data,
        &[-29.0 / 6.0, 1.0 / 6.0, 13.0 / 6.0],
        1.0e-13,
    );
    let jvp = interpret_element_operator(&factorization, OperatorAction::Jvp, &context).unwrap();
    assert_close(&jvp[&test_symbol].data, &[2.0, -3.0, 1.0], 1.0e-13);

    let epsilon = 1.0e-7;
    let perturbed = element_context(
        &compilation,
        &factorization,
        [1.0 + 2.0 * epsilon, 2.0 - epsilon, 4.0 + 3.0 * epsilon],
        [2.0, -1.0, 3.0],
    );
    let perturbed =
        interpret_element_operator(&factorization, OperatorAction::Residual, &perturbed).unwrap();
    let finite_difference = perturbed[&test_symbol]
        .data
        .iter()
        .zip(&residual[&test_symbol].data)
        .map(|(perturbed, base)| (perturbed - base) / epsilon)
        .collect::<Vec<_>>();
    assert_close(&finite_difference, &jvp[&test_symbol].data, 1.0e-8);
}

#[test]
fn tensor_program_digest_is_presentation_invariant() {
    let registry = UnitRegistry::si_bootstrap();
    let first = compile_semantics(POISSON, &registry).unwrap();
    let formatted_source = scientia::format_scientific_module(&first.source);
    let second = compile_semantics(&formatted_source, &registry).unwrap();
    let compile = |compilation: &scientia::SemanticCompilation| {
        let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
        let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
        factor_operator(&form, &requirements).unwrap()
    };
    assert_eq!(
        compile(&first).artifact_digest,
        compile(&second).artifact_digest
    );
}

#[test]
fn scalar_source_dual_keeps_explicit_negative_sign() {
    let (_, factorization) = poisson_factorization();
    let source = factorization
        .integrals
        .iter()
        .find(|integral| integral.primal.outputs[0].shape.is_empty())
        .unwrap();
    assert!(matches!(
        source.primal.outputs[0].expression,
        TensorScalarExpr::Unary {
            op: scientia::TensorUnaryOp::Neg,
            ..
        } | TensorScalarExpr::Binary {
            op: TensorBinaryOp::Mul,
            ..
        }
    ));
}

#[test]
fn shape_lookup_is_keyed_by_evaluation_not_requirement_order() {
    let compilation = compile_semantics(VALUE_AND_GRADIENT, &UnitRegistry::si_bootstrap()).unwrap();
    let form =
        compile_variational_form(&compilation.semantic, "ValueGradient", "residual").unwrap();
    let mut requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    for input in requirements
        .integral_groups
        .iter_mut()
        .flat_map(|group| &mut group.signature.inputs)
    {
        input.evaluations.reverse();
    }

    let factorization = factor_operator(&form, &requirements).unwrap();
    let trial = form
        .arguments
        .iter()
        .find(|argument| argument.role == scientia::FormArgumentRole::Trial)
        .unwrap()
        .symbol;
    let inputs = &factorization.integrals[0].tensor_program.inputs;
    assert!(inputs.iter().any(|input| {
        input.binding.symbol == trial
            && input.binding.evaluation.derivative == scientia::DerivativeEvaluation::Value
            && input.shape.is_empty()
    }));
    assert!(inputs.iter().any(|input| {
        input.binding.symbol == trial
            && input.binding.evaluation.derivative == scientia::DerivativeEvaluation::Gradient
            && input.shape == [2]
    }));
}

#[test]
fn malformed_derivative_shapes_and_active_sources_are_refused() {
    let compilation = compile_semantics(POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let unknown = form
        .captures
        .iter()
        .find(|capture| {
            matches!(
                capture.role,
                scientia::FormCaptureRole::PhysicalField(scientia::scientific::FieldRole::Unknown)
            )
        })
        .unwrap()
        .symbol;

    let mut invalid_shape = requirements.clone();
    let evaluation = invalid_shape
        .integral_groups
        .iter_mut()
        .flat_map(|group| &mut group.signature.inputs)
        .find(|input| input.symbol == unknown)
        .unwrap()
        .evaluations
        .first_mut()
        .unwrap();
    evaluation.derivative = scientia::DerivativeEvaluation::Divergence;
    assert!(matches!(
        factor_operator(&form, &invalid_shape),
        Err(TensorCompileError::Shape(message))
            if message.contains("requires shape [2], got []")
    ));

    let mut invalid_source = requirements;
    invalid_source
        .integral_groups
        .iter_mut()
        .flat_map(|group| &mut group.signature.inputs)
        .find(|input| input.symbol == unknown)
        .unwrap()
        .source = InputSourceRequirement::ExternalValue;
    assert!(matches!(
        factor_operator(&form, &invalid_source),
        Err(TensorCompileError::ActiveInputRequiresBasis {
            symbol,
            input_source: InputSourceRequirement::ExternalValue,
        }) if symbol == unknown
    ));
}

#[test]
fn interpreter_refuses_a_malformed_non_basis_active_artifact() {
    let (_, mut factorization) = poisson_factorization();
    let input = factorization
        .integrals
        .iter_mut()
        .flat_map(|integral| &mut integral.primal.inputs)
        .find(|input| input.role == TensorInputRole::Active)
        .unwrap();
    let binding = input.binding.clone();
    input.source = InputSourceRequirement::ExternalValue;
    let context = ElementExecutionContext {
        quadrature_weights: vec![1.0],
        ..ElementExecutionContext::default()
    };
    assert_eq!(
        interpret_element_operator(&factorization, OperatorAction::Residual, &context),
        Err(TensorInterpretError::NonBasisActiveInput(binding))
    );
}
