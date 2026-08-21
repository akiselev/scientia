use malleus::{
    AccessMode, BufferBinding, ExecutableModule, Interpreter, OperandId, validate_module,
};
use quantitas::UnitRegistry;
use scientia::{
    EvaluationSite, OPERATOR_SYSTEM_SCHEMA, OrientationRequirement, TensorSide,
    compile_authored_operator_system, compile_operator_system, compile_semantics,
};

const FC8_SYSTEMS: &str = r#"
module fc8.systems;

model Elasticity {
  domain Omega { dimension = 3; coordinates = cartesian; }
  field displacement: unknown vector(3) H1(order=1) on Omega;
  property lambda = lame_lambda(0);
  property mu = lame_mu(0);
  source body_force: MechanicalBodyForce;
  constitutive strain = sym_grad(displacement);
  constitutive stress = lambda * trace(strain) * identity(3) + 2 * mu * strain;
  equation momentum on Omega { -div(stress) = body_force; }
  boundary clamp on boundary("clamp") { dirichlet displacement = [0, 0, 0]; }
}

model Stokes {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field velocity: unknown vector(2) H1(order=2) on Omega;
  field pressure: unknown scalar L2(order=1) on Omega;
  property mu = dynamic_viscosity(0);
  source body_force: MechanicalBodyForce;
  constitutive strain = sym_grad(velocity);
  constitutive stress = 2 * mu * strain;
  equation momentum on Omega { -div(stress) + grad(pressure) = body_force; }
  equation incompressibility on Omega { div(velocity) = 0; }
  boundary walls on boundary("walls") { dirichlet velocity = [0, 0]; }
}

model Darcy {
  domain Omega { dimension = 3; coordinates = cartesian; }
  field flux: unknown vector(3) HDiv(order=0) on Omega;
  field pressure: unknown scalar L2(order=0) on Omega;
  property permeability = permeability_tensor(0);
  property viscosity = dynamic_viscosity(0);
  source source_term: MassSource;
  source body_force: BodyForce;
  constitutive mobility_inverse = viscosity * inverse(permeability);
  equation darcy_law on Omega { mobility_inverse * flux + grad(pressure) = body_force; }
  equation mass_balance on Omega { div(flux) = source_term; }
  boundary walls on boundary("walls") { neumann flux = 0; }
}

model Maxwell {
  domain Omega { dimension = 3; coordinates = cartesian; }
  field electric_re: unknown vector(3) HCurl(order=0) on Omega;
  field electric_im: unknown vector(3) HCurl(order=0) on Omega;
  parameter omega = excitation_frequency();
  property mu_inv = inverse_permeability(omega);
  property epsilon = permittivity(omega);
  property sigma = electrical_conductivity(omega);
  source current_re: CurrentDensity;
  source current_im: CurrentDensity;
  equation real_part on Omega {
    curl(mu_inv * curl(electric_re)) - omega ^ 2 * epsilon * electric_re
      + omega * sigma * electric_im = current_re;
  }
  equation imaginary_part on Omega {
    curl(mu_inv * curl(electric_im)) - omega ^ 2 * epsilon * electric_im
      - omega * sigma * electric_re = current_im;
  }
  boundary pec_re on boundary("pec") { dirichlet electric_re = tangential_zero(); }
  boundary pec_im on boundary("pec") { dirichlet electric_im = tangential_zero(); }
}
"#;

const DG_FACET: &str = r#"
module fc8.dg;
model TransportFacet {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field state: trial scalar DG(order=1) on Omega;
  field test: test scalar DG(order=1) on Omega;
  form upwind_skeleton {
    interior_facet(faces): jump(state) * average(test);
  }
}
"#;

fn execute_primal_with_ones(bundle: &scientia::StructuredPointKernelBundle) {
    let module = ExecutableModule::reference(validate_module(bundle.module.clone()).unwrap());
    let executable = &module.kernels()[bundle.primal_kernel_index];
    let kernel = executable.kernel().as_kernel();
    let mut buffers = kernel
        .operands
        .iter()
        .map(|operand| {
            let fill = if matches!(operand.access, AccessMode::Read | AccessMode::ReadWrite) {
                1.0
            } else {
                0.0
            };
            vec![fill; operand.region.offset + operand.region.length]
        })
        .collect::<Vec<_>>();
    let mut bindings = buffers
        .iter_mut()
        .enumerate()
        .map(|(index, values)| BufferBinding::new(OperandId::new(index), values))
        .collect::<Vec<_>>();
    Interpreter::run(executable, &mut bindings).unwrap();
    drop(bindings);
    assert!(buffers.iter().flatten().all(|value| value.is_finite()));
}

#[test]
fn mixed_gate_models_compile_complete_block_kernel_chains() {
    let compilation = compile_semantics(FC8_SYSTEMS, &UnitRegistry::si_bootstrap()).unwrap();
    let cases = [
        ("Elasticity", vec!["momentum"], 1, 1),
        ("Stokes", vec!["momentum", "incompressibility"], 2, 3),
        ("Darcy", vec!["darcy_law", "mass_balance"], 2, 3),
        ("Maxwell", vec!["real_part", "imaginary_part"], 2, 4),
    ];
    for (model, equations, rows, coordinates) in cases {
        let system = compile_operator_system(&compilation.semantic, model, &equations)
            .unwrap_or_else(|error| panic!("{model}: {error}"));
        assert_eq!(system.schema, OPERATOR_SYSTEM_SCHEMA);
        assert_eq!(system.blocks.len(), rows);
        assert_eq!(
            system
                .blocks
                .iter()
                .map(|block| block.coordinates.len())
                .sum::<usize>(),
            coordinates,
            "{model}"
        );
        assert!(
            system
                .blocks
                .iter()
                .all(|block| !block.kernels.bundles.is_empty())
        );
        for bundle in system
            .blocks
            .iter()
            .flat_map(|block| &block.kernels.bundles)
        {
            execute_primal_with_ones(bundle);
        }
    }
}

#[test]
fn complete_operator_system_round_trips_with_every_compiler_stage() {
    let compilation = compile_semantics(FC8_SYSTEMS, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_operator_system(
        &compilation.semantic,
        "Stokes",
        &["momentum", "incompressibility"],
    )
    .unwrap();
    let encoded = serde_json::to_vec(&system).unwrap();
    let decoded: scientia::OperatorSystem = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, system);
    for bundle in decoded
        .blocks
        .into_iter()
        .flat_map(|block| block.kernels.bundles)
    {
        validate_module(bundle.module).unwrap();
    }
}

#[test]
fn dg_two_sided_traces_lower_through_the_same_tensor_and_kernel_contracts() {
    let compilation = compile_semantics(DG_FACET, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_authored_operator_system(
        &compilation.semantic,
        "TransportFacet",
        &["upwind_skeleton"],
    )
    .unwrap();
    let block = &system.blocks[0];
    let requirements = &block.requirements;
    assert!(requirements.spaces.iter().all(|space| {
        space
            .orientations
            .contains(&OrientationRequirement::TwoSidedFacet)
    }));
    let factorization = &block.factorization;
    assert!(
        factorization.integrals[0]
            .primal
            .inputs
            .iter()
            .any(|input| {
                input.binding.evaluation.site == EvaluationSite::MinusTrace
                    && input.side == TensorSide::Minus
            })
    );
    assert!(
        factorization.integrals[0]
            .primal
            .inputs
            .iter()
            .any(|input| {
                input.binding.evaluation.site == EvaluationSite::PlusTrace
                    && input.side == TensorSide::Plus
            })
    );
    assert!(!block.kernels.bundles.is_empty());
    for bundle in &block.kernels.bundles {
        execute_primal_with_ones(bundle);
    }
}
