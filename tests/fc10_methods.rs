use malleus::{BufferBinding, ExecutableModule, Interpreter, OperandId, validate_module};
use quantitas::UnitRegistry;
use scientia::{
    AffineMethodKernelSpec, MethodCompileError, MethodFamily, MethodProgramKind,
    MethodSelectionReceipt, compile_boundary_integral_method, compile_conservation_law_method,
    compile_finite_difference_method, compile_network_dae_method, compile_particle_method,
    compile_semantics,
};
use std::sync::Arc;

const METHODS: &str = r#"
module fixtures.fc10;

model Conservation {
  domain Cells { dimension = 1; coordinates = cartesian; }
  field q: state scalar DG(order=0) on Cells { time_role = differential; };
  property speed = transport_speed(0);
  equation balance on Cells { dt(q) + div(speed * q) = 0; }
}

model Stencil {
  domain Grid { dimension = 2; coordinates = cartesian; }
  field u: state scalar H1(order=1) on Grid { time_role = differential; };
  equation diffusion on Grid { dt(u) - div(grad(u)) = 0; }
}

model Network {
  domain Graph { dimension = 0; coordinates = lumped; }
  field voltage: state scalar L2(order=0) on Graph { time_role = differential; };
  field current: state scalar L2(order=0) on Graph { time_role = differential; };
  equation node on Graph { dt(voltage) + current = 0; }
  equation branch on Graph { dt(current) - voltage = 0; }
}

model Particles {
  domain Cloud { dimension = 0; coordinates = particle_space; }
  field positions: state vector(2) L2(order=0) on Cloud { time_role = differential; };
  field velocities: state vector(2) L2(order=0) on Cloud { time_role = differential; };
  property mass = particle_mass(0);
  property stiffness = pair_stiffness(0);
  constitutive pair_forces = radial_pair_force(positions, stiffness);
  equation kinematics on Cloud { dt(positions) = velocities; }
  equation dynamics on Cloud { mass * dt(velocities) = pair_forces; }
}

model Boundary {
  domain Ambient { dimension = 2; coordinates = cartesian; }
  field density: unknown scalar H1(order=1) on Ambient;
  source incident: IncidentField;
  equation representation on Ambient { -div(grad(density)) = incident; }
  boundary surface on boundary("surface") { robin density = single_layer(density); }
}
"#;

fn semantics() -> scientia::SemanticModule {
    compile_semantics(METHODS, &UnitRegistry::si_bootstrap())
        .unwrap()
        .semantic
}

fn affine(name: &str, inputs: &[&str], coefficients: &[f64]) -> AffineMethodKernelSpec {
    AffineMethodKernelSpec {
        name: name.into(),
        inputs: inputs.iter().map(|input| (*input).into()).collect(),
        coefficients: coefficients.to_vec(),
        constant: 0.0,
    }
}

#[test]
fn five_sibling_compilers_produce_distinct_nonvariational_artifacts() {
    let module = semantics();
    let finite_volume = compile_conservation_law_method(
        &module,
        "Conservation",
        "balance",
        "q",
        affine("upwind_flux", &["minus", "plus"], &[2.0, 0.0]),
    )
    .unwrap();
    let finite_difference = compile_finite_difference_method(
        &module,
        "Stencil",
        "diffusion",
        "u",
        vec![-1, 0, 1],
        affine(
            "centered_second_difference",
            &["left", "center", "right"],
            &[1.0, -2.0, 1.0],
        ),
    )
    .unwrap();
    let network = compile_network_dae_method(
        &module,
        "Network",
        &["node", "branch"],
        &["voltage", "current"],
    )
    .unwrap();
    let particle = compile_particle_method(
        &module,
        "Particles",
        &["kinematics", "dynamics"],
        "positions",
        "velocities",
        "pair_forces",
    )
    .unwrap();
    let boundary = compile_boundary_integral_method(
        &module,
        "Boundary",
        "representation",
        "density",
        "surface",
    )
    .unwrap();

    let programs = [
        finite_volume,
        finite_difference,
        network,
        particle,
        boundary,
    ];
    assert_eq!(
        programs
            .iter()
            .map(|program| program.family())
            .collect::<Vec<_>>(),
        vec![
            MethodFamily::ConservationLawFiniteVolume,
            MethodFamily::StructuredStencilFiniteDifference,
            MethodFamily::NetworkDae,
            MethodFamily::Particle,
            MethodFamily::BoundaryIntegral,
        ]
    );
    assert!(programs.iter().all(|program| {
        program.receipt.selected_without_variational_form
            && program.receipt.source_semantic_digest == program.source_semantic_digest
            && program.schema == "scientia-method-program/2"
    }));
    assert!(programs.iter().all(|program| {
        let model = module
            .models
            .iter()
            .find(|model| model.name == program.model)
            .unwrap();
        Arc::ptr_eq(&model.expressions, &program.expressions)
    }));
    let MethodProgramKind::ConservationLawFiniteVolume(finite_volume) = &programs[0].kind else {
        panic!("first artifact must be finite volume")
    };
    assert_eq!(
        programs[0].receipt.selection,
        MethodSelectionReceipt::ConservationLawFiniteVolume {
            time_derivative: finite_volume.time_derivative,
            flux_divergence: finite_volume.flux_divergence,
            flux: finite_volume.flux,
        }
    );
    let MethodProgramKind::StructuredStencilFiniteDifference(finite_difference) = &programs[1].kind
    else {
        panic!("second artifact must be finite difference")
    };
    assert_eq!(
        programs[1].receipt.selection,
        MethodSelectionReceipt::StructuredStencilFiniteDifference {
            spatial_differential: finite_difference.spatial_differential,
        }
    );
    let digests = programs
        .iter()
        .map(|program| program.artifact_digest.clone())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(digests.len(), programs.len());
}

#[test]
fn fv_and_fd_affine_kernels_execute_with_malleus_reference_semantics() {
    let module = semantics();
    let cases = [
        (
            compile_conservation_law_method(
                &module,
                "Conservation",
                "balance",
                "q",
                affine("flux", &["minus", "plus"], &[2.0, 0.0]),
            )
            .unwrap(),
            vec![3.0, -7.0],
            6.0,
        ),
        (
            compile_finite_difference_method(
                &module,
                "Stencil",
                "diffusion",
                "u",
                vec![-1, 0, 1],
                affine("stencil", &["left", "center", "right"], &[1.0, -2.0, 1.0]),
            )
            .unwrap(),
            vec![1.0, 4.0, 9.0],
            2.0,
        ),
    ];
    for (program, mut values, expected) in cases {
        let kernel = program.local_kernel.unwrap();
        let executable = ExecutableModule::reference(validate_module(kernel.module).unwrap());
        let executable = &executable.kernels()[0];
        let mut output = [0.0];
        let input_count = values.len();
        let mut bindings = values
            .iter_mut()
            .enumerate()
            .map(|(index, value)| {
                BufferBinding::new(OperandId::new(index), std::slice::from_mut(value))
            })
            .collect::<Vec<_>>();
        bindings.push(BufferBinding::new(OperandId::new(input_count), &mut output));
        Interpreter::run(executable, &mut bindings).unwrap();
        assert_eq!(output[0], expected);
    }
}

#[test]
fn complete_method_program_round_trips_with_its_malleus_module() {
    let program = compile_conservation_law_method(
        &semantics(),
        "Conservation",
        "balance",
        "q",
        affine("flux", &["minus", "plus"], &[2.0, 0.0]),
    )
    .unwrap();
    let encoded = serde_json::to_vec(&program).unwrap();
    let decoded: scientia::MethodProgram = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, program);
    validate_module(decoded.local_kernel.unwrap().module).unwrap();
}

#[test]
fn compilers_refuse_wrong_method_domains_and_invalid_stencils() {
    let module = semantics();
    assert!(matches!(
        compile_network_dae_method(&module, "Stencil", &["diffusion"], &["u"]),
        Err(MethodCompileError::DomainMismatch(_))
    ));
    assert!(matches!(
        compile_finite_difference_method(
            &module,
            "Stencil",
            "diffusion",
            "u",
            vec![-1, -1, 1],
            affine("bad", &["a", "b", "c"], &[1.0, -2.0, 1.0]),
        ),
        Err(MethodCompileError::InvalidKernel(_))
    ));
}
