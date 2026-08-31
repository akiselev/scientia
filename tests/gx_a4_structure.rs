//! GX-A4 `scientia-operator-structure/1` (contract C5.4) fixtures.

use quantitas::UnitRegistry;
use scientia::{
    BlockClass, FormSymmetry, Linearity, NullspaceKind, StructurePropertyTangent,
    compile_operator_system, compile_semantics, compile_variational_form,
    derive_operator_structure, derive_operator_structure_for_system, derive_variational_form,
    factor_operator, infer_form_requirements,
};

const POISSON_DIRICHLET: &str = r#"
module gx_a4.poisson_dirichlet;
model Poisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  property k = diffusivity(0);
  source f: VolumetricSource;
  equation balance on Omega { -div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(); }
}
"#;

const POISSON_NEUMANN: &str = r#"
module gx_a4.poisson_neumann;
model Poisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  property k = diffusivity(0);
  source f: VolumetricSource;
  equation balance on Omega { -div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { neumann u = 0; }
}
"#;

fn poisson_structure(source: &str) -> scientia::OperatorStructure {
    let compilation = compile_semantics(source, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    derive_operator_structure(&form, &requirements, &factorization, None).unwrap()
}

#[test]
fn poisson_like_fixture_is_symmetric_and_linear_with_dirichlet_removing_the_nullspace() {
    let structure = poisson_structure(POISSON_DIRICHLET);
    structure.validate().unwrap();
    assert_eq!(structure.trial_linearity, Linearity::Linear);
    assert_eq!(structure.form_symmetry, FormSymmetry::Symmetric);
    assert!(structure.nullspace_candidates.is_empty());
    assert!(!structure.saddle_point);
}

#[test]
fn poisson_like_fixture_without_a_dirichlet_condition_has_a_constant_nullspace_candidate() {
    let structure = poisson_structure(POISSON_NEUMANN);
    structure.validate().unwrap();
    assert_eq!(structure.trial_linearity, Linearity::Linear);
    assert_eq!(structure.nullspace_candidates.len(), 1);
    assert_eq!(
        structure.nullspace_candidates[0].kind,
        NullspaceKind::Constant
    );
}

const NONLINEAR_HEAT: &str = r#"
module gx_a4.nonlinear_heat;
model NonlinearHeat {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega { time_role = differential; };
  provider thermal_conductivity(T: Dimensionless) -> Dimensionless { differentiability = analytic_provided; }
  property k = thermal_conductivity(T);
  source Q: VolumetricHeatSource;
  equation energy on Omega { dt(T) - div(k * grad(T)) = Q; }
  boundary walls on boundary("walls") { dirichlet T = exact_T(t); }
}
"#;

#[test]
fn nonlinear_provider_wrapped_fixture_is_nonlinear_with_property_dependence_recorded() {
    let compilation = compile_semantics(NONLINEAR_HEAT, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "NonlinearHeat", "energy").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let structure = derive_operator_structure(&form, &requirements, &factorization, None).unwrap();
    structure.validate().unwrap();

    match &structure.trial_linearity {
        Linearity::Nonlinear { active } => assert!(!active.is_empty()),
        Linearity::Linear => panic!("expected a nonlinear trial dependence"),
    }
    assert_eq!(structure.property_dependence.len(), 1);
    let dependence = &structure.property_dependence[0];
    assert!(!dependence.depends_on.is_empty());
    assert_eq!(dependence.tangent, StructurePropertyTangent::External);
}

const STOKES: &str = r#"
module gx_a4.stokes;
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
"#;

#[test]
fn stokes_shape_mixed_fixture_is_a_saddle_point_with_a_pressure_nullspace_candidate() {
    let compilation = compile_semantics(STOKES, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_operator_system(
        &compilation.semantic,
        "Stokes",
        &["momentum", "incompressibility"],
    )
    .unwrap();
    let structure = derive_operator_structure_for_system(&system, None).unwrap();
    structure.validate().unwrap();

    assert!(structure.saddle_point);
    let pressure = compilation
        .semantic
        .models
        .iter()
        .find(|model| model.name == "Stokes")
        .unwrap()
        .symbols
        .iter()
        .find(|symbol| symbol.name == "pressure")
        .unwrap()
        .id;
    let pressure_candidate = structure
        .nullspace_candidates
        .iter()
        .find(|candidate| candidate.field == pressure)
        .expect("pressure should be a nullspace candidate");
    assert_eq!(pressure_candidate.kind, NullspaceKind::Constant);

    // The pressure/velocity coupling block is present while the pressure diagonal is not.
    assert!(
        structure.blocks.iter().any(|block| block.row == pressure
            && block.present
            && block.class != BlockClass::Unknown)
    );
    assert!(
        structure
            .blocks
            .iter()
            .any(|block| block.row == pressure && block.column == pressure && !block.present)
    );
}

const ADVECTION: &str = r#"
module gx_a4.advection;
model Advection {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: trial scalar H1(order=1) on Omega;
  field v: test scalar H1(order=1) on Omega;
  provider wind(x: selector) -> Dimensionless { shape = vector(2); differentiability = analytic_provided; }
  property beta = wind(0);
  form residual {
    cell(Omega): dot(beta, grad(u)) * v;
  }
}
"#;

#[test]
fn asymmetric_advection_fixture_is_never_reported_symmetric() {
    let compilation = compile_semantics(ADVECTION, &UnitRegistry::si_bootstrap()).unwrap();
    let form = compile_variational_form(&compilation.semantic, "Advection", "residual").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let structure = derive_operator_structure(&form, &requirements, &factorization, None).unwrap();
    structure.validate().unwrap();
    assert_ne!(structure.form_symmetry, FormSymmetry::Symmetric);
}
