//! GX-A4 `scientia-operator-structure/1` (contract C5.4) fixtures.

use quantitas::UnitRegistry;
use scientia::{
    BlockClass, FormSymmetry, Linearity, NullspaceKind, StructurePropertyTangent,
    VerificationObligationKind, compile_operator_system, compile_semantics,
    compile_variational_form, derive_operator_structure, derive_operator_structure_for_system,
    derive_variational_form, derive_verification_profiles, factor_operator,
    infer_form_requirements,
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

/// Deliberately reaches into the sinbad checkout, matching the same opt-in convention as
/// `tests/binding_slots.rs`'s `sinbad_corpus_dir` (see its doc comment): skipped whenever
/// `SINBAD_WORKSPACE` is unset, so the hermetic gate never depends on it. Exercises E6's
/// SV2-B corpus models -- `25-stokes.res` and `13-mixed-darcy.res` -- through the same
/// `OperatorSystem`/`OperatorStructure`/`VerificationProfile` pipeline the inline `STOKES`
/// fixture above already covers, so a regression in the real corpus grammar or providers is
/// caught even when the inline fixture still passes.
#[test]
fn sinbad_saddle_point_corpus_models_derive_structure_and_inf_sup_obligations() {
    let Some(dir) = sinbad_corpus_dir() else {
        return;
    };
    let cases: &[(&str, &str, &[&str], &str)] = &[
        (
            "25-stokes.res",
            "StokesFlow",
            &["momentum", "incompressibility"],
            "Taylor-Hood",
        ),
        (
            "13-mixed-darcy.res",
            "MixedDarcy",
            &["darcy_law", "mass_balance"],
            "RT0-P0",
        ),
    ];
    for (file, model, equations, expected_pair) in cases {
        let path = dir.join(file);
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
        let compilation = compile_semantics(&source, &UnitRegistry::si_bootstrap())
            .unwrap_or_else(|diagnostics| panic!("{file} failed to elaborate: {diagnostics:?}"));

        let system = compile_operator_system(&compilation.semantic, model, equations)
            .unwrap_or_else(|error| panic!("{file} failed to compile an operator system: {error}"));
        let structure = derive_operator_structure_for_system(&system, None)
            .unwrap_or_else(|error| panic!("{file} failed to derive operator structure: {error}"));
        structure.validate().unwrap();
        assert!(
            structure.saddle_point,
            "{file}: mixed velocity/flux-pressure system must derive saddle_point = true"
        );
        assert!(
            structure
                .nullspace_candidates
                .iter()
                .any(|candidate| candidate.kind == NullspaceKind::Constant),
            "{file}: the pressure field must derive a constant nullspace candidate"
        );

        let profiles = derive_verification_profiles(&compilation);
        let profile = profiles
            .iter()
            .find(|profile| profile.model == *model)
            .unwrap_or_else(|| panic!("{file}: no verification profile for model {model}"));
        profile.validate().unwrap();
        let pair = profile
            .obligations
            .iter()
            .find_map(|obligation| match &obligation.kind {
                VerificationObligationKind::InfSup { pair } => Some(pair.as_str()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("{file}: no InfSup obligation derived from @inf_sup"));
        assert_eq!(pair, *expected_pair);
        assert!(
            !profile.obligations.iter().any(|obligation| matches!(
                &obligation.kind,
                VerificationObligationKind::Unsupported { name } if name == "inf_sup"
            )),
            "{file}: @inf_sup must not fall back to Unsupported"
        );
    }
}

/// The Sinbad corpus directory, only when the workspace coordinator opted in through
/// `SINBAD_WORKSPACE`; `None` skips the cross-repository sweep in hermetic runs. Mirrors
/// `tests/binding_slots.rs`'s helper of the same name.
fn sinbad_corpus_dir() -> Option<std::path::PathBuf> {
    let Some(workspace) = std::env::var_os("SINBAD_WORKSPACE") else {
        eprintln!("skipping: SINBAD_WORKSPACE is not set; corpus sweep is opt-in");
        return None;
    };
    let dir = std::path::PathBuf::from(workspace).join("sinbad/physics/corpus");
    assert!(
        dir.is_dir(),
        "SINBAD_WORKSPACE is set but {} is not a directory",
        dir.display()
    );
    Some(dir)
}
