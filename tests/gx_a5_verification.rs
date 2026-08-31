//! GX-A5 `scientia-verification-obligation/2` (contract C6.1) typed-kind fixtures.

use quantitas::UnitRegistry;
use scientia::{
    ConservationRelation, ExactSolutionSource, OrderBasis, RefinementAxis,
    VERIFICATION_OBLIGATION_SCHEMA, VerificationObligationKind, compile_semantics,
    derive_verification_profiles,
};

fn obligations(source: &str) -> Vec<scientia::VerificationObligation> {
    let compilation = compile_semantics(source, &UnitRegistry::si_bootstrap())
        .unwrap_or_else(|diagnostics| panic!("model failed to elaborate: {diagnostics:?}"));
    let profiles = derive_verification_profiles(&compilation);
    let [profile] = profiles.as_slice() else {
        panic!("expected exactly one model");
    };
    profile.validate().expect("derived profile is canonical");
    profile.obligations.clone()
}

fn symbol_named(model: &scientia::SemanticModel, name: &str) -> scientia::SymbolId {
    model
        .symbols
        .iter()
        .find(|symbol| symbol.name == name)
        .unwrap_or_else(|| panic!("no symbol named {name}"))
        .id
}

#[test]
fn schema_is_bumped_to_version_two() {
    assert_eq!(
        VERIFICATION_OBLIGATION_SCHEMA,
        "scientia-verification-obligation/2"
    );
}

const MMS_MODEL: &str = r#"
module gx_a5.mms;
model Diffusion {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega { time_role = differential; };
  provider exact_T(t: Time) -> Dimensionless { differentiability = analytic_provided; }
  source Q: VolumetricHeatSource;
  equation energy on Omega { dt(T) = Q; }
  boundary walls on boundary("walls") { dirichlet T = exact_T(t); }
  @mms(field = T);
}
"#;

#[test]
fn mms_with_no_authored_exact_expression_yields_a_provider_slot() {
    let compilation = compile_semantics(MMS_MODEL, &UnitRegistry::si_bootstrap()).unwrap();
    let field = symbol_named(&compilation.semantic.models[0], "T");
    let kinds = obligations(MMS_MODEL);
    let manufactured = kinds
        .iter()
        .find_map(|obligation| match &obligation.kind {
            VerificationObligationKind::ManufacturedSolution { field: f, exact } if *f == field => {
                Some(exact.clone())
            }
            _ => None,
        })
        .expect("expected a ManufacturedSolution obligation for T");
    assert_eq!(
        manufactured,
        ExactSolutionSource::Slot("provider/exact_T".into())
    );

    let convergence = kinds
        .iter()
        .find_map(|obligation| match &obligation.kind {
            VerificationObligationKind::Convergence {
                field: f,
                axis,
                order_basis,
            } if *f == field => Some((*axis, *order_basis)),
            _ => None,
        })
        .expect("expected a Convergence obligation for T");
    assert_eq!(convergence.0, RefinementAxis::MeshSize);
    assert_eq!(convergence.1, OrderBasis::SpaceOrderPlusOne);

    assert!(kinds.iter().all(|obligation| !matches!(
        obligation.kind,
        VerificationObligationKind::Unsupported { .. }
    )));
}

const PATCH_TEST_MODEL: &str = r#"
module gx_a5.patch_test;
model LinearElasticity {
  domain Omega { dimension = 3; coordinates = cartesian; }
  field displacement: unknown vector(3) H1(order=1) on Omega;
  property lambda = lame_lambda(0);
  property mu = lame_mu(0);
  source body_force: MechanicalBodyForce;
  constitutive strain = sym_grad(displacement);
  constitutive stress = lambda * trace(strain) * identity(3) + 2 * mu * strain;
  equation momentum on Omega { -div(stress) = body_force; }
  boundary clamp on boundary("clamp") { dirichlet displacement = [0, 0, 0]; }
  @patch_test(field = displacement);
  @rigid_body_modes(count = 6);
}
"#;

#[test]
fn patch_test_and_rigid_body_modes_resolve_the_companion_field() {
    let compilation = compile_semantics(PATCH_TEST_MODEL, &UnitRegistry::si_bootstrap()).unwrap();
    let field = symbol_named(&compilation.semantic.models[0], "displacement");
    let kinds = obligations(PATCH_TEST_MODEL);

    assert!(kinds.iter().any(|obligation| matches!(
        &obligation.kind,
        VerificationObligationKind::PatchTest { field: f } if *f == field
    )));
    assert!(kinds.iter().any(|obligation| matches!(
        &obligation.kind,
        VerificationObligationKind::RigidBodyModes { field: f, count: 6 } if *f == field
    )));
    assert!(kinds.iter().all(|obligation| !matches!(
        obligation.kind,
        VerificationObligationKind::Unsupported { .. }
    )));
}

const TEMPORAL_CONVERGENCE_MODEL: &str = r#"
module gx_a5.temporal_convergence;
model TransientDiffusion {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: state scalar H1(order=1) on Omega { time_role = differential; };
  source f: VolumetricSource;
  equation evolution on Omega { dt(u) = f; }
  @spatial_convergence(order = 2);
  @temporal_convergence(order = 2);
}
"#;

#[test]
fn spatial_and_temporal_convergence_annotations_use_the_sole_state_field_and_declared_order() {
    let compilation =
        compile_semantics(TEMPORAL_CONVERGENCE_MODEL, &UnitRegistry::si_bootstrap()).unwrap();
    let field = symbol_named(&compilation.semantic.models[0], "u");
    let kinds = obligations(TEMPORAL_CONVERGENCE_MODEL);

    assert!(kinds.iter().any(|obligation| matches!(
        &obligation.kind,
        VerificationObligationKind::TemporalConvergence { field: f, order_basis }
            if *f == field && *order_basis == OrderBasis::Declared(2.0)
    )));
    assert!(kinds.iter().any(|obligation| matches!(
        &obligation.kind,
        VerificationObligationKind::Convergence { field: f, axis: RefinementAxis::MeshSize, order_basis }
            if *f == field && *order_basis == OrderBasis::Declared(2.0)
    )));
}

const INF_SUP_MODEL: &str = r#"
module gx_a5.inf_sup;
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
  @inf_sup(pair = "Taylor-Hood");
}
"#;

#[test]
fn inf_sup_annotation_carries_the_declared_pairing() {
    let kinds = obligations(INF_SUP_MODEL);
    assert!(kinds.iter().any(|obligation| matches!(
        &obligation.kind,
        VerificationObligationKind::InfSup { pair } if pair == "Taylor-Hood"
    )));
}

const JVP_TAYLOR_MODEL: &str = r#"
module gx_a5.jvp_taylor;
model Electrothermal {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field V: unknown scalar H1(order=1) on Omega;
  field T: state scalar H1(order=1) on Omega { time_role = differential; };
  provider electrical_conductivity(T: Dimensionless) -> Dimensionless { differentiability = symbolic; }
  property sigma = electrical_conductivity(T);
  source joule: VolumetricHeatSource;
  equation electrical on Omega { div(sigma * grad(V)) = 0; }
  equation thermal on Omega { dt(T) = joule; }
  @jvp_taylor(block = electrical);
}
"#;

#[test]
fn jvp_taylor_annotation_produces_a_named_derivative_taylor_obligation() {
    let kinds = obligations(JVP_TAYLOR_MODEL);
    let named = kinds
        .iter()
        .filter(|obligation| {
            matches!(
                &obligation.kind,
                VerificationObligationKind::DerivativeTaylor { block: Some(_), .. }
            )
        })
        .count();
    // One from the `@jvp_taylor(block = electrical)` annotation, plus one auto-generated per
    // equation with active inputs (`electrical`, `thermal`) -- three `DerivativeTaylor`
    // obligations in total, all block-named.
    assert_eq!(named, 3);
}

const CONSERVATION_MODEL: &str = r#"
module gx_a5.conservation;
model CahnHilliard {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field c: state scalar H1(order=1) on Omega { time_role = differential; };
  field mu: unknown scalar H1(order=1) on Omega;
  source noise: VolumetricSource;
  equation cahn_hilliard on Omega { dt(c) = noise; }
  equation chemical_potential on Omega { mu = c; }
  @conservation(quantity = c);
}
"#;

#[test]
fn conservation_with_an_explicit_quantity_argument_is_typed() {
    let compilation = compile_semantics(CONSERVATION_MODEL, &UnitRegistry::si_bootstrap()).unwrap();
    let model = &compilation.semantic.models[0];
    let field = symbol_named(model, "c");
    let expected_expr = model
        .expressions
        .iter()
        .position(|expression| {
            matches!(&expression.kind, scientia::SemanticExprKind::Symbol { symbol } if *symbol == field)
                && model.declarations.iter().any(|declaration| {
                    matches!(&declaration.kind, scientia::SemanticDeclarationKind::Verification { arguments }
                        if arguments.get("quantity").is_some_and(|id| id.index() == expression.id.index()))
                })
        });
    assert!(
        expected_expr.is_some(),
        "the quantity argument should reference an arena expression"
    );

    let kinds = obligations(CONSERVATION_MODEL);
    assert!(kinds.iter().any(|obligation| matches!(
        &obligation.kind,
        VerificationObligationKind::Conservation {
            relation: ConservationRelation::GlobalBalance,
            ..
        }
    )));
}

/// A bare conservation-family annotation (`@energy_balance()`, `@charge_conservation()`, ...)
/// names no expression at all. `Conservation::quantity` is a real `ExprId`, and Scientia does
/// not fabricate one to point at an arbitrarily chosen model expression; this is reported as an
/// intentional deviation in the landing report, not a bug.
const BARE_ENERGY_BALANCE_MODEL: &str = r#"
module gx_a5.bare_conservation;
model Elastodynamics {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field displacement: state vector(2) H1(order=1) on Omega { time_role = differential; };
  source body_force: MechanicalBodyForce;
  equation momentum on Omega { dt(displacement) = body_force; }
  observable kinetic_energy { integrate(displacement); }
  @energy_balance();
}
"#;

#[test]
fn a_bare_conservation_annotation_stays_an_explicit_refusal_not_a_guess() {
    let kinds = obligations(BARE_ENERGY_BALANCE_MODEL);
    let unsupported = kinds.iter().find(|obligation| {
        matches!(
            &obligation.kind,
            VerificationObligationKind::Unsupported { name } if name == "energy_balance"
        )
    });
    let obligation = unsupported.expect("bare @energy_balance() should be an explicit refusal");
    assert_eq!(
        obligation.unsupported.as_ref().unwrap().code,
        "VERIFY_UNSUPPORTED_ANNOTATION"
    );
    assert!(!kinds.iter().any(|obligation| matches!(
        &obligation.kind,
        VerificationObligationKind::Conservation { .. }
    )));
}

/// A genuinely semantic-free annotation (contract C6.1's own examples) has no typed carrier and
/// stays `VERIFY_UNSUPPORTED_ANNOTATION`.
const SHOCK_TUBE_MODEL: &str = r#"
module gx_a5.shock_tube;
model CompressibleEuler {
  domain Omega { dimension = 1; coordinates = cartesian; }
  field density: state scalar DG(order=0) on Omega { time_role = differential; };
  equation continuity on Omega { dt(density) = 0; }
  @shock_tube();
}
"#;

#[test]
fn shock_tube_has_no_safe_generator_and_stays_unsupported() {
    let kinds = obligations(SHOCK_TUBE_MODEL);
    assert!(kinds.iter().any(|obligation| matches!(
        &obligation.kind,
        VerificationObligationKind::Unsupported { name } if name == "shock_tube"
    )));
}

/// Deliberately reaches into the sinbad checkout, matching the same opt-in convention as
/// `tests/binding_slots.rs`'s `sinbad_corpus_dir` (see its doc comment): skipped whenever
/// `SINBAD_WORKSPACE` is unset, so the hermetic gate never depends on it.
#[test]
fn corpus_sweep_reports_remaining_unsupported_standard_annotations() {
    let Some(workspace) = std::env::var_os("SINBAD_WORKSPACE") else {
        eprintln!("skipping: SINBAD_WORKSPACE is not set; corpus sweep is opt-in");
        return;
    };
    let dir = std::path::PathBuf::from(workspace).join("sinbad/physics/corpus");
    assert!(
        dir.is_dir(),
        "SINBAD_WORKSPACE is set but {} is not a directory",
        dir.display()
    );
    let mut entries = std::fs::read_dir(&dir)
        .unwrap_or_else(|error| panic!("reading {}: {error}", dir.display()))
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "res"))
        .collect::<Vec<_>>();
    entries.sort();

    // Annotations the GX-A5 task named explicitly as required to stop being refused. `mms` and
    // `limiting_case` are intentionally absent below where the corpus grammar gives them no
    // resolvable argument (see the landing report); every occurrence actually present in the
    // corpus for the rest must map to a typed kind.
    const STANDARD: &[&str] = &[
        "mms",
        "spatial_convergence",
        "temporal_convergence",
        "patch_test",
        "rigid_body_modes",
        "jvp_taylor",
        "inf_sup",
        "conservation",
        "interface_conservation",
    ];

    let mut remaining = 0usize;
    let mut standard_unsupported = Vec::new();
    for path in &entries {
        let source = std::fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
        let Ok(compilation) = compile_semantics(&source, &UnitRegistry::si_bootstrap()) else {
            eprintln!(
                "skipping {}: does not elaborate on this branch",
                path.display()
            );
            continue;
        };
        for profile in derive_verification_profiles(&compilation) {
            profile.validate().expect("derived profile is canonical");
            for obligation in &profile.obligations {
                let VerificationObligationKind::Unsupported { name } = &obligation.kind else {
                    continue;
                };
                remaining += 1;
                if STANDARD.contains(&name.as_str()) {
                    standard_unsupported.push(format!("{}: @{name}", path.display()));
                }
            }
        }
    }
    eprintln!("VERIFY_UNSUPPORTED_ANNOTATION remaining across the corpus: {remaining}");
    assert!(
        standard_unsupported.is_empty(),
        "standard annotations must not be refused: {standard_unsupported:#?}"
    );
}
