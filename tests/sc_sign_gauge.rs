//! `scientia-operator-structure/2` (SC, `sinbad/ARCHITECTURE.md` §7): per-block symmetry,
//! per-pair transpose relations, and the signed-graph residual gauge.

use quantitas::UnitRegistry;
use scientia::{
    BlockClass, FormSymmetry, OPERATOR_STRUCTURE_SCHEMA, OperatorStructure, SymbolId,
    compile_operator_system, compile_semantics, derive_operator_structure_for_system,
};

/// A repository-local copy of `sinbad/physics/corpus/13-mixed-darcy.res`: a two-row mixed
/// system whose flux row is `K⁻¹ q · w - p div(w)` and whose pressure row is `div(q) v`.
const MIXED_DARCY: &str = r#"
module sc.mixed_darcy;

model MixedDarcy {
    domain Omega { dimension = 3; coordinates = cartesian; }

    field flux: unknown vector(3) HDiv(order=0) on Omega;
    field pressure: unknown scalar L2(order=0) on Omega;

    provider permeability_tensor(material: selector) -> Permeability { differentiability = analytic_provided; }
    provider dynamic_viscosity(material: selector) -> DynamicViscosity { differentiability = analytic_provided; }
    provider inverse(permeability: Permeability) -> Permeability { differentiability = analytic_provided; }

    property permeability = permeability_tensor(0);
    property viscosity = dynamic_viscosity(0);
    source source_term: MassSource;
    source body_force: BodyForce;

    constitutive mobility_inverse = viscosity * inverse(permeability);

    equation darcy_law on Omega {
        mobility_inverse * flux + grad(pressure) = body_force;
    }

    equation mass_balance on Omega {
        div(flux) = source_term;
    }

    boundary impermeable on boundary("walls") {
        neumann flux = 0;
    }
}
"#;

/// A one-way coupled pair: `w`'s equation depends on `u`, `u`'s does not depend on `w`.
const ONE_SIDED: &str = r#"
module sc.one_sided;

model OneSided {
    domain Omega { dimension = 2; coordinates = cartesian; }
    field u: unknown scalar H1(order=1) on Omega;
    field w: unknown scalar H1(order=1) on Omega;
    source f: VolumetricSource;
    equation first on Omega { -div(grad(u)) = f; }
    equation second on Omega { -div(grad(w)) = u; }
    boundary walls_u on boundary("walls") { dirichlet u = 0; }
    boundary walls_w on boundary("walls") { dirichlet w = 0; }
}
"#;

/// Two rows whose first diagonal block is convective.
const CONVECTIVE: &str = r#"
module sc.convective;

model Convective {
    domain Omega { dimension = 2; coordinates = cartesian; }
    field c: unknown scalar H1(order=1) on Omega;
    field w: unknown scalar H1(order=1) on Omega;
    provider wind(x: selector) -> Dimensionless { shape = vector(2); differentiability = analytic_provided; }
    property beta = wind(0);
    equation transport on Omega { dot(beta, grad(c)) = w; }
    equation second on Omega { -div(grad(w)) = c; }
    boundary walls_c on boundary("walls") { dirichlet c = 0; }
    boundary walls_w on boundary("walls") { dirichlet w = 0; }
}
"#;

fn structure_of(source: &str, model: &str, equations: &[&str]) -> OperatorStructure {
    let compilation = compile_semantics(source, &UnitRegistry::si_bootstrap()).unwrap();
    let system = compile_operator_system(&compilation.semantic, model, equations).unwrap();
    let structure = derive_operator_structure_for_system(&system, None).unwrap();
    structure.validate().unwrap();
    assert_eq!(structure.schema, OPERATOR_STRUCTURE_SCHEMA);
    assert_eq!(OPERATOR_STRUCTURE_SCHEMA, "scientia-operator-structure/2");
    structure
}

#[test]
fn mixed_darcy_gauge_flips_the_pressure_row() {
    let structure = structure_of(MIXED_DARCY, "MixedDarcy", &["darcy_law", "mass_balance"]);
    let (flux, pressure) = (SymbolId(0), SymbolId(1));
    // /1 fix: the flux mass block is classified from the integrals in which `flux` is active,
    // not from the whole row's test evaluations (which include the constraint term's
    // divergence), so it is no longer reported convective.
    let diagonal = structure
        .blocks
        .iter()
        .find(|block| block.row == flux && block.column == flux)
        .unwrap();
    assert_eq!(diagonal.class, BlockClass::Reaction);
    assert_eq!(
        structure.block_symmetry,
        vec![(flux, FormSymmetry::Symmetric)],
        "only the flux row has a diagonal block, and it is symmetric"
    );
    assert_eq!(structure.transpose_relation.len(), 2);
    for relation in &structure.transpose_relation {
        assert_eq!(
            relation.sigma,
            Some(-1),
            "-p div(w) is minus the transpose of div(q) v"
        );
    }
    let gauge = structure
        .sign_gauge
        .as_ref()
        .unwrap_or_else(|| panic!("{:?}", structure.sign_gauge_reason));
    assert_eq!(gauge.signs, vec![(flux, 1), (pressure, -1)]);
    assert_eq!(gauge.proof.edges.len(), 1);
    assert_eq!(gauge.proof.spanning_forest, gauge.proof.edges);
    assert!(structure.sign_gauge_reason.is_none());
    // C5.4's unsigned system claim stays `Unknown`; the gauge is the additive `/2` claim.
    assert_eq!(structure.form_symmetry, FormSymmetry::Unknown);
}

#[test]
fn one_sided_coupling_has_no_gauge_and_says_why() {
    let structure = structure_of(ONE_SIDED, "OneSided", &["first", "second"]);
    assert_eq!(
        structure.block_symmetry,
        vec![
            (SymbolId(0), FormSymmetry::Symmetric),
            (SymbolId(1), FormSymmetry::Symmetric)
        ]
    );
    assert!(structure.transpose_relation.is_empty());
    assert!(structure.sign_gauge.is_none());
    let reason = structure.sign_gauge_reason.as_deref().unwrap();
    assert!(reason.contains("one-sided"), "{reason}");
}

#[test]
fn convective_diagonal_block_has_no_gauge_and_says_why() {
    let structure = structure_of(CONVECTIVE, "Convective", &["transport", "second"]);
    assert!(structure.sign_gauge.is_none());
    let reason = structure.sign_gauge_reason.as_deref().unwrap();
    assert!(reason.contains("convective"), "{reason}");
}

#[test]
fn gauge_is_part_of_the_identity_and_round_trips() {
    let structure = structure_of(MIXED_DARCY, "MixedDarcy", &["darcy_law", "mass_balance"]);
    let json = serde_json::to_string(&structure).unwrap();
    let decoded: OperatorStructure = serde_json::from_str(&json).unwrap();
    decoded.validate().unwrap();
    assert_eq!(decoded, structure);

    let mut tampered = structure.clone();
    tampered.sign_gauge.as_mut().unwrap().signs[1].1 = 1;
    let error = tampered.validate().unwrap_err().to_string();
    assert!(error.contains("STRUCTURE_IDENTITY_MISMATCH"), "{error}");

    let mut inconsistent = structure.clone();
    inconsistent.sign_gauge = None;
    let error = inconsistent.validate().unwrap_err().to_string();
    assert!(error.contains("STRUCTURE_INVALID"), "{error}");
}

/// Opt-in corpus sweep (`SINBAD_WORKSPACE`, same convention as `tests/gx_a4_structure.rs`):
/// Darcy gauges; Stokes does not, because its viscous stress reaches the tensor program as an
/// opaque constitutive input, and the reason names that diagonal block.
#[test]
fn sinbad_corpus_darcy_gauges_and_stokes_reports_the_opaque_momentum_block() {
    let Some(workspace) = std::env::var_os("SINBAD_WORKSPACE") else {
        eprintln!("skipping: SINBAD_WORKSPACE is not set; corpus sweep is opt-in");
        return;
    };
    let dir = std::path::PathBuf::from(workspace).join("sinbad/physics/corpus");
    let darcy = std::fs::read_to_string(dir.join("13-mixed-darcy.res")).unwrap();
    let structure = structure_of(&darcy, "MixedDarcy", &["darcy_law", "mass_balance"]);
    let gauge = structure.sign_gauge.as_ref().unwrap();
    assert_eq!(gauge.signs, vec![(SymbolId(0), 1), (SymbolId(1), -1)]);

    let stokes = std::fs::read_to_string(dir.join("25-stokes.res")).unwrap();
    let structure = structure_of(&stokes, "StokesFlow", &["momentum", "incompressibility"]);
    assert!(
        structure
            .transpose_relation
            .iter()
            .all(|r| r.sigma == Some(-1))
    );
    assert_eq!(
        structure.block_symmetry,
        vec![(SymbolId(0), FormSymmetry::Unknown)]
    );
    assert!(structure.sign_gauge.is_none());
    let reason = structure.sign_gauge_reason.as_deref().unwrap();
    assert!(reason.contains("row 0 is Unknown"), "{reason}");
}
