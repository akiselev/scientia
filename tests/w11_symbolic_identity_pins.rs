//! Identity pins recorded before W11 S1 (symbolic import): the kernels the round-trip checks
//! read must keep their identities and serialized bytes while the importer is added beside them.
use quantitas::{Dimension, QuantityKindId, UnitRegistry};
use scientia::scientific::{
    FrameSemantics, OutOfValidityPolicy, PropertyOutput, PropertySignature, TensorSymmetry,
    ValueShape,
};
use scientia::{
    DerivativeContract, FvMethodIntent, NumericalFluxIntent, PositivityIntent, PropertyDefinition,
    PropertyDomain, PropertyEvidence, PropertyInput, PropertyLocality, PropertyModel,
    ReconstructionIntent, SourceTreatment, compile_cell_functional,
    compile_conservation_law_system, compile_semantics, lower_conservation_law,
    lower_property_kernel, parse_expression,
};

const FV_FIXTURE: &str = include_str!("fixtures/scalar-conservation.res");
const POINT_SOURCE: &str = r#"
module w11.symbolic;
model Example {
 domain body { dimension = 2; coordinates = cartesian; }
 field u: unknown scalar H1(order=1) on body;
 input field observed: Dimensionless on body;
 source f: VolumetricSource;
 equation balance on body { -div(grad(u)) = f; }
 objective misfit { minimize integrate(0.5 * (u-observed) * (u-observed)); }
}
"#;

fn property(id: &str, inputs: &[&str], expression: &str) -> PropertyDefinition {
    PropertyDefinition {
        signature: PropertySignature {
            id: id.into(),
            inputs: inputs
                .iter()
                .map(|name| PropertyInput {
                    name: (*name).into(),
                    quantity_kind: QuantityKindId::new("Dimensionless"),
                    dimension: Dimension::DIMENSIONLESS,
                    shape: ValueShape::Scalar,
                    physical_min: None,
                    physical_max: None,
                    nominal: None,
                })
                .collect(),
            output: PropertyOutput {
                quantity_kind: QuantityKindId::new("Dimensionless"),
                dimension: Dimension::DIMENSIONLESS,
                shape: ValueShape::Scalar,
                symmetry: TensorSymmetry::None,
                frame: FrameSemantics::Scalar,
            },
            locality: PropertyLocality::Pointwise,
            differentiability: DerivativeContract::Symbolic,
        },
        model: PropertyModel::Expression(parse_expression(expression).unwrap()),
        domain: PropertyDomain {
            physical_bounds: vec![],
            validity_bounds: vec![],
            phase_constraints: vec![],
            composition_constraints: vec![],
            assumptions: vec![],
            out_of_validity: OutOfValidityPolicy::Warn,
        },
        evidence: PropertyEvidence {
            sources: vec![],
            dataset_digest: None,
            fit_digest: None,
            uncertainty: None,
            notes: Default::default(),
        },
    }
}

fn bytes(value: &impl serde::Serialize) -> String {
    blake3::hash(&serde_json::to_vec(value).unwrap())
        .to_hex()
        .to_string()
}

#[test]
fn property_point_and_fv_identities_are_unchanged() {
    let registry = UnitRegistry::si_bootstrap();
    let thermal = lower_property_kernel(
        &property("conductivity", &["T"], "0.5 + T/4 - T*T/64"),
        &registry,
    )
    .unwrap();
    let shear = lower_property_kernel(
        &property("shear_modulus", &["E", "nu"], "E / (2 * (1 + nu))"),
        &registry,
    )
    .unwrap();
    let module = compile_semantics(POINT_SOURCE, &registry).unwrap().semantic;
    let misfit = compile_cell_functional(&module, "Example", "misfit").unwrap();
    let fv = compile_semantics(FV_FIXTURE, &registry).unwrap().semantic;
    let system = compile_conservation_law_system(&fv, "LinearAdvection1D", "balance", "u").unwrap();
    let llf = lower_conservation_law(
        &fv,
        &system,
        &FvMethodIntent {
            numerical_flux: NumericalFluxIntent::LocalLaxFriedrichs,
            reconstruction: ReconstructionIntent::Constant,
            source: SourceTreatment::Pointwise,
            positivity: PositivityIntent::None,
        },
        &[],
    )
    .unwrap();
    let observed = [
        thermal.identity.hex.clone(),
        bytes(&thermal.module),
        shear.identity.hex.clone(),
        bytes(&shear.module),
        misfit.artifact_digest.hex.clone(),
        bytes(&misfit),
        llf.artifact_digest.hex.clone(),
        llf.face_flux.digest.hex.clone(),
        bytes(&llf),
    ];
    let pinned = [
        "f5876b356d03ed557892a48fc377a66ba09376ad49569c73dafaeff79ebd8a6e",
        "af095d0248d030499deaaabf52dd19bc98ee489a5f5c244895af7788b740984f",
        "bf0ad8cb8b840e291fb048e1aff2e9d1bbd690a954be6e131dffd59ffe0597a4",
        "9a0f7f213d05f6b46bb49982d53dd1735d799944373105f972fe95626f4e77d1",
        "f372eb71bfcd5b3b4b15c46654cc05824c50003ad9f29b9fc5e098dc640e2ee9",
        "6431460473b32ae04861728ca63f410c7be03932424fd6994d85c8a8b527a786",
        "6f2ba12a44b1df8f628ab8482067253e93a9c3d96f81b78b53156a22b0370346",
        "cadb5f5800935cc4d8b801ca8263f88ce0cc8b54ae8fb947988a92a6a7b05584",
        "360e670d6ac873b5f595852f486877fda05d6d71a2fc1e04f32c5f66c7a641ee",
    ];
    assert_eq!(observed, pinned);
}
