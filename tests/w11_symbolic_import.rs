//! W11 S1: Malleus symbolic exports imported into Resolvent and cross-checked against the
//! Scientia source (round trip, JVP/VJP against Resolvent derivatives, interval enclosures).
use malleus::symbolic::{ExactConstant, SymbolicKernel, SymbolicNode, export_symbolic};
use malleus::{BinaryOp, DerivativeMode, Executable, UnaryOp, validate};
use quantitas::{Dimension, QuantityKindId, UnitRegistry};
use resolvent::{SymbolName, TermStore};
use scientia::scientific::{
    FrameSemantics, OutOfValidityPolicy, PropertyOutput, PropertySignature, TensorSymmetry,
    ValueShape,
};
use scientia::{
    CheckOutcome, DerivativeContract, EnclosureLowering, EnclosureOutcome, EquivalenceMethod,
    FvKernelRole, FvMethodIntent, KernelProvenance, NumericalFluxIntent, PositivityIntent,
    PropertyDefinition, PropertyDomain, PropertyEvidence, PropertyInput, PropertyLocality,
    PropertyModel, ReconstructionIntent, SampleDomain, SemanticModule, SourceTreatment,
    SymbolicVerificationEntry, SymbolicVerificationReport, compile_cell_functional,
    compile_conservation_law_system, compile_semantics, fv_kernel_sources, lower_conservation_law,
    lower_property_kernel, parse_expression, property_value_sources, verify_conservation_program,
    verify_export, verify_point_expression, verify_property_kernel,
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

fn source(name: &str) -> SymbolName {
    SymbolName::new("Scientia", name)
}

fn verified(outcome: &CheckOutcome) -> bool {
    matches!(outcome, CheckOutcome::Verified(_))
}

fn contained(entry: &SymbolicVerificationEntry) -> bool {
    entry.enclosures.len() == 3
        && entry
            .enclosures
            .iter()
            .all(|evidence| matches!(evidence.outcome, EnclosureOutcome::Contained { points, .. } if points > 0))
}

fn fv_module() -> SemanticModule {
    compile_semantics(FV_FIXTURE, &UnitRegistry::si_bootstrap())
        .unwrap()
        .semantic
}

fn fv_intent(numerical_flux: NumericalFluxIntent) -> FvMethodIntent {
    FvMethodIntent {
        numerical_flux,
        reconstruction: ReconstructionIntent::Constant,
        source: SourceTreatment::Pointwise,
        positivity: PositivityIntent::None,
    }
}

fn fv_entries(
    module: &SemanticModule,
    model: &str,
    flux: NumericalFluxIntent,
) -> Vec<SymbolicVerificationEntry> {
    let system = compile_conservation_law_system(module, model, "balance", "u").unwrap();
    let program = lower_conservation_law(module, &system, &fv_intent(flux), &[]).unwrap();
    let domain = SampleDomain::default()
        .with(SymbolName::new("Scientia.fv", "n0"), (-1.0, 1.0))
        .with(SymbolName::new("Scientia.fv", "n1"), (-1.0, 1.0));
    verify_conservation_program(module, &system, &program, &domain).unwrap()
}

fn kernel_export(kernel: &malleus::StructuredKernel) -> SymbolicKernel {
    export_symbolic(&Executable::reference(validate(kernel.clone()).unwrap())).unwrap()
}

#[test]
fn thermal_property_round_trips_with_derivatives_and_enclosures() {
    let definition = property("conductivity", &["T"], "0.5 + 0.25*T - 0.015625*T*T");
    let kernel = lower_property_kernel(&definition, &UnitRegistry::si_bootstrap()).unwrap();
    let entries = verify_property_kernel(&definition, &kernel, &SampleDomain::default()).unwrap();
    assert_eq!(entries.len(), 2, "value kernel and one tangent");
    let value = &entries[0];
    assert!(matches!(
        value.provenance,
        KernelProvenance::PropertyValue { .. }
    ));
    assert_eq!(value.rounded_literals, 0);
    assert_eq!(value.derivatives.len(), 2);
    assert_eq!(value.derivatives[0].mode, DerivativeMode::Jvp);
    assert_eq!(value.derivatives[1].mode, DerivativeMode::Vjp);
    assert!(value.derivatives.iter().all(|d| verified(&d.outcome)));
    assert_eq!(value.derivatives[0].independents, vec!["T".to_string()]);
    assert!(contained(value), "{:?}", value.enclosures);
    let tangent = &entries[1];
    assert!(matches!(
        &tangent.provenance,
        KernelProvenance::PropertyTangent { input, .. } if input == "T"
    ));
    assert!(contained(tangent), "{:?}", tangent.enclosures);
}

#[test]
fn elastic_and_rounded_and_transcendental_properties_round_trip() {
    let registry = UnitRegistry::si_bootstrap();
    let shear = property("shear_modulus", &["E", "nu"], "E / (2 * (1 + nu))");
    let kernel = lower_property_kernel(&shear, &registry).unwrap();
    let domain = SampleDomain::default()
        .with(source("E"), (1.0, 3.0))
        .with(source("nu"), (0.0, 0.4375));
    let entries = verify_property_kernel(&shear, &kernel, &domain).unwrap();
    assert_eq!(entries.len(), 3, "value kernel and two tangents");
    assert!(entries[0].derivatives.iter().all(|d| verified(&d.outcome)));
    assert_eq!(entries[0].derivatives[0].independents, vec!["E", "nu"]);
    // Interval lowerings support neg/add/sub/mul/fma only: division is a recorded refusal.
    assert!(
        entries[0]
            .enclosures
            .iter()
            .all(|evidence| matches!(&evidence.outcome, EnclosureOutcome::Refused { .. }))
    );

    // A decimal literal that is not binary64 is compared at the kernel's binary64 value.
    let rounded = property("rounded", &["T"], "0.1*T + 0.3");
    let kernel = lower_property_kernel(&rounded, &registry).unwrap();
    let entries = verify_property_kernel(&rounded, &kernel, &SampleDomain::default()).unwrap();
    assert_eq!(entries[0].rounded_literals, 2);
    assert!(contained(&entries[0]));

    let arrhenius = property("rate", &["T"], "2 * exp(-1500 / T)");
    let kernel = lower_property_kernel(&arrhenius, &registry).unwrap();
    let domain = SampleDomain::default().with(source("T"), (250.0, 400.0));
    let entries = verify_property_kernel(&arrhenius, &kernel, &domain).unwrap();
    assert_eq!(entries.len(), 2);
    assert!(entries[0].derivatives.iter().all(|d| verified(&d.outcome)));
}

#[test]
fn point_expression_round_trips_with_bundle_products() {
    let registry = UnitRegistry::si_bootstrap();
    let module = compile_semantics(POINT_SOURCE, &registry).unwrap().semantic;
    let misfit = compile_cell_functional(&module, "Example", "misfit").unwrap();
    let entry = verify_point_expression(&module, &misfit, &SampleDomain::default()).unwrap();
    assert!(matches!(
        entry.provenance,
        KernelProvenance::PointExpression { .. }
    ));
    assert_eq!(entry.outputs, 1);
    assert!(entry.derivatives.len() >= 2, "{:?}", entry.derivatives);
    assert!(entry.derivatives.iter().all(|d| verified(&d.outcome)));
    assert!(
        entry
            .derivatives
            .iter()
            .any(|d| d.mode == DerivativeMode::Vjp)
    );
    assert!(contained(&entry), "{:?}", entry.enclosures);
}

#[test]
fn fv_generic_fluxes_on_smooth_laws_round_trip() {
    let module = fv_module();
    for (model, flux) in [
        ("LinearAdvection1D", NumericalFluxIntent::Central),
        ("LinearAdvection1D", NumericalFluxIntent::LocalLaxFriedrichs),
        ("LinearAdvection2D", NumericalFluxIntent::LocalLaxFriedrichs),
        ("LinearAdvection2D", NumericalFluxIntent::Hll),
        ("Burgers1D", NumericalFluxIntent::Central),
    ] {
        let entries = fv_entries(&module, model, flux.clone());
        let roles = entries
            .iter()
            .map(|entry| match &entry.provenance {
                KernelProvenance::FiniteVolume { role, .. } => *role,
                other => panic!("{other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            roles,
            vec![
                FvKernelRole::PhysicalFluxNormal,
                FvKernelRole::CharacteristicSpeed,
                FvKernelRole::FaceFlux
            ]
        );
        for entry in &entries {
            assert!(
                entry.derivatives.iter().all(|d| verified(&d.outcome)),
                "{model} {flux:?}: {:?}",
                entry.derivatives
            );
        }
        let face = &entries[2];
        assert_eq!(face.derivatives.len(), 1, "{model} {flux:?}");
        assert_eq!(face.derivatives[0].independents, vec!["u_minus", "u_plus"]);
        if matches!(flux, NumericalFluxIntent::Central) {
            assert!(contained(face), "{model}: {:?}", face.enclosures);
        } else {
            assert!(face.import.piecewise > 0);
        }
        // The physical flux F·n is polynomial: every enclosure lowering admits it.
        assert!(
            contained(&entries[0]),
            "{model}: {:?}",
            entries[0].enclosures
        );
    }
}

#[test]
fn fv_kinked_face_flux_round_trips_and_records_the_d6_refusal() {
    let module = fv_module();
    let entries = fv_entries(
        &module,
        "Burgers1D",
        NumericalFluxIntent::LocalLaxFriedrichs,
    );
    let face = &entries[2];
    match &face.derivatives[0].outcome {
        CheckOutcome::Refused { code, .. } => {
            assert_eq!(code, "FV_NONSMOOTH_DERIVATIVE_REFUSED")
        }
        other => panic!("{other:?}"),
    }
    // The kinked speed |u| is a piecewise term: no interval lowering admits abs.
    assert!(
        face.enclosures
            .iter()
            .all(|evidence| matches!(evidence.outcome, EnclosureOutcome::Refused { .. }))
    );
    assert!(entries[0].derivatives.iter().all(|d| verified(&d.outcome)));
    assert_eq!(
        entries
            .iter()
            .map(|entry| entry.enclosures[0].lowering)
            .collect::<Vec<_>>(),
        vec![EnclosureLowering::Naive; 3]
    );
}

#[test]
fn perturbed_exports_are_detected() {
    let definition = property("conductivity", &["T"], "0.5 + 0.25*T - 0.015625*T*T");
    let kernel = lower_property_kernel(&definition, &UnitRegistry::si_bootstrap()).unwrap();
    let mut store = TermStore::new().unwrap();
    let sources = property_value_sources(&definition, &kernel, &mut store).unwrap();
    let mut export = kernel_export(&kernel.module.kernels[kernel.value_kernel]);
    verify_export(&mut store, &export, &sources, &SampleDomain::default()).unwrap();
    let constant = export
        .nodes
        .iter()
        .position(|node| matches!(node, SymbolicNode::Constant(c) if c.value() == 0.25))
        .unwrap();
    export.nodes[constant] = SymbolicNode::Constant(ExactConstant::new(0.25f64.next_up()).unwrap());
    let refusal =
        verify_export(&mut store, &export, &sources, &SampleDomain::default()).unwrap_err();
    assert_eq!(refusal.code, "SYMBOLIC_EQUIVALENCE_MISMATCH", "{refusal}");

    // Swapping the operands of the LLF jump changes the sign of the dissipation.
    let module = fv_module();
    let system =
        compile_conservation_law_system(&module, "LinearAdvection1D", "balance", "u").unwrap();
    let program = lower_conservation_law(
        &module,
        &system,
        &fv_intent(NumericalFluxIntent::LocalLaxFriedrichs),
        &[],
    )
    .unwrap();
    let mut store = TermStore::new().unwrap();
    let sources =
        fv_kernel_sources(&module, &system, &program, &program.face_flux, &mut store).unwrap();
    let mut export = kernel_export(&program.face_flux.kernel);
    verify_export(&mut store, &export, &sources, &SampleDomain::default()).unwrap();
    let jump = export
        .nodes
        .iter()
        .position(|node| {
            matches!(node, SymbolicNode::Binary { op: BinaryOp::Sub, lhs, rhs }
                if matches!(export.nodes[*lhs], SymbolicNode::Input { .. })
                    && matches!(export.nodes[*rhs], SymbolicNode::Input { .. }))
        })
        .unwrap();
    if let SymbolicNode::Binary { lhs, rhs, .. } = &mut export.nodes[jump] {
        std::mem::swap(lhs, rhs);
    }
    let refusal =
        verify_export(&mut store, &export, &sources, &SampleDomain::default()).unwrap_err();
    assert_eq!(refusal.code, "SYMBOLIC_EQUIVALENCE_MISMATCH", "{refusal}");
}

#[test]
fn fma_imports_exactly_and_unrepresentable_or_unmapped_inputs_refuse() {
    let definition = property("affine", &["x", "y", "z"], "x*y + z");
    let kernel = lower_property_kernel(&definition, &UnitRegistry::si_bootstrap()).unwrap();
    let mut store = TermStore::new().unwrap();
    let sources = property_value_sources(&definition, &kernel, &mut store).unwrap();
    let original = kernel_export(&kernel.module.kernels[kernel.value_kernel]);
    let input = |operand: usize| {
        original
            .inputs
            .iter()
            .position(|input| input.operand.index() == operand)
            .unwrap()
    };
    let mut fused = original.clone();
    fused.nodes = (0..original.inputs.len())
        .map(|input| SymbolicNode::Input { input })
        .collect();
    fused.nodes.push(SymbolicNode::Fma {
        lhs: input(0),
        rhs: input(1),
        addend: input(2),
        single_rounding: true,
    });
    fused.outputs[0].node = fused.nodes.len() - 1;
    let (statistics, _) =
        verify_export(&mut store, &fused, &sources, &SampleDomain::default()).unwrap();
    assert_eq!(statistics.fused_multiply_adds, 1);

    let mut floored = fused.clone();
    floored.nodes.push(SymbolicNode::Unary {
        op: UnaryOp::Floor,
        arg: input(0),
    });
    floored.outputs[0].node = floored.nodes.len() - 1;
    let refusal =
        verify_export(&mut store, &floored, &sources, &SampleDomain::default()).unwrap_err();
    assert_eq!(refusal.code, "SYMBOLIC_OP_UNREPRESENTABLE");

    let mut unmapped = sources.clone();
    unmapped
        .symbols
        .retain(|(operand, _), _| operand.index() != 2);
    let refusal =
        verify_export(&mut store, &original, &unmapped, &SampleDomain::default()).unwrap_err();
    assert_eq!(refusal.code, "SYMBOLIC_INPUT_UNMAPPED");
}

#[test]
fn verification_report_records_methods_and_is_identity_checked() {
    let registry = UnitRegistry::si_bootstrap();
    let definition = property("conductivity", &["T"], "0.5 + 0.25*T - 0.015625*T*T");
    let kernel = lower_property_kernel(&definition, &registry).unwrap();
    let mut entries =
        verify_property_kernel(&definition, &kernel, &SampleDomain::default()).unwrap();
    let module = fv_module();
    entries.extend(fv_entries(
        &module,
        "LinearAdvection1D",
        NumericalFluxIntent::LocalLaxFriedrichs,
    ));
    let report = SymbolicVerificationReport::new(entries);
    report.validate_identity().unwrap();
    assert_eq!(report.schema, "scientia-symbolic-verification/1");
    let value_digest = malleus::kernel_digest(&kernel.module.kernels[kernel.value_kernel]);
    assert!(report.verifies(&value_digest));
    assert!(report.verifies(&malleus::kernel_digest(&program_face_kernel(&module))));
    for entry in &report.entries {
        assert!(matches!(
            entry.round_trip,
            EquivalenceMethod::CanonicalEquality | EquivalenceMethod::ExactSamples { .. }
        ));
    }
    let json = serde_json::to_string(&report).unwrap();
    assert!(json.contains(r#""kind":"property_value""#), "{json}");
    assert!(json.contains(r#""kind":"finite_volume""#));
    assert!(json.contains(r#""lowering":"naive""#));
    let decoded: SymbolicVerificationReport = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, report);
    let mut tampered = report.clone();
    tampered.entries.pop();
    assert_eq!(
        tampered.validate_identity().unwrap_err().code,
        "SYMBOLIC_REPORT_IDENTITY"
    );
}

fn program_face_kernel(module: &SemanticModule) -> malleus::StructuredKernel {
    let system =
        compile_conservation_law_system(module, "LinearAdvection1D", "balance", "u").unwrap();
    lower_conservation_law(
        module,
        &system,
        &fv_intent(NumericalFluxIntent::LocalLaxFriedrichs),
        &[],
    )
    .unwrap()
    .face_flux
    .kernel
}
