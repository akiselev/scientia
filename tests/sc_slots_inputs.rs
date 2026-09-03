//! SC runner-free packages (`sinbad/ARCHITECTURE.md` §3.3, §2.1): defined-source slot
//! classification, `input field` / `input value` declarations, and `SourceLocator`.

use quantitas::UnitRegistry;
use scientia::{
    ModuleDigest, NoImports, SemanticDeclarationKind, SemanticRole, SlotKind, SlotStatus,
    compile_semantics, derive_binding_slots, format_scientific_module, parse_scientific_module,
    resolve_modules, semantic_digest,
};

const INPUTS: &str = r#"
module sc.inputs;

model HeatConduction {
    domain body { dimension = 3; coordinates = cartesian; }
    field T: state scalar H1(order=1) on body { time_role = differential; };
    provider conductivity(T: Dimensionless) -> ThermalConductivity { differentiability = symbolic; }
    property k = conductivity(T);
    input field Q: VolumetricHeatSource on body;
    input value ambient: ThermodynamicTemperature;
    source joule = k * T;
    source external_load: VolumetricHeatSource;
    equation energy on body { dt(T) - div(k * grad(T)) = Q + joule + external_load; }
    boundary walls on boundary("walls") { dirichlet T = ambient; }
}
"#;

#[test]
fn defined_sources_are_model_defined_and_inputs_are_required_slots() {
    let compilation = compile_semantics(INPUTS, &UnitRegistry::si_bootstrap()).unwrap();
    let manifests = derive_binding_slots(&compilation);
    let manifest = &manifests[0];
    let slot = |id: &str| {
        manifest
            .slots
            .iter()
            .find(|slot| slot.id == id)
            .unwrap_or_else(|| panic!("no slot {id}"))
    };

    let joule = slot("source/joule");
    assert_eq!(joule.status, SlotStatus::ModelDefined);
    assert_eq!(joule.kind, SlotKind::ExternalValue);
    assert!(
        joule.expression.is_some(),
        "the defining expression is recorded"
    );

    let load = slot("source/external_load");
    assert_eq!(load.status, SlotStatus::Required);
    assert!(load.expression.is_none());

    let q = slot("input/Q");
    assert_eq!(q.status, SlotStatus::Required);
    assert_eq!(q.kind, SlotKind::ExternalValue);
    let ambient = slot("input/ambient");
    assert_eq!(ambient.status, SlotStatus::Required);
    assert_eq!(ambient.kind, SlotKind::Parameter);

    let model = &compilation.semantic.models[0];
    let body = model.domains[0].id;
    let q_decl = model
        .declarations
        .iter()
        .find(|declaration| declaration.name == "Q")
        .unwrap();
    assert_eq!(q_decl.role, SemanticRole::Source);
    assert_eq!(
        q_decl.kind,
        SemanticDeclarationKind::InputField { domain: body }
    );
    assert_eq!(q_decl.domain, Some(body));
    assert_eq!(
        model.symbols[q_decl.symbol.unwrap().index()].domain,
        Some(body)
    );
    let ambient_decl = model
        .declarations
        .iter()
        .find(|declaration| declaration.name == "ambient")
        .unwrap();
    assert_eq!(ambient_decl.role, SemanticRole::Parameter);
    assert_eq!(ambient_decl.kind, SemanticDeclarationKind::InputValue);
}

#[test]
fn input_declarations_format_idempotently_and_refuse_definitions() {
    let module = parse_scientific_module(INPUTS).unwrap();
    let formatted = format_scientific_module(&module);
    assert!(formatted.contains("input field Q: VolumetricHeatSource on body;"));
    assert!(formatted.contains("input value ambient: ThermodynamicTemperature;"));
    let reparsed = parse_scientific_module(&formatted).unwrap();
    assert_eq!(format_scientific_module(&reparsed), formatted);
    assert_eq!(semantic_digest(&reparsed), semantic_digest(&module));

    let defined = INPUTS.replace(
        "input value ambient: ThermodynamicTemperature;",
        "input value ambient: ThermodynamicTemperature = 300;",
    );
    let errors = parse_scientific_module(&defined).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.to_string().contains("cannot carry a definition")),
        "{errors:?}"
    );
    let no_domain = INPUTS.replace(
        "input field Q: VolumetricHeatSource on body;",
        "input field Q: VolumetricHeatSource;",
    );
    let errors = parse_scientific_module(&no_domain).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.to_string().contains("must declare `on <domain>`")),
        "{errors:?}"
    );
}

#[test]
fn input_is_a_soft_keyword() {
    let source = r#"
module sc.soft_input;
model Soft {
    domain Omega { dimension = 2; coordinates = cartesian; }
    field u: unknown scalar H1(order=1) on Omega;
    provider gain(input: Dimensionless) -> Dimensionless { differentiability = analytic_provided; }
    parameter input: Dimensionless = 2;
    property g = gain(input);
    equation balance on Omega { -div(g * grad(u)) = 0; }
}
"#;
    let compilation = compile_semantics(source, &UnitRegistry::si_bootstrap()).unwrap();
    assert!(
        compilation.semantic.models[0]
            .declarations
            .iter()
            .any(|declaration| declaration.name == "input"
                && declaration.role == SemanticRole::Parameter)
    );
}

#[test]
fn source_locator_carries_the_module_digest() {
    let compilation = compile_semantics(INPUTS, &UnitRegistry::si_bootstrap()).unwrap();
    let module = parse_scientific_module(INPUTS).unwrap();
    let resolved = resolve_modules(module.clone(), &NoImports).unwrap();
    let expected = ModuleDigest::of(&module);
    assert_eq!(compilation.module_digest(), expected);
    assert_eq!(resolved.module_digests.get("sc.inputs"), Some(&expected));
    let declaration = &compilation.semantic.models[0].declarations[0];
    let locator = compilation.locate(declaration.span);
    assert_eq!(locator.module, expected);
    assert_eq!(locator.span, declaration.span);
    // The locator is presentation-independent in its module half: reformatting the module
    // moves spans but not the digest.
    let reformatted = parse_scientific_module(&format_scientific_module(&module)).unwrap();
    assert_eq!(ModuleDigest::of(&reformatted), expected);
}
