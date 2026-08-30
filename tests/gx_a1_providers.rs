//! GX-A1 (contract C1): `provider` declarations, typed `ProviderCall` elaboration, and their
//! diagnostics. `tests/binding_slots.rs` covers the `scientia-binding-slots/1` manifest (C2).

use quantitas::UnitRegistry;
use scientia::{
    SemanticExprKind, SemanticRole, SemanticShape, compile_semantics, format_scientific_module,
    parse_scientific_module,
};

const DECLARED: &str = r#"
module gx_a1.declared;

model Declared {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega {
    quantity = ThermodynamicTemperature;
    unit = K;
    time_role = differential;
  };

  provider thermal_conductivity(T: ThermodynamicTemperature) -> ThermalConductivity {
    shape = scalar;
    locality = pointwise;
    differentiability = symbolic;
    domain { T in [200 K, 2000 K]; }
  }
  provider density(material: selector) -> Density;
  provider exact_T(t: Time) -> ThermodynamicTemperature;

  property k = thermal_conductivity(T);
  property rho = density(0);

  source Q: VolumetricHeatSource;
  equation energy on Omega { -div(k * grad(T)) = Q; }
  initial { T = exact_T(0 s); }
}
"#;

#[test]
fn provider_declarations_parse_with_and_without_a_body() {
    let module = parse_scientific_module(DECLARED).unwrap();
    let model = &module.models[0];
    assert_eq!(model.providers.len(), 3);

    let thermal_conductivity = model
        .providers
        .iter()
        .find(|provider| provider.name == "thermal_conductivity")
        .unwrap();
    assert_eq!(thermal_conductivity.inputs.len(), 1);
    assert_eq!(thermal_conductivity.inputs[0].name, "T");
    assert_eq!(
        thermal_conductivity.inputs[0].kind.as_deref(),
        Some("ThermodynamicTemperature")
    );
    assert_eq!(thermal_conductivity.output_kind, "ThermalConductivity");
    assert_eq!(thermal_conductivity.domain.len(), 1);
    assert_eq!(thermal_conductivity.domain[0].input, "T");

    let density = model
        .providers
        .iter()
        .find(|provider| provider.name == "density")
        .unwrap();
    assert_eq!(density.inputs[0].kind, None, "`selector` parses as no kind");

    let exact_t = model
        .providers
        .iter()
        .find(|provider| provider.name == "exact_T")
        .unwrap();
    assert!(exact_t.domain.is_empty());
}

#[test]
fn provider_formatting_round_trips_and_is_idempotent() {
    let module = parse_scientific_module(DECLARED).unwrap();
    let first_pass = format_scientific_module(&module);
    let reparsed = parse_scientific_module(&first_pass).unwrap();
    let second_pass = format_scientific_module(&reparsed);
    assert_eq!(first_pass, second_pass, "formatting must be a fixpoint");

    // The provider grammar itself round-trips, not just the fixpoint of formatting: arity,
    // selector vs. quantity-kind inputs, and the domain bound all survive. Spans differ because
    // reformatting changes byte offsets, so compare content rather than raw struct equality.
    let project = |providers: &[scientia::ProviderDecl]| {
        providers
            .iter()
            .map(|provider| {
                (
                    provider.name.clone(),
                    provider.output_kind.clone(),
                    provider
                        .inputs
                        .iter()
                        .map(|input| (input.name.clone(), input.kind.clone()))
                        .collect::<Vec<_>>(),
                    provider
                        .domain
                        .iter()
                        .map(|bound| (bound.input.clone(), bound.min.value, bound.max.value))
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        project(&module.models[0].providers),
        project(&reparsed.models[0].providers)
    );
}

#[test]
fn declared_provider_calls_elaborate_to_typed_provider_call_nodes() {
    let compilation = compile_semantics(DECLARED, &UnitRegistry::si_bootstrap()).unwrap();
    assert!(compilation.advisories.is_empty());
    let model = &compilation.semantic.models[0];
    let provider_call = model
        .expressions
        .iter()
        .find(|expression| {
            matches!(
                expression.kind,
                SemanticExprKind::ProviderCall { provider, .. }
                    if model.providers[provider.index()].name == "thermal_conductivity"
            )
        })
        .expect("thermal_conductivity(T) elaborates to a ProviderCall node");
    assert_eq!(provider_call.ty.role, SemanticRole::Provider);
    assert!(matches!(
        provider_call.ty.shape,
        SemanticShape::Numeric(scientia::scientific::ValueShape::Scalar)
    ));
    // GX-F3: the output kind now resolves through the `QuantityKindRegistry` (bare tail lookup),
    // so it carries the registry's canonical namespaced id rather than the authored bare name.
    assert_eq!(
        provider_call
            .ty
            .quantity_kind
            .as_ref()
            .map(|kind| kind.as_str()),
        Some("si:ThermalConductivity")
    );
}

#[test]
fn provider_arity_mismatch_is_a_typed_error() {
    const SOURCE: &str = r#"
module gx_a1.arity;
model Arity {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega { time_role = differential; };
  provider thermal_conductivity(T: ThermodynamicTemperature) -> ThermalConductivity;
  property k = thermal_conductivity(T, T);
  source Q: VolumetricHeatSource;
  equation energy on Omega { -div(k * grad(T)) = Q; }
}
"#;
    let diagnostics = compile_semantics(SOURCE, &UnitRegistry::si_bootstrap()).unwrap_err();
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "TYPE_PROVIDER_ARITY"),
        "expected TYPE_PROVIDER_ARITY, got {diagnostics:#?}"
    );
}

#[test]
fn provider_selector_input_requires_an_integer_literal() {
    const SOURCE: &str = r#"
module gx_a1.selector;
model Selector {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega { time_role = differential; };
  provider density(material: selector) -> Density;
  property rho = density(T);
  source Q: VolumetricHeatSource;
  equation energy on Omega { -div(rho * grad(T)) = Q; }
}
"#;
    let diagnostics = compile_semantics(SOURCE, &UnitRegistry::si_bootstrap()).unwrap_err();
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "TYPE_PROVIDER_SELECTOR"),
        "expected TYPE_PROVIDER_SELECTOR, got {diagnostics:#?}"
    );
}

#[test]
fn provider_input_dimension_mismatch_is_a_typed_error() {
    const SOURCE: &str = r#"
module gx_a1.input_kind;
model InputKind {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega {
    quantity = ThermodynamicTemperature;
    unit = K;
    time_role = differential;
  };
  parameter length: Dimensionless = 1;
  provider thermal_conductivity(T: ThermodynamicTemperature) -> ThermalConductivity;
  property k = thermal_conductivity(length);
  source Q: VolumetricHeatSource;
  equation energy on Omega { -div(k * grad(T)) = Q; }
}
"#;
    let diagnostics = compile_semantics(SOURCE, &UnitRegistry::si_bootstrap()).unwrap_err();
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "TYPE_PROVIDER_INPUT_KIND"),
        "expected TYPE_PROVIDER_INPUT_KIND, got {diagnostics:#?}"
    );
}

#[test]
fn calls_to_undeclared_providers_still_elaborate_with_an_advisory() {
    const SOURCE: &str = r#"
module gx_a1.undeclared;
model Undeclared {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega { time_role = differential; };
  property k = thermal_conductivity(T);
  source Q: VolumetricHeatSource;
  equation energy on Omega { -div(k * grad(T)) = Q; }
}
"#;
    let compilation = compile_semantics(SOURCE, &UnitRegistry::si_bootstrap())
        .expect("an undeclared provider call does not fail elaboration");
    assert!(
        compilation
            .advisories
            .iter()
            .any(|diagnostic| diagnostic.code == "RESOLVE_UNDECLARED_PROVIDER"),
        "expected RESOLVE_UNDECLARED_PROVIDER advisory, got {:#?}",
        compilation.advisories
    );
    // The call still elaborates as an opaque `Call`, not a `ProviderCall`, so every existing
    // downstream consumer of opaque calls keeps working unchanged.
    let model = &compilation.semantic.models[0];
    assert!(model.expressions.iter().any(|expression| matches!(
        &expression.kind,
        SemanticExprKind::Call { function, .. } if function == "thermal_conductivity"
    )));
}

#[test]
fn a_provider_cannot_reuse_an_intrinsic_name() {
    const SOURCE: &str = r#"
module gx_a1.reserved_name;
model ReservedName {
  domain Omega { dimension = 2; coordinates = cartesian; }
  provider grad(x: Dimensionless) -> Dimensionless;
}
"#;
    let diagnostics = compile_semantics(SOURCE, &UnitRegistry::si_bootstrap()).unwrap_err();
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RESOLVE_DUPLICATE_NAME"),
        "expected RESOLVE_DUPLICATE_NAME, got {diagnostics:#?}"
    );
}
