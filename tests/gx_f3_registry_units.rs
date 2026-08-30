//! GX-F3: `QuantityKindRegistry` consumption (replacing the old five-kind `known_kind_dimension`
//! table), `RESOLVE_UNKNOWN_QUANTITY_KIND` severity (error inside a provider signature, advisory
//! on a `source`/`field`/`parameter` kind attribute), compound-unit resolution on provider and
//! field `unit = ...` attributes, and SI canonicalization of a unit-bearing numeric literal in a
//! general expression (closing the FC4 "unit-bearing literal before numeric canonicalization"
//! refusal).

use quantitas::{Dimension, QuantityKindRegistry, UnitRegistry};
use scientia::{
    Registries, SemanticDeclarationKind, SemanticExprKind, SourceSeverity, compile_semantics,
    compile_semantics_with, scientific::NoImports,
};

fn registries() -> (UnitRegistry, QuantityKindRegistry) {
    (
        UnitRegistry::si_bootstrap(),
        QuantityKindRegistry::si_bootstrap(),
    )
}

#[test]
fn unknown_quantity_kind_is_an_error_on_a_provider_input() {
    const SOURCE: &str = r#"
module gx_f3.provider_input_kind;
model M {
  domain Omega { dimension = 1; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega { time_role = differential; };
  provider bogus(x: NotARealKind) -> ThermalConductivity;
  property k = bogus(T);
}
"#;
    let (units, kinds) = registries();
    let diagnostics =
        compile_semantics_with(SOURCE, Registries::new(&units, &kinds), &NoImports).unwrap_err();
    let diagnostic = diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "RESOLVE_UNKNOWN_QUANTITY_KIND")
        .expect("unknown provider input kind is reported");
    assert_eq!(diagnostic.severity, SourceSeverity::Error);
    assert_eq!(
        &SOURCE[diagnostic.span.start..diagnostic.span.end],
        "NotARealKind"
    );
}

#[test]
fn unknown_quantity_kind_is_an_error_on_a_provider_output() {
    const SOURCE: &str = r#"
module gx_f3.provider_output_kind;
model M {
  domain Omega { dimension = 1; coordinates = cartesian; }
  provider bogus(material: selector) -> NotARealKindEither;
  property k = bogus(0);
}
"#;
    let (units, kinds) = registries();
    let diagnostics =
        compile_semantics_with(SOURCE, Registries::new(&units, &kinds), &NoImports).unwrap_err();
    let diagnostic = diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "RESOLVE_UNKNOWN_QUANTITY_KIND")
        .expect("unknown provider output kind is reported");
    assert_eq!(diagnostic.severity, SourceSeverity::Error);
    assert_eq!(
        &SOURCE[diagnostic.span.start..diagnostic.span.end],
        "NotARealKindEither"
    );
}

#[test]
fn unknown_quantity_kind_is_only_advisory_on_source_field_and_parameter_attributes() {
    const SOURCE: &str = r#"
module gx_f3.advisory_kinds;
model M {
  domain Omega { dimension = 1; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega {
    quantity = NotARegisteredKind;
    time_role = differential;
  };
  parameter p: AlsoNotRegistered = 1;
  source s: StillNotRegistered;
  equation trivial on Omega { T = T; }
}
"#;
    let compilation = compile_semantics(SOURCE, &UnitRegistry::si_bootstrap())
        .expect("advisories do not fail elaboration");
    let codes: Vec<&str> = compilation
        .advisories
        .iter()
        .filter(|diagnostic| diagnostic.code == "RESOLVE_UNKNOWN_QUANTITY_KIND")
        .map(|diagnostic| diagnostic.code.as_str())
        .collect();
    assert_eq!(
        codes.len(),
        3,
        "field, parameter, and source each report one advisory: {:#?}",
        compilation.advisories
    );
    assert!(
        compilation
            .advisories
            .iter()
            .all(|diagnostic| diagnostic.severity == SourceSeverity::Warning
                || diagnostic.code == "RESOLVE_UNDECLARED_PROVIDER"),
    );
}

#[test]
fn provider_input_kind_is_now_checked_for_a_non_temperature_kind() {
    // Before GX-F3, `known_kind_dimension` resolved only `ThermodynamicTemperature` /
    // `TemperatureDifference` / `Dimensionless`, so `TYPE_PROVIDER_INPUT_KIND` could only ever
    // fire for a temperature-kinded input. `Length` now resolves too, so passing a
    // `ThermodynamicTemperature`-kinded argument where `Length` is declared is a real,
    // catchable mismatch.
    const SOURCE: &str = r#"
module gx_f3.provider_length_kind;
model M {
  domain Omega { dimension = 1; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega {
    quantity = ThermodynamicTemperature;
    unit = K;
    time_role = differential;
  };
  provider needs_length(x: Length) -> Density;
  property rho = needs_length(T);
}
"#;
    let diagnostics = compile_semantics(SOURCE, &UnitRegistry::si_bootstrap()).unwrap_err();
    let diagnostic = diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "TYPE_PROVIDER_INPUT_KIND")
        .expect("a temperature-kinded argument against a Length-kinded input is a real mismatch");
    assert_eq!(diagnostic.severity, SourceSeverity::Error);
}

#[test]
fn compound_unit_resolves_on_a_provider_output_attribute() {
    const SOURCE: &str = r#"
module gx_f3.provider_compound_unit;
model M {
  domain Omega { dimension = 1; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega {
    quantity = ThermodynamicTemperature;
    unit = K;
    time_role = differential;
  };
  provider thermal_conductivity(T: ThermodynamicTemperature) -> ThermalConductivity {
    unit = W/(m*K);
  }
  property k = thermal_conductivity(T);
}
"#;
    let compilation = compile_semantics(SOURCE, &UnitRegistry::si_bootstrap())
        .expect("compound provider unit resolves");
    let provider = &compilation.semantic.models[0].providers[0];
    let expected = UnitRegistry::si_bootstrap()
        .parse_unit_expression("W/(m*K)")
        .unwrap();
    assert_eq!(provider.output.dimension, Some(expected.dimension));
}

#[test]
fn compound_unit_resolves_on_a_field_unit_attribute() {
    const SOURCE: &str = r#"
module gx_f3.field_compound_unit;
model M {
  domain Omega { dimension = 1; coordinates = cartesian; }
  field k: coefficient scalar H1(order=1) on Omega {
    quantity = ThermalConductivity;
    unit = W/(m*K);
  };
}
"#;
    let compilation = compile_semantics(SOURCE, &UnitRegistry::si_bootstrap())
        .expect("compound field unit resolves");
    let field = &compilation.semantic.models[0].symbols[0];
    let expected = UnitRegistry::si_bootstrap()
        .parse_unit_expression("W/(m*K)")
        .unwrap();
    assert_eq!(field.ty.dimension, Some(expected.dimension));
}

/// Finds the `SemanticExprKind::Number` node behind `constant <name> = <literal>;`.
fn constant_number_literal(model: &scientia::SemanticModel, name: &str) -> (f64, Option<String>) {
    let declaration = model
        .declarations
        .iter()
        .find(|declaration| declaration.name == name)
        .unwrap_or_else(|| panic!("declaration `{name}` exists"));
    let SemanticDeclarationKind::Value { value: Some(expr) } = &declaration.kind else {
        panic!("`{name}` has a value expression");
    };
    match &model.expressions[expr.index()].kind {
        SemanticExprKind::Number { value, unit, .. } => {
            (*value, unit.as_ref().map(|unit| unit.as_str().to_owned()))
        }
        other => panic!("`{name}` is not a Number node: {other:?}"),
    }
}

#[test]
fn unit_bearing_literals_canonicalize_to_si_during_elaboration() {
    // Contract GX-F3 item 4's three worked examples, plus a prefixed unit to prove the scale is
    // actually applied (not just dimension resolution): `kPa` is a factor of 1000 above the
    // coherent SI pascal.
    const SOURCE: &str = r#"
module gx_f3.si_literals;
model M {
  domain Omega { dimension = 1; coordinates = cartesian; }
  constant conductivity = 2.5 W/(m*K);
  constant temperature = 300 K;
  constant diffusivity = 1.0 m^2/s;
  constant pressure = 2.5 kPa;
}
"#;
    let compilation = compile_semantics(SOURCE, &UnitRegistry::si_bootstrap())
        .expect("unit-bearing literals elaborate");
    let model = &compilation.semantic.models[0];

    let (value, unit) = constant_number_literal(model, "conductivity");
    assert!(
        (value - 2.5).abs() < 1e-12,
        "W/(m*K) is already coherent SI"
    );
    assert_eq!(unit.as_deref(), Some("W/(m*K)"));

    let (value, unit) = constant_number_literal(model, "temperature");
    assert!((value - 300.0).abs() < 1e-12, "kelvin is already SI");
    assert_eq!(unit.as_deref(), Some("si:kelvin"));

    let (value, unit) = constant_number_literal(model, "diffusivity");
    assert!((value - 1.0).abs() < 1e-12, "m^2/s is already coherent SI");
    assert_eq!(unit.as_deref(), Some("m^2/s"));

    let (value, _unit) = constant_number_literal(model, "pressure");
    assert!(
        (value - 2500.0).abs() < 1e-9,
        "kPa must scale by 1000 to the coherent SI pascal, got {value}"
    );
}

#[test]
fn unresolvable_literal_unit_is_reported_and_does_not_scale() {
    const SOURCE: &str = r#"
module gx_f3.bad_literal_unit;
model M {
  domain Omega { dimension = 1; coordinates = cartesian; }
  constant bogus = 5 furlongs;
}
"#;
    let diagnostics = compile_semantics(SOURCE, &UnitRegistry::si_bootstrap()).unwrap_err();
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RESOLVE_UNKNOWN_UNIT"),
        "{diagnostics:#?}"
    );
}

#[test]
fn property_kernel_constant_with_a_compound_unit_canonicalizes_to_si() {
    use scientia::scientific::{
        FrameSemantics, OutOfValidityPolicy, PropertyDomain, PropertyEvidence, PropertyOutput,
        PropertySignature, TensorSymmetry, ValueShape,
    };
    use scientia::{
        DerivativeContract, PropertyDefinition, PropertyModel, lower_property_kernel,
        parse_expression,
    };

    let definition = PropertyDefinition {
        signature: PropertySignature {
            id: "diffusivity".into(),
            inputs: vec![],
            output: PropertyOutput {
                quantity_kind: quantitas::QuantityKindId::new("si:Diffusivity"),
                dimension: Dimension::DIMENSIONLESS,
                shape: ValueShape::Scalar,
                symmetry: TensorSymmetry::None,
                frame: FrameSemantics::Scalar,
            },
            locality: scientia::scientific::PropertyLocality::Pointwise,
            differentiability: DerivativeContract::None,
        },
        model: PropertyModel::Constant(parse_expression("1.0 m^2/s").unwrap()),
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
    };
    let kernel = lower_property_kernel(&definition, &UnitRegistry::si_bootstrap())
        .expect("a compound-unit constant now lowers instead of refusing RESOLVE_UNKNOWN_UNIT");
    assert_eq!(kernel.module.kernels.len(), 1);
    assert!(kernel.tangents.is_empty());
    malleus::validate(kernel.module.kernels[kernel.value_kernel].clone()).unwrap();
}
