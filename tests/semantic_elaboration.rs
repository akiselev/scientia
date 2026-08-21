use quantitas::{Dimension, UnitRegistry};
use scientia::{
    Frame, ScientificError, SemanticExprKind, SemanticRole, SemanticShape, compile_semantics,
    parse_scientific_module, parse_scientific_module_diagnostics, resolve_modules,
    semantic_arena_digest,
};
use std::collections::BTreeMap;

const VALID: &str = r#"
module acceptance.semantic;
model Typed {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega {
    quantity = ThermodynamicTemperature;
    unit = K;
    nominal = 300 K;
    time_role = differential;
  };
  field velocity: coefficient vector(2) H1(order=1) on Omega;
  parameter dt_value: TemperatureDifference [K] = 1 K;
  property k = conductivity(T);
  constitutive flux = -k * grad(T);
  source Q: VolumetricHeatSource;
  equation energy on Omega { dt(T) + dot(velocity, grad(T)) = Q; }
  initial { T = 300 K; }
  boundary walls on boundary("walls") { dirichlet T = 300 K; }
  observable mean_T { integrate(T); }
  invariant positive { T > 0 K; }
  @mms(field = T);
}
"#;

fn only_diagnostic(source: &str) -> scientia::SourceDiagnostic {
    let diagnostics = compile_semantics(source, &UnitRegistry::si_bootstrap()).unwrap_err();
    assert_eq!(
        diagnostics.len(),
        1,
        "unexpected diagnostics: {diagnostics:#?}"
    );
    diagnostics.into_iter().next().unwrap()
}

#[test]
fn elaboration_resolves_roles_domains_types_and_expression_identities() {
    let compilation = compile_semantics(VALID, &UnitRegistry::si_bootstrap()).unwrap();
    let model = &compilation.semantic.models[0];
    let temperature = model
        .symbols
        .iter()
        .find(|symbol| symbol.name == "T")
        .unwrap();
    assert!(matches!(
        temperature.ty.role,
        SemanticRole::PhysicalField(scientia::scientific::FieldRole::State)
    ));
    assert_eq!(temperature.ty.dimension, Some(Dimension::TEMPERATURE));
    assert!(matches!(temperature.ty.shape, SemanticShape::Numeric(_)));
    assert!(matches!(temperature.ty.frame, Frame::Domain(_)));
    assert!(model.expressions.iter().all(|expression| {
        expression.span.end > expression.span.start
            && match expression.kind {
                SemanticExprKind::Symbol { symbol } => symbol.index() < model.symbols.len(),
                _ => true,
            }
    }));
}

#[test]
fn elaboration_is_deterministic_across_presentation_changes() {
    let registry = UnitRegistry::si_bootstrap();
    let first = compile_semantics(VALID, &registry).unwrap();
    let decorated = VALID
        .replace("model Typed", "// comment\nmodel   Typed")
        .replace("field T", "field    T");
    let second = compile_semantics(&decorated, &registry).unwrap();
    assert_eq!(
        semantic_arena_digest(&first.semantic),
        semantic_arena_digest(&second.semantic)
    );
}

#[test]
fn malformed_unit_and_quantity_kind_point_at_authored_tokens() {
    let unknown = r#"module x; model M { domain D { dimension = 1; coordinates = cartesian; } field u: state scalar H1(1) on D { unit = furlong; }; }"#;
    let diagnostic = only_diagnostic(unknown);
    assert_eq!(diagnostic.code, "RESOLVE_UNKNOWN_UNIT");
    assert_eq!(
        &unknown[diagnostic.span.start..diagnostic.span.end],
        "furlong"
    );

    let mismatch = r#"module x; model M { domain D { dimension = 1; coordinates = cartesian; } field T: state scalar H1(1) on D { quantity = ThermodynamicTemperature; unit = m; }; }"#;
    let diagnostic = only_diagnostic(mismatch);
    assert_eq!(diagnostic.code, "RESOLVE_UNIT_KIND_MISMATCH");
    assert_eq!(
        &mismatch[diagnostic.span.start..diagnostic.span.end],
        "ThermodynamicTemperature"
    );
}

#[test]
fn malformed_roles_names_axes_and_frames_have_stable_precise_diagnostics() {
    let role = r#"module x; model M { domain D { dimension = 1; coordinates = cartesian; } field u: mystery scalar H1(1) on D; }"#;
    let diagnostics = parse_scientific_module_diagnostics(role).unwrap_err();
    assert_eq!(diagnostics[0].code, "PARSE_SYNTAX");
    assert_eq!(
        &role[diagnostics[0].span.start..diagnostics[0].span.end],
        "mystery"
    );

    let name = r#"module x; model M { domain D { dimension = 1; coordinates = cartesian; } field u: state scalar H1(1) on D; equation e on D { u = missing; } }"#;
    let diagnostic = only_diagnostic(name);
    assert_eq!(diagnostic.code, "RESOLVE_UNKNOWN_NAME");
    assert_eq!(&name[diagnostic.span.start..diagnostic.span.end], "missing");

    let axis = r#"module x; model M { domain D { dimension = 3; coordinates = cartesian; } field u: state vector(3) H1(1) on D; observable bad { u[3]; } }"#;
    let diagnostic = only_diagnostic(axis);
    assert_eq!(diagnostic.code, "TYPE_AXIS_BOUNDS");
    assert_eq!(&axis[diagnostic.span.start..diagnostic.span.end], "3");

    let frame = r#"module x; model M { domain D { dimension = 2; coordinates = cylindrical; } }"#;
    let diagnostic = only_diagnostic(frame);
    assert_eq!(diagnostic.code, "RESOLVE_FRAME_DIMENSION_MISMATCH");
    assert_eq!(
        &frame[diagnostic.span.start..diagnostic.span.end],
        "cylindrical"
    );
}

#[test]
fn role_and_frame_mismatches_are_not_silently_coerced() {
    let wrong_initial = r#"module x; model M { domain D { dimension = 1; coordinates = cartesian; } parameter p = 1; initial { p = 2; } }"#;
    assert_eq!(only_diagnostic(wrong_initial).code, "TYPE_ROLE_MISMATCH");

    let cross_domain = r#"module x; model M { domain A { dimension = 1; coordinates = cartesian; } domain B { dimension = 1; coordinates = cartesian; } field a: state scalar H1(1) on A; field b: state scalar H1(1) on B; equation bad { a + b = 0; } }"#;
    assert_eq!(only_diagnostic(cross_domain).code, "TYPE_FRAME_MISMATCH");
}

#[test]
fn module_resolution_is_deterministic_and_reports_the_import_span() {
    let missing = "module root; use absent.module; model M {}";
    let root = parse_scientific_module(missing).unwrap();
    let error = resolve_modules(root, &BTreeMap::new()).unwrap_err();
    let diagnostic = error.diagnostic();
    assert_eq!(diagnostic.code, "RESOLVE_MISSING_MODULE");
    assert_eq!(
        &missing[diagnostic.span.start..diagnostic.span.end],
        "absent.module"
    );

    let root = parse_scientific_module("module root; use dependency; model M {}").unwrap();
    let mut compact = BTreeMap::new();
    compact.insert("dependency".into(), "module dependency; model D {}".into());
    let mut decorated = BTreeMap::new();
    decorated.insert(
        "dependency".into(),
        "// presentation only\nmodule   dependency;\nmodel D {}".into(),
    );
    let first = resolve_modules(root.clone(), &compact).unwrap();
    let second = resolve_modules(root, &decorated).unwrap();
    assert_eq!(first.semantic_digest, second.semantic_digest);

    let _exhaustive_match = matches!(error, ScientificError::MissingModule { .. });
    assert!(_exhaustive_match);
}
