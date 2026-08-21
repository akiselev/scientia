use scientia::{format_scientific_module, parse_scientific_module, semantic_digest};

const SOURCE: &str = r#"
module acceptance.source;
use physics.thermal;
model Complete {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega {
    quantity = ThermodynamicTemperature;
    unit = K;
    nominal = 300 K;
    min = 1 K;
    max = 3000 K;
    time_role = differential;
  };
  parameter alpha: ThermalExpansion [K] = 1 K;
  constant one: Dimensionless = 1;
  source Q: VolumetricHeatSource;
  property k = conductivity(T);
  constitutive flux = -k * grad(T);
  equation energy on Omega { dt(T) - div(k * grad(T)) = Q; }
  form weak_energy {
    cell(Omega): dt(T);
    boundary(walls): T;
    interior_facet(interior): k;
  }
  initial { T = 300 K; }
  boundary walls on boundary("walls") { dirichlet T = 300 K; }
  interface join on interface("join") { interface T = 300 K; }
  observable mean_T { integrate(T); }
  invariant positive { T > 0 K; }
  @mms(field = T);
}
"#;

#[test]
fn whitespace_and_comments_do_not_change_semantic_digest() {
    let a = parse_scientific_module(SOURCE).unwrap();
    let decorated = SOURCE
        .replace(
            "model Complete",
            "// presentation-only comment\n\nmodel   Complete",
        )
        .replace("field T", "field   T");
    let b = parse_scientific_module(&decorated).unwrap();
    assert_eq!(semantic_digest(&a), semantic_digest(&b));
}

#[test]
fn format_roundtrip_preserves_all_current_scientific_semantics() {
    let a = parse_scientific_module(SOURCE).unwrap();
    let formatted = format_scientific_module(&a);
    let b = parse_scientific_module(&formatted).unwrap();
    assert_eq!(semantic_digest(&a), semantic_digest(&b));
    assert_eq!(formatted, format_scientific_module(&b));
}

#[test]
fn every_current_semantic_declaration_has_a_nonempty_source_span() {
    let module = parse_scientific_module(SOURCE).unwrap();
    assert!(module.span.end > module.span.start);
    let m = &module.models[0];
    assert!(m.span.end > m.span.start);
    for span in m
        .domains
        .iter()
        .map(|x| x.span)
        .chain(m.fields.iter().map(|x| x.span))
        .chain(m.parameters.iter().map(|x| x.span))
        .chain(m.constants.iter().map(|x| x.span))
        .chain(m.sources.iter().map(|x| x.span))
        .chain(m.properties.iter().map(|x| x.span))
        .chain(m.constitutive_laws.iter().map(|x| x.span))
        .chain(m.equations.iter().map(|x| x.span))
        .chain(m.forms.iter().map(|x| x.span))
        .chain(m.initial_conditions.iter().map(|x| x.span))
        .chain(m.boundary_conditions.iter().map(|x| x.span))
        .chain(m.interface_conditions.iter().map(|x| x.span))
        .chain(m.observables.iter().map(|x| x.span))
        .chain(m.invariants.iter().map(|x| x.span))
        .chain(m.verifications.iter().map(|x| x.span))
        .chain(
            m.forms
                .iter()
                .flat_map(|form| form.integrals.iter().map(|x| x.span)),
        )
    {
        assert!(span.end > span.start, "empty declaration span: {span:?}");
    }
}
