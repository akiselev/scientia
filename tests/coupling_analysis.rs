use scientia::scientific::CouplingReason;
use scientia::{
    IncidenceSystem, derive_coupling_graph, maximum_matching, parse_scientific_module,
    semantic_digest,
};

fn source(reordered: bool) -> String {
    let fields = if reordered {
        r#"
  field V: unknown scalar H1(order=1) on Omega;
  field T: state scalar H1(order=1) on Omega { time_role = differential; };
"#
    } else {
        r#"
  field T: state scalar H1(order=1) on Omega { time_role = differential; };
  field V: unknown scalar H1(order=1) on Omega;
"#
    };
    let equations = if reordered {
        r#"
  equation thermal on Omega { dt(T) - div(grad(T)) = joule; }
  equation electrical on Omega { div(effective_sigma * grad(V)) = 0; }
"#
    } else {
        r#"
  equation electrical on Omega { div(effective_sigma * grad(V)) = 0; }
  equation thermal on Omega { dt(T) - div(grad(T)) = joule; }
"#
    };
    format!(
        r#"module acceptance.coupling;
model Coupled {{
  domain Omega {{ dimension = 2; coordinates = cartesian; }}
{fields}
  property base_sigma = conductivity(T);
  property effective_sigma = 2 * base_sigma;
  constitutive current = effective_sigma * grad(V);
  source joule = dot(current, current) / effective_sigma;
{equations}
}}
"#
    )
}

fn canonical_edges(source: &str) -> Vec<(String, String, Vec<String>)> {
    let module = parse_scientific_module(source).unwrap();
    let mut edges = derive_coupling_graph(&module.models[0])
        .edges
        .into_iter()
        .map(|edge| (edge.from, edge.to, edge.path))
        .collect::<Vec<_>>();
    edges.sort();
    edges
}

#[test]
fn declaration_reordering_preserves_semantic_digest_and_coupling_graph() {
    let a = parse_scientific_module(&source(false)).unwrap();
    let b = parse_scientific_module(&source(true)).unwrap();
    assert_eq!(semantic_digest(&a), semantic_digest(&b));
    assert_eq!(
        canonical_edges(&source(false)),
        canonical_edges(&source(true))
    );
}

#[test]
fn nested_property_and_constitutive_dependencies_reach_cross_blocks() {
    let module = parse_scientific_module(&source(false)).unwrap();
    let graph = derive_coupling_graph(&module.models[0]);

    let thermal_from_voltage = graph
        .edges
        .iter()
        .find(|edge| edge.from == "V" && edge.to == "thermal")
        .expect("Joule source must couple voltage into thermal residual");
    assert!(
        thermal_from_voltage
            .path
            .iter()
            .any(|name| name == "current")
    );
    assert!(thermal_from_voltage.path.iter().any(|name| name == "joule"));

    let electrical_from_temperature = graph
        .edges
        .iter()
        .find(|edge| edge.from == "T" && edge.to == "electrical")
        .expect("conductivity chain must couple temperature into electrical residual");
    assert!(
        electrical_from_temperature
            .path
            .iter()
            .any(|name| name == "base_sigma")
    );
    assert!(
        electrical_from_temperature
            .path
            .iter()
            .any(|name| name == "effective_sigma")
    );
    assert!(matches!(
        electrical_from_temperature.reason,
        CouplingReason::PropertyDependency(_) | CouplingReason::ConstitutiveDependency(_)
    ));

    let nonzero = |residual: &str, unknown: &str| {
        graph.derivatives.iter().any(|block| {
            block.residual == residual && block.unknown == unknown && block.structurally_nonzero
        })
    };
    assert!(nonzero("thermal", "T"));
    assert!(nonzero("thermal", "V"));
    assert!(nonzero("electrical", "T"));
    assert!(nonzero("electrical", "V"));
}

#[test]
fn structural_incidence_agrees_with_transitive_coupling_dependencies() {
    let module = parse_scientific_module(&source(false)).unwrap();
    let model = &module.models[0];
    let graph = derive_coupling_graph(model);
    let incidence = IncidenceSystem::from_model(model).unwrap();

    for (row, equation) in model.equations.iter().enumerate() {
        for (column, variable) in incidence.variables.iter().enumerate() {
            let coupled = graph
                .edges
                .iter()
                .any(|edge| edge.from == *variable && edge.to == equation.name);
            assert_eq!(
                incidence.rows[row].contains(&column),
                coupled,
                "{variable} -> {}",
                equation.name
            );
        }
    }
    assert!(maximum_matching(&incidence).is_perfect());
}
