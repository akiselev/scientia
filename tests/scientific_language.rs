use scientia::{format_scientific_module, parse_scientific_module, semantic_digest};

fn valid_module(index: usize) -> String {
    format!(
        r#"module corpus.valid_{index};
model M{index} {{
    domain Omega {{ dimension = 2; coordinates = cartesian; }}
    field u: state scalar H1(order=1) on Omega {{ nominal = 1 K; time_role = differential; }};
    property a = 1 + 0.01 * u;
    source f: Source;
    equation balance on Omega {{ a * dt(u) - div(a * grad(u)) = f; }}
    initial {{ u = 1 K; }}
    observable total {{ integrate(u); }}
}}
"#
    )
}

#[test]
fn generated_corpus_has_fifty_valid_modules() {
    for index in 0..50 {
        let source = valid_module(index);
        let module = parse_scientific_module(&source)
            .unwrap_or_else(|errors| panic!("valid corpus item {index} failed: {errors:?}"));
        assert_eq!(module.models.len(), 1);
    }
}

#[test]
fn generated_corpus_has_fifty_invalid_modules() {
    for index in 0..50 {
        let source = format!(
            "module corpus.invalid_{index};\nmodel Bad{index} {{\n  nonsense alpha;\n  nonsense beta;\n}}\n"
        );
        let errors = parse_scientific_module(&source).expect_err("invalid corpus item parsed");
        assert!(
            errors.len() >= 2,
            "recovery must preserve multiple diagnostics for item {index}: {errors:?}"
        );
    }
}

#[test]
fn formatting_is_idempotent_and_semantics_stable() {
    let first = parse_scientific_module(&valid_module(7)).unwrap();
    let formatted_once = format_scientific_module(&first);
    let second = parse_scientific_module(&formatted_once).unwrap();
    let formatted_twice = format_scientific_module(&second);
    assert_eq!(formatted_once, formatted_twice);
    assert_eq!(semantic_digest(&first), semantic_digest(&second));
}

#[test]
fn declaration_order_does_not_change_coupling_dependencies() {
    let a = r#"
module corpus.order_a;
model Coupled {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega { time_role = differential; };
  field V: unknown scalar H1(order=1) on Omega;
  property sigma = conductivity(T);
  equation electrical on Omega { div(sigma * grad(V)) = 0; }
  equation thermal on Omega { dt(T) - div(grad(T)) = sigma * dot(grad(V), grad(V)); }
}
"#;
    let b = r#"
module corpus.order_b;
model Coupled {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field V: unknown scalar H1(order=1) on Omega;
  field T: state scalar H1(order=1) on Omega { time_role = differential; };
  property sigma = conductivity(T);
  equation thermal on Omega { dt(T) - div(grad(T)) = sigma * dot(grad(V), grad(V)); }
  equation electrical on Omega { div(sigma * grad(V)) = 0; }
}
"#;
    let ma = parse_scientific_module(a).unwrap();
    let mb = parse_scientific_module(b).unwrap();
    let mut ea = scientia::derive_coupling_graph(&ma.models[0]).edges;
    let mut eb = scientia::derive_coupling_graph(&mb.models[0]).edges;
    ea.sort_by(|x, y| (&x.from, &x.to, &x.path).cmp(&(&y.from, &y.to, &y.path)));
    eb.sort_by(|x, y| (&x.from, &x.to, &x.path).cmp(&(&y.from, &y.to, &y.path)));
    assert_eq!(ea, eb);
}
