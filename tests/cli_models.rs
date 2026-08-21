use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

const MULTI_MODEL_SOURCE: &str = r#"
module cli.multiple;

model First {
  domain D { dimension = 1; coordinates = cartesian; }
  field x: unknown scalar H1(order=1) on D;
  equation identity on D { x = 0; }
}

model Mixed {
  domain D { dimension = 2; coordinates = cartesian; }
  field u: trial scalar H1(order=1) on D;
  field v: test scalar H1(order=1) on D;
  form mixed_form { cell(D): dot(grad(u), grad(v)); }
}

model Maxwell {
  domain D { dimension = 2; coordinates = cartesian; }
  field potential: unknown scalar H1(order=1) on D;
  equation balance on D { -div(grad(potential)) = 0; }
  boundary wall on boundary("wall") { dirichlet potential = 0; }
}
"#;

fn fixture_path() -> PathBuf {
    std::env::temp_dir().join(format!(
        "scientia-cli-models-{}-{}.res",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ))
}

fn run(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_scientia"))
        .args(arguments)
        .output()
        .unwrap()
}

#[test]
fn model_qualified_selectors_address_every_model_aware_command() {
    let path = fixture_path();
    fs::write(&path, MULTI_MODEL_SOURCE).unwrap();
    let path_text = path.to_str().unwrap();
    let cases = [
        ("form", "Mixed:mixed_form", "\"model\": \"Mixed\""),
        ("requirements", "Mixed:mixed_form", "\"model\": \"Mixed\""),
        ("operator", "Mixed:mixed_form", "scientia-tensor-program/1"),
        ("derive-form", "Maxwell:balance", "\"model\": \"Maxwell\""),
        (
            "derive-requirements",
            "Maxwell:balance",
            "\"model\": \"Maxwell\"",
        ),
        (
            "derive-operator",
            "Maxwell:balance",
            "scientia-operator-factorization/1",
        ),
        ("coupling", "Mixed", "mixed_form"),
        ("structural", "Maxwell", "\"n_equations\": 1"),
    ];
    for (command, selector, expected) in cases {
        let output = run(&[command, path_text, selector]);
        assert!(
            output.status.success(),
            "{command} {selector}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains(expected),
            "{command} {selector}: {}",
            String::from_utf8_lossy(&output.stdout)
        );
    }
    let explain = run(&["explain", path_text, "First", "x"]);
    assert!(explain.status.success());
    assert!(String::from_utf8_lossy(&explain.stdout).contains("\"model\": \"First\""));

    let ambiguous = run(&["requirements", path_text, "mixed_form"]);
    assert!(!ambiguous.status.success());
    assert!(
        String::from_utf8_lossy(&ambiguous.stderr).contains("requires a `model:item` selector")
    );
    fs::remove_file(path).unwrap();
}
