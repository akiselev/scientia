//! SV1-A: `DerivativeRequest` production from `.res` `objective`/`observable` declarations
//! (GX decision 8), on the inverse-Poisson conductivity model.

use quantitas::{Dimension, UnitRegistry};
use scientia::{
    ActiveInputRole, DerivativeLevel, DerivativeProductSpec, DerivativeRequestSpec,
    DerivativeStateConvention, LINKED_DERIVATIVE_REQUEST_SCHEMA, LinkedDerivativeRequest,
    ObjectiveSense, SemanticDeclarationKind, SlotKind, SlotStatus, compile_semantics,
    derive_binding_slots, derive_derivative_request, format_scientific_module,
    parse_scientific_module,
};

/// `sinbad/physics/corpus/01-poisson.res` plus the two SV1-A declarations an inverse problem
/// needs: held-out observation data as an `input field`, and a misfit `objective`.
const INVERSE_POISSON: &str = r#"
module sv1.inverse_poisson;

model InversePoisson {
    domain Omega { dimension = 2; coordinates = cartesian; }

    field u: unknown scalar H1(order=1) on Omega;
    provider diffusivity(material: selector) -> Diffusivity { differentiability = analytic_provided; }
    provider exact_u( ) -> Dimensionless { differentiability = analytic_provided; }

    property k = diffusivity(0);
    source f: VolumetricSource;
    input field u_obs: Dimensionless on Omega;

    equation balance on Omega {
        -div(k * grad(u)) = f;
    }

    boundary walls on boundary("walls") {
        dirichlet u = exact_u();
    }

    observable energy { integrate(0.5 * k * dot(grad(u), grad(u))); }
    objective misfit { minimize integrate(0.5 * (u - u_obs) * (u - u_obs)); }
}
"#;

fn spec(objective: &str, active: &[&str], frozen: &[&str]) -> DerivativeRequestSpec {
    DerivativeRequestSpec {
        model: "InversePoisson".into(),
        objective: objective.into(),
        active: active.iter().map(|s| s.to_string()).collect(),
        frozen: frozen.iter().map(|s| s.to_string()).collect(),
        product: DerivativeProductSpec::Gradient,
        state: DerivativeStateConvention::ConvergedState,
        level: DerivativeLevel::Discrete,
    }
}

#[test]
fn inverse_poisson_conductivity_gradient_request_is_linked_and_typed() {
    let compilation = compile_semantics(INVERSE_POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let linked = derive_derivative_request(
        &compilation,
        &spec("misfit", &["provider/diffusivity"], &[]),
    )
    .unwrap();
    linked.validate().unwrap();
    assert_eq!(linked.schema, LINKED_DERIVATIVE_REQUEST_SCHEMA);
    assert_eq!(linked.request.schema, "scientia-derivative-request/1");
    linked.request.validate().unwrap();

    // Objective: sense, arena links, dependency closure through `k = diffusivity(0)`.
    let model = &compilation.semantic.models[0];
    let declaration = model
        .declarations
        .iter()
        .find(|declaration| declaration.name == "misfit")
        .unwrap();
    let SemanticDeclarationKind::Objective { value, sense } = declaration.kind else {
        panic!("objective declaration kind");
    };
    assert_eq!(sense, ObjectiveSense::Minimize);
    assert_eq!(linked.objective.declaration, declaration.id);
    assert_eq!(linked.objective.expression, value);
    assert_eq!(linked.objective.sense, ObjectiveSense::Minimize);
    assert_eq!(linked.objective.slot, "observable/misfit");
    let symbol = |name: &str| model.symbols.iter().find(|s| s.name == name).unwrap().id;
    assert_eq!(
        linked.objective.depends_on,
        vec![symbol("u"), symbol("u_obs")]
    );
    assert_eq!(
        linked.request.objective.functional.semantic_expression,
        "integrate(((0.5 * (u - u_obs)) * (u - u_obs)))"
    );
    assert_eq!(linked.request.objective.sense, ObjectiveSense::Minimize);

    // Design variable: the case property slot behind `k`, typed from the provider signature.
    let [input] = linked.inputs.as_slice() else {
        panic!("one input");
    };
    assert_eq!(input.name, "provider/diffusivity");
    assert_eq!(input.role, ActiveInputRole::DesignVariable);
    assert!(input.active);
    assert!(matches!(
        input.kind,
        SlotKind::Provider { provider: Some(_) }
    ));
    assert_eq!(input.provider, Some(model.providers[0].id));
    let diffusivity = Dimension::LENGTH
        .checked_powi(2)
        .unwrap()
        .checked_quotient(Dimension::TIME)
        .unwrap();
    assert_eq!(input.dimension, Some(diffusivity));
    // Gradient units: the misfit is dimensionless, so dJ/dk carries s/m^2.
    assert_eq!(
        input.gradient_dimension,
        Some(
            Dimension::DIMENSIONLESS
                .checked_quotient(diffusivity)
                .unwrap()
        )
    );
    assert_eq!(linked.request.design_variables.len(), 1);
    assert_eq!(linked.request.design_variables[0].dimension, diffusivity);
    assert!(linked.request.controls.is_empty());
    assert_eq!(
        linked.request.active_set.active,
        vec!["provider/diffusivity".to_string()]
    );
    assert_eq!(linked.request.product, DerivativeProductSpec::Gradient);
    assert_eq!(
        linked.request.convention.state,
        DerivativeStateConvention::ConvergedState
    );

    // The objective is also an observable slot, so the same case machinery evaluates J.
    let manifest = &derive_binding_slots(&compilation)[0];
    let slot = manifest
        .slots
        .iter()
        .find(|slot| slot.id == "observable/misfit")
        .unwrap();
    assert_eq!(slot.kind, SlotKind::Observable);
    assert_eq!(slot.status, SlotStatus::ModelDefined);
    assert_eq!(slot.expression, Some(value));

    // Serde round trip and identity.
    let json = serde_json::to_string(&linked).unwrap();
    let decoded: LinkedDerivativeRequest = serde_json::from_str(&json).unwrap();
    decoded.validate().unwrap();
    assert_eq!(decoded, linked);
    let again = derive_derivative_request(
        &compilation,
        &spec("misfit", &["provider/diffusivity"], &[]),
    )
    .unwrap();
    assert_eq!(again.identity, linked.identity);
    let mut tampered = linked.clone();
    tampered.inputs[0].active = false;
    assert!(tampered.validate().is_err());
}

#[test]
fn controls_and_frozen_inputs_partition_the_request() {
    let compilation = compile_semantics(INVERSE_POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let linked = derive_derivative_request(
        &compilation,
        &spec(
            "misfit",
            &["source/f", "provider/diffusivity"],
            &["input/u_obs", "boundary/walls/u"],
        ),
    )
    .unwrap();
    linked.validate().unwrap();
    let names = linked
        .inputs
        .iter()
        .map(|input| (input.name.as_str(), input.role, input.active))
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec![
            ("boundary/walls/u", ActiveInputRole::Control, false),
            ("input/u_obs", ActiveInputRole::Control, false),
            (
                "provider/diffusivity",
                ActiveInputRole::DesignVariable,
                true
            ),
            ("source/f", ActiveInputRole::Control, true),
        ]
    );
    assert_eq!(linked.request.controls.len(), 3);
    let support = |name: &str| {
        linked
            .request
            .controls
            .iter()
            .find(|control| control.name == name)
            .unwrap()
            .support
            .clone()
    };
    assert_eq!(support("source/f"), "model");
    assert_eq!(support("input/u_obs"), "domain Omega");
    assert_eq!(support("boundary/walls/u"), "region walls");
    assert_eq!(
        linked.request.active_set.frozen,
        vec!["boundary/walls/u".to_string(), "input/u_obs".to_string()]
    );
}

#[test]
fn refusals_are_typed_and_name_the_case_slot_behind_a_model_defined_property() {
    let compilation = compile_semantics(INVERSE_POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let code = |s: DerivativeRequestSpec| derive_derivative_request(&compilation, &s).unwrap_err();

    let error = code(spec("misfit", &["property/k"], &[]));
    assert_eq!(error.code, "DERIVATIVE_MODEL_DEFINED_SLOT");
    assert!(
        error.message.contains("`provider/diffusivity`"),
        "{}",
        error.message
    );

    assert_eq!(
        code(spec("misfit", &["domain/Omega"], &[])).code,
        "DERIVATIVE_SLOT_NOT_DIFFERENTIABLE"
    );
    assert_eq!(
        code(spec("misfit", &["observable/energy"], &[])).code,
        "DERIVATIVE_SLOT_NOT_DIFFERENTIABLE"
    );
    assert_eq!(
        code(spec("misfit", &[], &[])).code,
        "DERIVATIVE_NO_ACTIVE_INPUT"
    );
    assert_eq!(
        code(spec("nope", &["provider/diffusivity"], &[])).code,
        "DERIVATIVE_UNKNOWN_OBJECTIVE"
    );
    assert_eq!(
        code(spec("misfit", &["provider/nope"], &[])).code,
        "DERIVATIVE_UNKNOWN_SLOT"
    );
    assert_eq!(
        code(spec(
            "misfit",
            &["provider/diffusivity"],
            &["provider/diffusivity"]
        ))
        .code,
        "DERIVATIVE_DUPLICATE_INPUT"
    );
    let mut wrong_model = spec("misfit", &["provider/diffusivity"], &[]);
    wrong_model.model = "Poisson".into();
    assert_eq!(code(wrong_model).code, "DERIVATIVE_UNKNOWN_MODEL");

    // An undeclared provider is an Unbound slot and cannot be differentiated.
    let undeclared = INVERSE_POISSON.replace(
        "provider diffusivity(material: selector) -> Diffusivity { differentiability = analytic_provided; }\n",
        "",
    );
    let compilation = compile_semantics(&undeclared, &UnitRegistry::si_bootstrap()).unwrap();
    let error = derive_derivative_request(
        &compilation,
        &spec("misfit", &["provider/diffusivity"], &[]),
    )
    .unwrap_err();
    assert_eq!(error.code, "DERIVATIVE_UNBOUND_SLOT");
}

#[test]
fn objective_grammar_formats_idempotently_and_refuses_a_missing_sense() {
    let module = parse_scientific_module(INVERSE_POISSON).unwrap();
    let formatted = format_scientific_module(&module);
    assert!(
        formatted.contains(
            "objective misfit { minimize integrate(((0.5 * (u - u_obs)) * (u - u_obs))); }"
        )
    );
    let reparsed = parse_scientific_module(&formatted).unwrap();
    assert_eq!(format_scientific_module(&reparsed), formatted);

    let broken = INVERSE_POISSON.replace("{ minimize integrate", "{ integrate");
    let errors = parse_scientific_module(&broken).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.to_string().contains("must open with `minimize`")),
        "{errors:?}"
    );
}

/// Opt-in corpus check (`SINBAD_WORKSPACE`): the unmodified corpus Poisson model's `energy`
/// observable can be requested as a `Measure` objective with respect to the conductivity slot,
/// which is what `sinbad run --gradient` consumes for inverse Poisson before the corpus file
/// gains an `objective`.
#[test]
fn sinbad_corpus_poisson_energy_is_a_measurable_objective_in_the_conductivity_slot() {
    let Some(workspace) = std::env::var_os("SINBAD_WORKSPACE") else {
        eprintln!("skipping: SINBAD_WORKSPACE is not set; corpus sweep is opt-in");
        return;
    };
    let path = std::path::PathBuf::from(workspace).join("sinbad/physics/corpus/01-poisson.res");
    let source = std::fs::read_to_string(path).unwrap();
    let compilation = compile_semantics(&source, &UnitRegistry::si_bootstrap()).unwrap();
    let mut request = spec("energy", &["provider/diffusivity"], &["source/f"]);
    request.model = "Poisson".into();
    request.product = DerivativeProductSpec::Vjp;
    let linked = derive_derivative_request(&compilation, &request).unwrap();
    linked.validate().unwrap();
    assert_eq!(linked.objective.sense, ObjectiveSense::Measure);
    assert_eq!(linked.request.product, DerivativeProductSpec::Vjp);
    assert_eq!(linked.request.design_variables.len(), 1);
    assert_eq!(linked.request.controls.len(), 1);
    assert_eq!(
        linked.parent_semantic_digest.hex,
        linked.request.parent_semantic_digest
    );
}
