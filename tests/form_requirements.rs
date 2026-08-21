use quantitas::UnitRegistry;
use scientia::{
    DerivativeEvaluation, ElementFamilyRequirement, EvaluationSite, FormAssumption,
    GeometryPreprocessingRequirement, InputSourceRequirement, OrientationRequirement,
    PullbackRequirement, QuadraturePrecision, RequirementInferenceError, SemanticMeasure,
    SpaceComposition, TraceMapping, TraceRequirement, compile_semantics, compile_variational_form,
    derive_variational_form, infer_form_requirements,
};

const ALL_SPACES: &str = r#"
module requirements.spaces;
model Spaces {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field h1_trial: trial scalar H1(order=2) on Omega;
  field h1_test: test scalar H1(order=2) on Omega;
  field l2_trial: trial scalar L2(order=1) on Omega;
  field l2_test: test scalar L2(order=1) on Omega;
  field dg_trial: trial scalar DG(order=1) on Omega;
  field dg_test: test scalar DG(order=1) on Omega;
  field curl_trial: trial vector(2) HCurl(order=1) on Omega;
  field curl_test: test vector(2) HCurl(order=1) on Omega;
  field div_trial: trial vector(2) HDiv(order=1) on Omega;
  field div_test: test vector(2) HDiv(order=1) on Omega;
  form product {
    cell(Omega): dot(grad(h1_trial), grad(h1_test));
    cell(Omega): l2_trial * l2_test;
    interior_facet(faces): trace_minus(dg_trial) * trace_plus(dg_test);
    interior_facet(faces): inner(trace_minus(curl_trial), trace_plus(curl_test));
    interior_facet(faces): normal_component_minus(div_trial) * normal_component_plus(div_test);
  }
}
"#;

const FC3_GATE_FIXTURES: &[(&str, &str, &[&str])] = &[
    (
        r#"
module fixture.poisson;
model Poisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  property k = diffusivity(0);
  source f: VolumetricSource;
  equation balance on Omega { -div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(); }
}
"#,
        "Poisson",
        &["balance"],
    ),
    (
        r#"
module fixture.transient;
model TransientDiffusion {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: state scalar H1(order=1) on Omega { time_role = differential; };
  property capacity = storage_capacity(u);
  property k = diffusivity(u);
  source f: VolumetricSource;
  equation evolution on Omega { capacity * dt(u) - div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(t); }
}
"#,
        "TransientDiffusion",
        &["evolution"],
    ),
    (
        r#"
module fixture.heat;
model NonlinearHeat {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega { time_role = differential; };
  property rho = density(T);
  property cp = specific_heat(T);
  property k = thermal_conductivity(T);
  source Q: VolumetricHeatSource;
  equation energy on Omega { rho * cp * dt(T) - div(k * grad(T)) = Q; }
  boundary walls on boundary("walls") { dirichlet T = exact_T(t); }
}
"#,
        "NonlinearHeat",
        &["energy"],
    ),
    (
        r#"
module fixture.elasticity;
model LinearElasticity {
  domain Omega { dimension = 3; coordinates = cartesian; }
  field displacement: unknown vector(3) H1(order=1) on Omega;
  property lambda = lame_lambda(0);
  property mu = lame_mu(0);
  source body_force: MechanicalBodyForce;
  constitutive strain = sym_grad(displacement);
  constitutive stress = lambda * trace(strain) * identity(3) + 2 * mu * strain;
  equation momentum on Omega { -div(stress) = body_force; }
  boundary clamp on boundary("clamp") { dirichlet displacement = [0, 0, 0]; }
}
"#,
        "LinearElasticity",
        &["momentum"],
    ),
    (
        r#"
module fixture.stokes;
model StokesFlow {
  domain Fluid { dimension = 2; coordinates = cartesian; }
  field velocity: unknown vector(2) H1(order=2) on Fluid;
  field pressure: unknown scalar L2(order=1) on Fluid;
  property mu = dynamic_viscosity(0);
  source body_force: MechanicalBodyForce;
  constitutive strain_rate = sym_grad(velocity);
  constitutive viscous_stress = 2 * mu * strain_rate;
  equation momentum on Fluid { -div(viscous_stress) + grad(pressure) = body_force; }
  equation incompressibility on Fluid { div(velocity) = 0; }
  boundary walls on boundary("walls") { dirichlet velocity = [0, 0]; }
}
"#,
        "StokesFlow",
        &["momentum", "incompressibility"],
    ),
];

#[test]
fn fc3_gate_forms_all_infer_mesh_free_requirements() {
    for (source, model, equations) in FC3_GATE_FIXTURES {
        let compilation = compile_semantics(source, &UnitRegistry::si_bootstrap()).unwrap();
        for equation in *equations {
            let form = derive_variational_form(&compilation.semantic, model, equation).unwrap();
            let requirements = infer_form_requirements(&compilation.semantic, &form)
                .unwrap_or_else(|error| panic!("{model}::{equation}: {error}"));
            assert!(!requirements.spaces.is_empty(), "{model}::{equation}");
            assert!(
                !requirements.integral_groups.is_empty(),
                "{model}::{equation}"
            );
        }
    }
}

#[test]
fn constitutive_chains_expose_state_spaces_and_model_defined_inputs() {
    let (source, model_name, _) = FC3_GATE_FIXTURES[4];
    let compilation = compile_semantics(source, &UnitRegistry::si_bootstrap()).unwrap();
    let model = &compilation.semantic.models[0];
    let form = derive_variational_form(&compilation.semantic, model_name, "momentum").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let symbol = |name: &str| {
        model
            .symbols
            .iter()
            .find(|symbol| symbol.name == name)
            .unwrap()
            .id
    };
    let velocity = symbol("velocity");
    let viscous_stress = symbol("viscous_stress");

    assert!(
        requirements
            .spaces
            .iter()
            .any(|space| space.symbol == velocity),
        "the constitutive dependency must contribute its physical field space"
    );
    let inputs = requirements
        .integral_groups
        .iter()
        .flat_map(|group| &group.signature.inputs)
        .collect::<Vec<_>>();
    assert!(inputs.iter().any(|input| {
        input.symbol == velocity
            && input.source == InputSourceRequirement::Basis
            && input
                .evaluations
                .iter()
                .any(|evaluation| evaluation.derivative == DerivativeEvaluation::SymmetricGradient)
    }));
    assert!(inputs.iter().any(|input| {
        input.symbol == viscous_stress
            && matches!(
                input.source,
                InputSourceRequirement::ModelDefinedConstitutive { .. }
            )
    }));
    assert!(inputs.iter().all(|input| {
        input.source != InputSourceRequirement::Basis
            || requirements
                .spaces
                .iter()
                .any(|space| space.symbol == input.symbol)
    }));
}

#[test]
fn fc3_models_product_h1_l2_dg_hcurl_hdiv_and_trace_requirements() {
    let compilation = compile_semantics(ALL_SPACES, &UnitRegistry::si_bootstrap()).unwrap();
    let form = compile_variational_form(&compilation.semantic, "Spaces", "product").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let wire = serde_json::to_string(&requirements).unwrap();
    let round_trip: scientia::FormRequirements = serde_json::from_str(&wire).unwrap();

    assert_eq!(
        requirements.space_system.composition,
        SpaceComposition::Product
    );
    assert_eq!(round_trip, requirements);
    assert_eq!(requirements.spaces.len(), 10);
    assert_eq!(requirements.elements.len(), 10);
    assert!(requirements.elements.iter().any(|element| {
        element.family == ElementFamilyRequirement::Hcurl && element.topological_dimension == 2
    }));
    assert!(requirements.elements.iter().any(|element| {
        element.family == ElementFamilyRequirement::Hdiv && element.topological_dimension == 2
    }));

    let curl = requirements
        .spaces
        .iter()
        .find(|space| space.pullback == PullbackRequirement::CovariantPiola)
        .unwrap();
    assert!(
        curl.orientations
            .contains(&OrientationRequirement::EdgeTangential)
    );
    assert!(
        curl.orientations
            .contains(&OrientationRequirement::TwoSidedFacet)
    );
    assert!(curl.traces.contains(&TraceRequirement::MinusTangential));

    let div_spaces = requirements
        .spaces
        .iter()
        .filter(|space| space.pullback == PullbackRequirement::ContravariantPiola)
        .collect::<Vec<_>>();
    assert!(
        div_spaces
            .iter()
            .any(|space| space.traces.contains(&TraceRequirement::MinusNormal))
    );
    assert!(
        div_spaces
            .iter()
            .any(|space| space.traces.contains(&TraceRequirement::PlusNormal))
    );

    let dg = requirements
        .spaces
        .iter()
        .find(|space| {
            space.pullback == PullbackRequirement::Broken
                && space.traces.contains(&TraceRequirement::MinusValue)
        })
        .unwrap();
    assert!(
        dg.orientations
            .contains(&OrientationRequirement::TwoSidedFacet)
    );
}

#[test]
fn preprocessing_and_quadrature_intent_are_explicit() {
    let compilation = compile_semantics(
        r#"
module requirements.preprocessing;
model Diffusion {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: trial scalar H1(order=2) on Omega;
  field v: test scalar H1(order=2) on Omega;
  field k: coefficient scalar L2(order=1) on Omega;
  form residual { cell(Omega): k * dot(grad(u), grad(v)); }
}
"#,
        &UnitRegistry::si_bootstrap(),
    )
    .unwrap();
    let form = compile_variational_form(&compilation.semantic, "Diffusion", "residual").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let signature = &requirements.integral_groups[0].signature;

    assert_eq!(signature.quadrature.minimum_polynomial_degree, 3);
    assert_eq!(
        signature.quadrature.precision,
        QuadraturePrecision::PolynomialExact
    );
    assert!(
        signature
            .geometry
            .contains(&GeometryPreprocessingRequirement::Jacobian)
    );
    assert!(
        signature
            .geometry
            .contains(&GeometryPreprocessingRequirement::InverseJacobian)
    );
    assert!(
        signature
            .geometry
            .contains(&GeometryPreprocessingRequirement::JacobianDeterminant)
    );
    assert_eq!(
        signature
            .inputs
            .iter()
            .filter(|input| {
                input.evaluations.iter().any(|evaluation| {
                    evaluation.derivative == DerivativeEvaluation::Gradient
                        && evaluation.site == EvaluationSite::Cell
                        && evaluation.trace_mapping.is_none()
                })
            })
            .count(),
        2
    );
}

#[test]
fn integral_grouping_is_complete_and_invariant_to_integral_order() {
    let source = |integrals: &str| {
        format!(
            r#"
module requirements.order;
model Ordered {{
  domain Omega {{ dimension = 2; coordinates = cartesian; }}
  field u: trial scalar H1(order=1) on Omega;
  field v: test scalar H1(order=1) on Omega;
  form residual {{ {integrals} }}
}}
"#
        )
    };
    let first = source(
        "cell(Omega): dot(grad(u), grad(v)); boundary(walls): u * v; cell(Omega): u * v; cell(Omega): dot(grad(u), grad(v));",
    );
    let second = source(
        "cell(Omega): u * v; cell(Omega): dot(grad(u), grad(v)); cell(Omega): dot(grad(u), grad(v)); boundary(walls): u * v;",
    );
    let registry = UnitRegistry::si_bootstrap();
    let first = compile_semantics(&first, &registry).unwrap();
    let second = compile_semantics(&second, &registry).unwrap();
    let first_form = compile_variational_form(&first.semantic, "Ordered", "residual").unwrap();
    let second_form = compile_variational_form(&second.semantic, "Ordered", "residual").unwrap();
    let first = infer_form_requirements(&first.semantic, &first_form).unwrap();
    let second = infer_form_requirements(&second.semantic, &second_form).unwrap();

    assert_eq!(first.integral_groups.len(), 3);
    assert_eq!(
        first
            .integral_groups
            .iter()
            .map(|group| group.occurrences.len())
            .collect::<Vec<_>>(),
        second
            .integral_groups
            .iter()
            .map(|group| group.occurrences.len())
            .collect::<Vec<_>>()
    );
    assert_eq!(first.artifact_digest, second.artifact_digest);
    assert_ne!(
        first.receipt.source_form_digest,
        second.receipt.source_form_digest
    );
}

#[test]
fn requirement_identity_is_invariant_to_domain_and_field_declaration_order() {
    let first = r#"
module requirements.declaration_order;
model Ordered {
  domain Used { dimension = 2; coordinates = cartesian; }
  domain Unused { dimension = 3; coordinates = cartesian; }
  field u: trial scalar H1(order=2) on Used;
  field v: test scalar H1(order=2) on Used;
  form residual { cell(Used): dot(grad(u), grad(v)); }
}
"#;
    let second = r#"
module requirements.declaration_order;
model Ordered {
  domain Unused { dimension = 3; coordinates = cartesian; }
  domain Used { dimension = 2; coordinates = cartesian; }
  field v: test scalar H1(order=2) on Used;
  field u: trial scalar H1(order=2) on Used;
  form residual { cell(Used): dot(grad(u), grad(v)); }
}
"#;
    let registry = UnitRegistry::si_bootstrap();
    let first = compile_semantics(first, &registry).unwrap();
    let second = compile_semantics(second, &registry).unwrap();
    let first_form = compile_variational_form(&first.semantic, "Ordered", "residual").unwrap();
    let second_form = compile_variational_form(&second.semantic, "Ordered", "residual").unwrap();
    let first = infer_form_requirements(&first.semantic, &first_form).unwrap();
    let second = infer_form_requirements(&second.semantic, &second_form).unwrap();

    assert_eq!(first.artifact_digest, second.artifact_digest);
    assert_ne!(
        first.receipt.source_form_digest,
        second.receipt.source_form_digest
    );
}

#[test]
fn coefficient_dependent_quadrature_is_not_claimed_exact() {
    let compilation = compile_semantics(
        r#"
module requirements.coefficient;
model Nonlinear {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: trial scalar H1(order=1) on Omega;
  field v: test scalar H1(order=1) on Omega;
  property k = conductivity(0);
  form residual { cell(Omega): k * dot(grad(u), grad(v)); }
}
"#,
        &UnitRegistry::si_bootstrap(),
    )
    .unwrap();
    let form = compile_variational_form(&compilation.semantic, "Nonlinear", "residual").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    assert_eq!(
        requirements.integral_groups[0]
            .signature
            .quadrature
            .precision,
        QuadraturePrecision::CoefficientDependent
    );
}

#[test]
fn derived_poisson_emits_essential_constraints_and_boundary_partition() {
    let compilation = compile_semantics(
        r#"
module requirements.poisson;
model Poisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  equation balance on Omega { -div(grad(u)) = 1; }
  boundary walls on boundary("walls") { dirichlet u = 0; }
}
"#,
        &UnitRegistry::si_bootstrap(),
    )
    .unwrap();
    let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();

    assert_eq!(requirements.essential_constraints.len(), 1);
    assert_eq!(requirements.boundary_partitions.len(), 1);
    assert_eq!(
        requirements.boundary_partitions[0].exterior_regions.len(),
        1
    );
}

#[test]
fn incompatible_spaces_axes_measures_and_boundary_data_are_refused() {
    let registry = UnitRegistry::si_bootstrap();
    let l2 = compile_semantics(
        r#"
module requirements.invalid.space;
model Invalid {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: trial scalar L2(order=1) on Omega;
  field v: test scalar L2(order=1) on Omega;
  form residual { cell(Omega): dot(grad(u), grad(v)); }
}
"#,
        &registry,
    )
    .unwrap();
    let l2_form = compile_variational_form(&l2.semantic, "Invalid", "residual").unwrap();
    assert!(matches!(
        infer_form_requirements(&l2.semantic, &l2_form),
        Err(RequirementInferenceError::IncompatibleDifferentialSpace { .. })
    ));

    let axes = compile_semantics(
        r#"
module requirements.invalid.axes;
model Invalid {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: trial vector(2) H1(order=1) on Omega;
  form residual { cell(Omega): u; }
}
"#,
        &registry,
    )
    .unwrap();
    let axes_form = compile_variational_form(&axes.semantic, "Invalid", "residual").unwrap();
    assert!(matches!(
        infer_form_requirements(&axes.semantic, &axes_form),
        Err(RequirementInferenceError::NonScalarIntegrand { .. })
    ));

    let domains = compile_semantics(
        r#"
module requirements.invalid.measure;
model Invalid {
  domain A { dimension = 2; coordinates = cartesian; }
  domain B { dimension = 2; coordinates = cartesian; }
  field u: trial scalar H1(order=1) on A;
  field v: test scalar H1(order=1) on A;
  form residual { cell(A): u * v; }
}
"#,
        &registry,
    )
    .unwrap();
    let mut domains_form =
        compile_variational_form(&domains.semantic, "Invalid", "residual").unwrap();
    domains_form.integrals[0].measure = SemanticMeasure::Cell {
        domain: domains.semantic.models[0].domains[1].id,
    };
    assert!(matches!(
        infer_form_requirements(&domains.semantic, &domains_form),
        Err(RequirementInferenceError::IncompatibleMeasureDomains { .. })
    ));

    let poisson = compile_semantics(
        r#"
module requirements.invalid.boundary;
model Poisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  equation balance on Omega { -div(grad(u)) = 1; }
  boundary walls on boundary("walls") { dirichlet u = 0; }
}
"#,
        &registry,
    )
    .unwrap();
    let mut poisson_form =
        derive_variational_form(&poisson.semantic, "Poisson", "balance").unwrap();
    let equation = poisson.semantic.models[0]
        .declarations
        .iter()
        .find(|declaration| declaration.name == "balance")
        .unwrap()
        .id;
    for assumption in &mut poisson_form.receipt.assumptions {
        if let FormAssumption::TestTraceVanishes { condition, .. } = assumption {
            *condition = equation;
        }
    }
    assert!(matches!(
        infer_form_requirements(&poisson.semantic, &poisson_form),
        Err(RequirementInferenceError::IncompatibleBoundaryData { .. })
    ));
}

#[test]
fn normal_trace_of_hcurl_is_refused() {
    let compilation = compile_semantics(
        r#"
module requirements.invalid.trace;
model Invalid {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: trial vector(2) HCurl(order=1) on Omega;
  field v: test vector(2) HCurl(order=1) on Omega;
  form residual {
    boundary(walls): normal_component(u) * normal_component(v);
  }
}
"#,
        &UnitRegistry::si_bootstrap(),
    )
    .unwrap();
    let form = compile_variational_form(&compilation.semantic, "Invalid", "residual").unwrap();
    assert!(matches!(
        infer_form_requirements(&compilation.semantic, &form),
        Err(RequirementInferenceError::IncompatibleNormalTrace { .. })
    ));
}

#[test]
fn hdiv_normal_preprocessing_names_normal_trace_and_geometry() {
    let compilation = compile_semantics(
        r#"
module requirements.hdiv;
model Flux {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: trial vector(2) HDiv(order=1) on Omega;
  field v: test vector(2) HDiv(order=1) on Omega;
  form residual {
    boundary(walls): normal_component(u) * normal_component(v);
  }
}
"#,
        &UnitRegistry::si_bootstrap(),
    )
    .unwrap();
    let form = compile_variational_form(&compilation.semantic, "Flux", "residual").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let signature = &requirements.integral_groups[0].signature;
    assert!(
        signature
            .geometry
            .contains(&GeometryPreprocessingRequirement::FacetNormal)
    );
    assert!(signature.inputs.iter().all(|input| {
        input.evaluations.iter().all(|evaluation| {
            evaluation.site == EvaluationSite::ExteriorTrace
                && evaluation.trace_mapping == Some(TraceMapping::Normal)
        })
    }));
}
