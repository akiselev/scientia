//! Batch P (`sinbad/ARCHITECTURE.md` §12 "P" item 1): the natural-boundary trace-shape
//! refusal. Before this package a derived equation with a constitutive flux and no boundary
//! condition (08's `div(current_density)` with `current_density = -sigma(T) * grad(V)`) kept a
//! state-computed facet integral for the implicit natural boundary, and FC3 pushed the
//! `Normal` trace mapping onto every leaf of the constitutive expansion -- including the scalar
//! `T` inside `sigma(T)` -- so FC4 refused with `TENSOR_SHAPE: normal trace requires a vector or
//! rank-two tensor`. Now:
//!
//! - a region with no boundary condition for the test field is *naturally closed*: the zero
//!   flux datum is substituted, no facet integral is emitted, and the receipt records
//!   `BoundaryTermDisposition::NaturallyClosed { flux }` with the strong flux expression;
//! - a `Normal` trace mapping is redistributed by shape: it stays on the operand carrying the
//!   contracted axis and degenerates to a plain trace on scalar factors, an opaque model-defined
//!   symbol's definition is evaluated at the facet as one value, and inline nonlinear tensor
//!   structure is refused typed (`REQ_NORMAL_TRACE_NONLINEAR`, `REQ_NORMAL_TRACE_UNSUPPORTED`);
//! - provider calls anywhere inside a residual term (or boundary value) are lifted into
//!   compiler-synthesized property captures, so `rho * convect(u, u)` and a bare
//!   `reaction(c, phi)` source bind like a named `property`.

use quantitas::UnitRegistry;
use scientia::{
    BasisEvaluationRequirement, BoundaryTermDisposition, DerivativeEvaluation, EvaluationSite,
    FormRequirements, FormTransformation, InputPreprocessingRequirement, InputSourceRequirement,
    SemanticExprKind, SemanticMeasure, SymbolId, TensorInputRole, TraceMapping, compile_semantics,
    compile_variational_form, derive_variational_form, factor_operator, infer_form_requirements,
    lower_operator_kernels,
};

/// Every per-integral-group input requirement of `requirements`, flattened.
fn all_inputs(
    requirements: &FormRequirements,
) -> impl Iterator<Item = &InputPreprocessingRequirement> {
    requirements
        .integral_groups
        .iter()
        .flat_map(|group| group.signature.inputs.iter())
}

/// Shaped like `08-electrothermal-joule.res`'s `electrical` equation: a constitutive flux over a
/// temperature-dependent provider property, no boundary declaration at all.
const CONSTITUTIVE_FLUX_NO_BOUNDARY: &str = r#"
module p.constitutive_flux;
model ConstitutiveFlux {
    domain Omega { dimension = 2; coordinates = cartesian; }
    field V: unknown scalar H1(order=1) on Omega;
    field T: state scalar H1(order=1) on Omega {
        quantity = ThermodynamicTemperature; unit = K; time_role = differential;
    };
    provider electrical_conductivity(T: ThermodynamicTemperature) -> ElectricalConductivity { differentiability = symbolic; }
    property sigma = electrical_conductivity(T);
    constitutive current_density = -sigma * grad(V);
    equation electrical on Omega { div(current_density) = 0; }
}
"#;

#[test]
fn constitutive_flux_with_no_boundary_condition_is_naturally_closed_and_factors() {
    let compilation =
        compile_semantics(CONSTITUTIVE_FLUX_NO_BOUNDARY, &UnitRegistry::si_bootstrap()).unwrap();
    let form =
        derive_variational_form(&compilation.semantic, "ConstitutiveFlux", "electrical").unwrap();
    let model = &compilation.semantic.models[0];
    let synthetic_region = scientia::RegionId::generated_for(model.domains[0].id);

    // One naturally closed boundary term, no facet integral, strong flux recorded.
    assert_eq!(form.receipt.boundary_terms.len(), 1);
    let term = &form.receipt.boundary_terms[0];
    assert_eq!(term.region, synthetic_region);
    let BoundaryTermDisposition::NaturallyClosed { flux } = term.disposition else {
        panic!("expected NaturallyClosed, got {:?}", term.disposition);
    };
    let current_density = model
        .symbols
        .iter()
        .find(|symbol| symbol.name == "current_density")
        .unwrap()
        .id;
    let SemanticExprKind::NormalComponent { value, .. } = form.expressions[flux.index()].kind
    else {
        panic!("the strong flux of a divergence term is the operand's normal component");
    };
    assert!(matches!(
        form.expressions[value.index()].kind,
        SemanticExprKind::Symbol { symbol } if symbol == current_density
    ));
    assert!(
        form.integrals
            .iter()
            .all(|integral| matches!(integral.measure, SemanticMeasure::Cell { .. }))
    );
    assert!(form.receipt.transformations.iter().any(|transformation| matches!(
        transformation,
        FormTransformation::SubstituteNaturalClosure { region } if *region == synthetic_region
    )));

    // FC3 no longer asks for a "normal trace" of anything: the closed term is gone, so every
    // evaluation is a cell evaluation.
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    for input in all_inputs(&requirements) {
        for evaluation in &input.evaluations {
            assert_eq!(evaluation.site, EvaluationSite::Cell, "{input:?}");
            assert_eq!(evaluation.trace_mapping, None, "{input:?}");
        }
    }

    // FC4 and FC5 succeed, which is the whole point of batch P.
    let factorization = factor_operator(&form, &requirements).unwrap();
    assert_eq!(factorization.integrals.len(), form.integrals.len());
    let kernels = lower_operator_kernels(&factorization).unwrap();
    assert_eq!(kernels.bundles.len(), factorization.integrals.len());
}

/// Authored form: `normal_component(k * grad(u))` written inline, where `k` is a scalar
/// provider property. The normal mapping must land on `grad(u)` only.
const INLINE_NORMAL_COMPONENT: &str = r#"
module p.inline_normal;
model InlineNormal {
    domain Omega { dimension = 2; coordinates = cartesian; }
    field u: trial scalar H1(order=1) on Omega;
    field v: test scalar H1(order=1) on Omega;
    provider conductivity(u: Concentration) -> Diffusivity { differentiability = symbolic; }
    property k = conductivity(u);
    form flux_pairing {
        cell(Omega): dot(k * grad(u), grad(v));
        boundary(walls): normal_component(k * grad(u)) * v;
    }
}
"#;

#[test]
fn an_inline_normal_component_maps_only_the_axis_carrying_operand() {
    let compilation =
        compile_semantics(INLINE_NORMAL_COMPONENT, &UnitRegistry::si_bootstrap()).unwrap();
    let form =
        compile_variational_form(&compilation.semantic, "InlineNormal", "flux_pairing").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let model = &compilation.semantic.models[0];
    let symbol = |name: &str| model.symbols.iter().find(|s| s.name == name).unwrap().id;
    let evaluations = |symbol: SymbolId| -> Vec<BasisEvaluationRequirement> {
        all_inputs(&requirements)
            .filter(|input| input.symbol == symbol)
            .flat_map(|input| input.evaluations.iter().cloned())
            .collect()
    };
    // `k` is scalar: its facet evaluation is a plain value trace, never a normal trace.
    assert!(evaluations(symbol("k")).iter().any(|evaluation| {
        evaluation.site == EvaluationSite::ExteriorTrace
            && evaluation.derivative == DerivativeEvaluation::Value
            && evaluation.trace_mapping == Some(TraceMapping::Value)
    }));
    assert!(
        evaluations(symbol("k"))
            .iter()
            .all(|evaluation| { evaluation.trace_mapping != Some(TraceMapping::Normal) })
    );
    // `grad(u)` carries the axis: it is the one normal-mapped evaluation.
    assert!(evaluations(symbol("u")).iter().any(|evaluation| {
        evaluation.site == EvaluationSite::ExteriorTrace
            && evaluation.derivative == DerivativeEvaluation::Gradient
            && evaluation.trace_mapping == Some(TraceMapping::Normal)
    }));
    // FC4 accepts the same redistribution: the facet integral factors with `k` as a scalar
    // External input and `grad(u)·n` as a scalar Active input.
    let factorization = factor_operator(&form, &requirements).unwrap();
    let facet = factorization
        .integrals
        .iter()
        .find(|integral| matches!(integral.measure, SemanticMeasure::ExteriorFacet { .. }))
        .unwrap();
    let k_input = facet
        .primal
        .inputs
        .iter()
        .find(|input| input.binding.symbol == symbol("k"))
        .unwrap();
    assert!(k_input.shape.is_empty());
    assert_eq!(k_input.role, TensorInputRole::External);
    let u_input = facet
        .primal
        .inputs
        .iter()
        .find(|input| {
            input.binding.symbol == symbol("u")
                && input.binding.evaluation.derivative == DerivativeEvaluation::Gradient
        })
        .unwrap();
    assert!(
        u_input.shape.is_empty(),
        "grad(u)·n is a scalar: {u_input:?}"
    );
    assert_eq!(
        u_input.binding.evaluation.trace_mapping,
        Some(TraceMapping::Normal)
    );
}

/// Inline normal trace of a computed vector (a vector-valued provider call) cannot be expressed
/// by leaf mappings and is refused typed rather than mis-shaped.
const INLINE_NORMAL_OF_COMPUTED_VECTOR: &str = r#"
module p.inline_normal_computed;
model InlineNormalComputed {
    domain Omega { dimension = 2; coordinates = cartesian; }
    field u: trial scalar H1(order=1) on Omega;
    field v: test scalar H1(order=1) on Omega;
    provider drift(u: Concentration) -> Velocity { shape = vector(2); differentiability = symbolic; }
    form drift_pairing {
        cell(Omega): u * v;
        boundary(walls): normal_component(drift(u)) * v;
    }
}
"#;

#[test]
fn an_inline_normal_trace_of_a_computed_vector_is_refused_typed() {
    let compilation = compile_semantics(
        INLINE_NORMAL_OF_COMPUTED_VECTOR,
        &UnitRegistry::si_bootstrap(),
    )
    .unwrap();
    let form = compile_variational_form(
        &compilation.semantic,
        "InlineNormalComputed",
        "drift_pairing",
    )
    .unwrap();
    let error = infer_form_requirements(&compilation.semantic, &form).unwrap_err();
    assert!(
        error
            .to_string()
            .starts_with("REQ_NORMAL_TRACE_UNSUPPORTED"),
        "{error}"
    );
}

/// The same computed vector named in a `constitutive` symbol is fine: the symbol is an opaque
/// facet input whose value is contracted with the normal, and its definition's leaves are
/// plain traces.
const NAMED_NORMAL_OF_COMPUTED_VECTOR: &str = r#"
module p.named_normal_computed;
model NamedNormalComputed {
    domain Omega { dimension = 2; coordinates = cartesian; }
    field u: trial scalar H1(order=1) on Omega;
    field v: test scalar H1(order=1) on Omega;
    provider drift(u: Concentration) -> Velocity { shape = vector(2); differentiability = symbolic; }
    constitutive flux = u * drift(u);
    form drift_pairing {
        cell(Omega): u * v;
        boundary(walls): normal_component(flux) * v;
    }
}
"#;

#[test]
fn a_named_flux_under_a_normal_trace_leaves_its_definition_leaves_as_plain_traces() {
    let compilation = compile_semantics(
        NAMED_NORMAL_OF_COMPUTED_VECTOR,
        &UnitRegistry::si_bootstrap(),
    )
    .unwrap();
    let form = compile_variational_form(
        &compilation.semantic,
        "NamedNormalComputed",
        "drift_pairing",
    )
    .unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let model = &compilation.semantic.models[0];
    let symbol = |name: &str| model.symbols.iter().find(|s| s.name == name).unwrap().id;
    let flux = all_inputs(&requirements)
        .find(|input| input.symbol == symbol("flux"))
        .unwrap();
    assert!(matches!(
        flux.source,
        InputSourceRequirement::ModelDefinedConstitutive { .. }
    ));
    assert!(flux.evaluations.iter().any(|evaluation| {
        evaluation.site == EvaluationSite::ExteriorTrace
            && evaluation.trace_mapping == Some(TraceMapping::Normal)
    }));
    assert!(
        all_inputs(&requirements)
            .filter(|input| input.symbol == symbol("u"))
            .flat_map(|input| input.evaluations.iter())
            .all(|evaluation| evaluation.trace_mapping != Some(TraceMapping::Normal))
    );
    let factorization = factor_operator(&form, &requirements).unwrap();
    let facet = factorization
        .integrals
        .iter()
        .find(|integral| matches!(integral.measure, SemanticMeasure::ExteriorFacet { .. }))
        .unwrap();
    let flux_input = facet
        .primal
        .inputs
        .iter()
        .find(|input| input.binding.symbol == symbol("flux"))
        .unwrap();
    assert!(
        flux_input.shape.is_empty(),
        "flux·n is scalar: {flux_input:?}"
    );
}

/// Shaped like `27-boussinesq-convection.res`'s momentum equation: a vector-valued provider
/// call nested in a product (`rho * convect(u, u)`) and a scalar one as a bare source term.
const NESTED_PROVIDER_CALLS: &str = r#"
module p.nested_provider_calls;
model NestedProviderCalls {
    domain Omega { dimension = 2; coordinates = cartesian; }
    field u: state vector(2) H1(order=2) on Omega { time_role = differential; };
    field c: state scalar H1(order=1) on Omega { time_role = differential; };
    provider density(c: Concentration) -> Density { differentiability = symbolic; }
    provider convect(u: Velocity, u2: Velocity) -> Acceleration { shape = vector(2); differentiability = symbolic; }
    provider reaction(c: Concentration) -> SpeciesSource { differentiability = symbolic; }
    property rho = density(c);
    equation momentum on Omega { rho * dt(u) + rho * convect(u, u) = 0; }
    equation species on Omega { dt(c) = reaction(c); }
}
"#;

#[test]
fn provider_calls_inside_residual_terms_are_lifted_into_property_captures() {
    let compilation =
        compile_semantics(NESTED_PROVIDER_CALLS, &UnitRegistry::si_bootstrap()).unwrap();
    for (equation, expected_shape) in [("momentum", vec![2usize]), ("species", vec![])] {
        let form = derive_variational_form(&compilation.semantic, "NestedProviderCalls", equation)
            .unwrap();
        let lifted = form
            .captures
            .iter()
            .filter(|capture| capture.symbol.is_generated() && capture.definition.is_some())
            .collect::<Vec<_>>();
        assert_eq!(
            lifted.len(),
            1,
            "{equation}: one lifted provider call: {lifted:?}"
        );
        let definition = lifted[0].definition.unwrap();
        assert!(matches!(
            compilation.semantic.models[0].expressions[definition.index()].kind,
            SemanticExprKind::ProviderCall { .. }
        ));
        let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
        let factorization = factor_operator(&form, &requirements).unwrap();
        let input = factorization
            .integrals
            .iter()
            .flat_map(|integral| integral.primal.inputs.iter())
            .find(|input| input.binding.symbol == lifted[0].symbol)
            .expect("the lifted call is a QFunction input");
        assert_eq!(input.role, TensorInputRole::External);
        assert!(matches!(
            input.source,
            InputSourceRequirement::ModelDefinedProperty { .. }
        ));
        assert_eq!(input.shape, expected_shape);
        lower_operator_kernels(&factorization).unwrap();
    }
}
