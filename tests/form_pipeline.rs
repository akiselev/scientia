use quantitas::UnitRegistry;
use scientia::{
    BoundaryTermDisposition, DifferentialOperator, FormArgumentRole, FormAssumption,
    FormCompileError, FormComplexConvention, FormEvaluation, FormEvaluationContext, FormSample,
    FormSide, FormTransformation, FormValue, InputEvaluation, LocalInputRole,
    LocalIterationContract, RegionKind, SemanticExprKind, SemanticMeasure, SemanticShape,
    TraceSide, compile_semantics, compile_variational_form, derive_variational_form,
    derive_variational_form_for, factor_local_integral, interpret_form, interpret_integral,
    lower_local_program, required_evaluations, semantic_arena_digest,
};

const FORM_SOURCE: &str = r#"
module pipeline.test;
model Diffusion {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: trial scalar H1(order=2) on Omega;
  field v: test scalar H1(order=2) on Omega;
  field conductivity: coefficient scalar L2(order=1) on Omega;
  parameter reaction: Rate;
  form residual {
    cell(Omega): conductivity * dot(grad(u), grad(v));
    cell(Omega): reaction * u * v;
  }
}
"#;

// Repository-local FC2 fixtures. Their mathematical shapes mirror Sinbad's product corpus, but
// Scientia tests must remain buildable without a sibling Sinbad checkout.
const POISSON_SOURCE: &str = r#"
module fixture.poisson;
model Poisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  property k = diffusivity(0);
  source f: VolumetricSource;
  equation balance on Omega { -div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(); }
}
"#;

const TRANSIENT_DIFFUSION_SOURCE: &str = r#"
module fixture.transient_diffusion;
model TransientDiffusion {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: state scalar H1(order=1) on Omega { time_role = differential; };
  property capacity = storage_capacity(u);
  property k = diffusivity(u);
  source f: VolumetricSource;
  equation evolution on Omega { capacity * dt(u) - div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(t); }
}
"#;

const NONLINEAR_HEAT_SOURCE: &str = r#"
module fixture.nonlinear_heat;
model NonlinearHeat {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega {
    quantity = ThermodynamicTemperature;
    unit = K;
    time_role = differential;
  };
  property rho = density(T);
  property cp = specific_heat(T);
  property k = thermal_conductivity(T);
  source Q: VolumetricHeatSource;
  equation energy on Omega { rho * cp * dt(T) - div(k * grad(T)) = Q; }
  boundary walls on boundary("walls") { dirichlet T = exact_T(t); }
}
"#;

const ELASTICITY_SOURCE: &str = r#"
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
"#;

const STOKES_SOURCE: &str = r#"
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
"#;

#[test]
fn typed_identities_and_roles_survive_form_factorization() {
    let compilation = compile_semantics(FORM_SOURCE, &UnitRegistry::si_bootstrap()).unwrap();
    let form = compile_variational_form(&compilation.semantic, "Diffusion", "residual").unwrap();

    assert_eq!(
        form.source_semantic_digest.hex,
        semantic_arena_digest(&compilation.semantic)
    );
    assert_eq!(form.arguments[0].role, FormArgumentRole::Trial);
    assert_eq!(form.arguments[1].role, FormArgumentRole::Test);
    assert!(matches!(
        form.integrals[0].measure,
        SemanticMeasure::Cell { .. }
    ));

    let diffusion = factor_local_integral(&form, 0).unwrap();
    assert_eq!(diffusion.iteration, LocalIterationContract::QuadraturePoint);
    assert_eq!(diffusion.source_form_digest, form.artifact_digest);
    assert_eq!(diffusion.receipt.source_form_digest, form.artifact_digest);
    assert_eq!(diffusion.inputs[0].role, LocalInputRole::TrialBasis);
    assert_eq!(diffusion.inputs[1].role, LocalInputRole::TestBasis);
    assert_eq!(
        diffusion.inputs[2].role,
        LocalInputRole::PhysicalField(scientia::scientific::FieldRole::Coefficient)
    );
    assert!(
        diffusion.inputs[..2]
            .iter()
            .all(|input| input.evaluations == [InputEvaluation::Gradient])
    );

    let reaction = factor_local_integral(&form, 1).unwrap();
    let lowered = lower_local_program(&reaction).unwrap();
    assert_eq!(
        lowered.receipt.source_program_digest,
        reaction.artifact_digest
    );
    assert_eq!(
        lowered.receipt.iteration,
        LocalIterationContract::QuadraturePoint
    );
    malleus::validate(lowered.kernel).unwrap();
}

#[test]
fn presentation_changes_do_not_change_form_or_program_digests() {
    let registry = UnitRegistry::si_bootstrap();
    let compact = compile_semantics(FORM_SOURCE, &registry).unwrap();
    let formatted = scientia::format_scientific_module(&compact.source);
    let formatted = compile_semantics(&formatted, &registry).unwrap();
    let first = compile_variational_form(&compact.semantic, "Diffusion", "residual").unwrap();
    let second = compile_variational_form(&formatted.semantic, "Diffusion", "residual").unwrap();

    assert_eq!(first.artifact_digest, second.artifact_digest);
    assert_eq!(
        factor_local_integral(&first, 0).unwrap().artifact_digest,
        factor_local_integral(&second, 0).unwrap().artifact_digest
    );
}

#[test]
fn facet_measures_resolve_to_region_ids() {
    let compilation = compile_semantics(
        r#"
module facet.test;
model Flux {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: trial scalar H1(order=1) on Omega;
  field v: test scalar H1(order=1) on Omega;
  form boundary_residual { boundary(walls): u * v; }
}
"#,
        &UnitRegistry::si_bootstrap(),
    )
    .unwrap();
    let model = &compilation.semantic.models[0];
    let form =
        compile_variational_form(&compilation.semantic, "Flux", "boundary_residual").unwrap();
    let SemanticMeasure::ExteriorFacet { region } = form.integrals[0].measure else {
        panic!("boundary measure did not retain a typed exterior-facet region")
    };

    assert_eq!(model.regions[region.index()].name, "walls");
    assert_eq!(
        model.regions[region.index()].kind,
        RegionKind::ExteriorFacet
    );
}

#[test]
fn fc2_gate_models_derive_typed_forms_or_explicit_equation_forms() {
    let fixtures = [
        (POISSON_SOURCE, "Poisson", &["balance"][..]),
        (
            TRANSIENT_DIFFUSION_SOURCE,
            "TransientDiffusion",
            &["evolution"][..],
        ),
        (NONLINEAR_HEAT_SOURCE, "NonlinearHeat", &["energy"][..]),
        (ELASTICITY_SOURCE, "LinearElasticity", &["momentum"][..]),
        (
            STOKES_SOURCE,
            "StokesFlow",
            &["momentum", "incompressibility"][..],
        ),
    ];
    for (source, model, equations) in fixtures {
        let compilation = compile_semantics(source, &UnitRegistry::si_bootstrap()).unwrap();
        for equation in equations {
            let form = derive_variational_form(&compilation.semantic, model, equation).unwrap();
            assert_eq!(form.arity.test, 1, "{model}::{equation}");
            assert_eq!(form.arity.trial, 0, "{model}::{equation}");
            assert!(!form.integrals.is_empty(), "{model}::{equation}");
            assert!(
                form.integrals
                    .iter()
                    .all(|integral| integral.side == FormSide::Cell),
                "the gate fixtures impose essential exterior conditions"
            );
        }
    }
}

#[test]
fn poisson_derivation_records_ibp_and_eliminated_boundary_term() {
    let compilation = compile_semantics(POISSON_SOURCE, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();

    assert!(form.receipt.transformations.iter().any(|transformation| {
        matches!(
            transformation,
            FormTransformation::IntegrateByParts {
                operator: scientia::DifferentialOperator::Divergence,
                ..
            }
        )
    }));
    assert_eq!(form.receipt.boundary_terms.len(), 1);
    assert_eq!(
        form.receipt.complex_convention,
        FormComplexConvention::ExplicitConjugationOnly
    );
    assert!(form.receipt.assumptions.iter().any(|assumption| matches!(
        assumption,
        FormAssumption::ExteriorRegionsPartitionBoundary { regions, .. } if regions.len() == 1
    )));
    assert!(matches!(
        form.receipt.boundary_terms[0].disposition,
        BoundaryTermDisposition::EliminatedByEssentialCondition { .. }
    ));
}

#[test]
fn two_sided_facets_require_explicit_trace_sides() {
    let compilation = compile_semantics(
        r#"
module sides.test;
model Flux {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: trial scalar DG(order=1) on Omega;
  field v: test scalar DG(order=1) on Omega;
  form invalid { interior_facet(faces): trace(u) * trace(v); }
  form ambiguous { interior_facet(faces): u * v; }
  form valid { interior_facet(faces): trace_minus(u) * trace_plus(v); }
}
"#,
        &UnitRegistry::si_bootstrap(),
    )
    .unwrap();
    assert!(matches!(
        compile_variational_form(&compilation.semantic, "Flux", "invalid"),
        Err(FormCompileError::InvalidSideSemantics {
            code: "FORM_INVALID_SIDE",
            ..
        })
    ));
    assert!(matches!(
        compile_variational_form(&compilation.semantic, "Flux", "ambiguous"),
        Err(FormCompileError::InvalidSideSemantics {
            code: "FORM_INVALID_SIDE",
            ..
        })
    ));
    let valid = compile_variational_form(&compilation.semantic, "Flux", "valid").unwrap();
    assert_eq!(valid.integrals[0].side, FormSide::Interior);
}

#[test]
fn deterministic_form_interpreter_evaluates_scalar_and_contraction_forms() {
    let compilation = compile_semantics(
        r#"
module interpreter.test;
model Algebra {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: trial vector(2) H1(order=1) on Omega;
  field v: test vector(2) H1(order=1) on Omega;
  parameter alpha: Dimensionless;
  form bilinear { cell(Omega): alpha * dot(u, v); }
}
"#,
        &UnitRegistry::si_bootstrap(),
    )
    .unwrap();
    let form = compile_variational_form(&compilation.semantic, "Algebra", "bilinear").unwrap();
    let mut context = FormEvaluationContext::default();
    for argument in &form.arguments {
        context.bind_symbol(
            argument.symbol,
            match argument.role {
                FormArgumentRole::Trial => FormValue::vector(vec![2.0, 3.0]),
                FormArgumentRole::Test => FormValue::vector(vec![5.0, 7.0]),
            },
        );
    }
    context.bind_symbol(form.captures[0].symbol, FormValue::real(2.0));

    assert_eq!(
        interpret_integral(&form, 0, &context).unwrap(),
        FormValue::real(62.0)
    );
}

#[test]
fn interpreter_evaluates_derived_poisson_form_from_independent_point_data() {
    let compilation = compile_semantics(POISSON_SOURCE, &UnitRegistry::si_bootstrap()).unwrap();
    let model = &compilation.semantic.models[0];
    let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
    let field = model
        .symbols
        .iter()
        .find(|symbol| symbol.name == "u")
        .unwrap();
    let property = model
        .symbols
        .iter()
        .find(|symbol| symbol.name == "k")
        .unwrap();
    let source = model
        .symbols
        .iter()
        .find(|symbol| symbol.name == "f")
        .unwrap();
    let test = form.arguments[0].symbol;
    let mut context = FormEvaluationContext::default();
    context.bind_symbol(property.id, FormValue::real(2.0));
    context.bind_symbol(source.id, FormValue::real(7.0));
    context.bind_symbol(test, FormValue::real(2.0));
    for expression in &form.expressions {
        if let SemanticExprKind::Symbol { symbol } = expression.kind {
            if symbol == field.id {
                context.bind_evaluation(
                    expression.id,
                    FormEvaluation::Gradient,
                    FormValue::vector(vec![3.0, 4.0]),
                );
            } else if symbol == test {
                context.bind_evaluation(
                    expression.id,
                    FormEvaluation::Gradient,
                    FormValue::vector(vec![5.0, 6.0]),
                );
            }
        }
    }
    let samples = form
        .integrals
        .iter()
        .enumerate()
        .map(|(integral_index, _)| FormSample {
            integral_index,
            weight: 1.0,
            context: context.clone(),
        })
        .collect::<Vec<_>>();

    // 2 * [3,4] dot [5,6] - 7 * 2 = 64.
    assert_eq!(
        interpret_form(&form, &samples).unwrap(),
        FormValue::real(64.0)
    );
}

#[test]
fn jump_and_average_are_deterministic_two_sided_operations() {
    let compilation = compile_semantics(
        r#"
module jump.test;
model Flux {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: trial scalar DG(order=1) on Omega;
  field v: test scalar DG(order=1) on Omega;
  form flux { interior_facet(faces): jump(u) * average(v); }
}
"#,
        &UnitRegistry::si_bootstrap(),
    )
    .unwrap();
    let form = compile_variational_form(&compilation.semantic, "Flux", "flux").unwrap();
    let mut context = FormEvaluationContext::default();
    for expression in &form.expressions {
        if let SemanticExprKind::Symbol { symbol } = expression.kind {
            let name = &compilation.semantic.models[0].symbols[symbol.index()].name;
            let (minus, plus) = if name == "u" { (5.0, 2.0) } else { (7.0, 1.0) };
            context.bind_evaluation(
                expression.id,
                FormEvaluation::Trace(TraceSide::Minus),
                FormValue::real(minus),
            );
            context.bind_evaluation(
                expression.id,
                FormEvaluation::Trace(TraceSide::Plus),
                FormValue::real(plus),
            );
        }
    }

    assert_eq!(
        interpret_integral(&form, 0, &context).unwrap(),
        FormValue::real(12.0)
    );
}

#[test]
fn neumann_data_is_substituted_with_a_digest_linked_boundary_receipt() {
    let compilation = compile_semantics(
        r#"
module neumann.test;
model Poisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  equation balance on Omega { -div(grad(u)) = 0; }
  boundary load on boundary("load") { neumann u = 2; }
}
"#,
        &UnitRegistry::si_bootstrap(),
    )
    .unwrap();
    let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
    assert!(form.integrals.iter().any(|integral| {
        matches!(integral.measure, SemanticMeasure::ExteriorFacet { .. })
            && integral.side == FormSide::Exterior
    }));
    assert!(matches!(
        form.receipt.boundary_terms[0].disposition,
        BoundaryTermDisposition::Substituted { .. }
    ));
    assert!(form.receipt.transformations.iter().any(|transformation| {
        matches!(
            transformation,
            FormTransformation::SubstituteBoundaryCondition { .. }
        )
    }));
}

#[test]
fn interface_point_and_conjugation_semantics_are_explicit() {
    let compilation = compile_semantics(
        r#"
module complete.form.surface;
model Surface {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: trial vector(2) DG(order=1) on Omega;
  field v: test vector(2) DG(order=1) on Omega;
  form interface_form {
    interface(contact): inner(trace_minus(u), trace_plus(v));
  }
  form point_form { point(sensor): conj(u[0]) * v[0]; }
}
"#,
        &UnitRegistry::si_bootstrap(),
    )
    .unwrap();
    let interface =
        compile_variational_form(&compilation.semantic, "Surface", "interface_form").unwrap();
    let point = compile_variational_form(&compilation.semantic, "Surface", "point_form").unwrap();

    assert!(matches!(
        interface.integrals[0].measure,
        SemanticMeasure::Interface { .. }
    ));
    assert_eq!(interface.integrals[0].side, FormSide::Interface);
    assert!(interface.expressions.iter().any(|expression| matches!(
        expression.kind,
        SemanticExprKind::Contraction {
            conjugate_lhs: true,
            ..
        }
    )));
    assert!(matches!(
        point.integrals[0].measure,
        SemanticMeasure::Point { .. }
    ));
    assert_eq!(point.integrals[0].side, FormSide::Point);
    assert!(
        point
            .expressions
            .iter()
            .any(|expression| matches!(expression.kind, SemanticExprKind::Conjugate { .. }))
    );
}

#[test]
fn duplicate_neumann_substitution_is_refused() {
    let compilation = compile_semantics(
        r#"
module duplicate.neumann;
model Diffusion {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  equation balance on Omega { -div(grad(u)) - div(grad(u)) = 0; }
  boundary load on boundary("load") { neumann u = 2; }
}
"#,
        &UnitRegistry::si_bootstrap(),
    )
    .unwrap();

    assert!(matches!(
        derive_variational_form(&compilation.semantic, "Diffusion", "balance"),
        Err(FormCompileError::AmbiguousNeumannFlux { .. })
    ));
}

#[test]
fn zero_rhs_does_not_create_a_degenerate_integral() {
    let compilation = compile_semantics(STOKES_SOURCE, &UnitRegistry::si_bootstrap()).unwrap();
    let form =
        derive_variational_form(&compilation.semantic, "StokesFlow", "incompressibility").unwrap();

    assert_eq!(form.integrals.len(), 1);
}

#[test]
fn invalid_derived_differential_shape_is_diagnosed() {
    let compilation = compile_semantics(
        r#"
module invalid.derived.differential;
model GradientEquation {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  source vector_source: VectorSource;
  equation balance on Omega { grad(u) = vector_source; }
  boundary walls on boundary("walls") { dirichlet u = 0; }
}
"#,
        &UnitRegistry::si_bootstrap(),
    )
    .unwrap();
    let model = &compilation.semantic.models[0];
    let field = model
        .symbols
        .iter()
        .find(|symbol| symbol.name == "u")
        .unwrap();

    assert!(matches!(
        derive_variational_form_for(
            &compilation.semantic,
            "GradientEquation",
            "balance",
            field.id,
        ),
        Err(FormCompileError::InvalidDerivedDifferential {
            operator: DifferentialOperator::Divergence,
            shape: SemanticShape::Numeric(scientia::scientific::ValueShape::Scalar),
        })
    ));
}

#[test]
fn interpreter_requirements_are_available_without_walking_the_arena() {
    let compilation = compile_semantics(POISSON_SOURCE, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "Poisson", "balance").unwrap();
    let requirements = required_evaluations(&form, 0).unwrap();

    assert_eq!(
        requirements
            .iter()
            .filter(|requirement| requirement.evaluation == FormEvaluation::Gradient)
            .count(),
        2
    );
    assert!(
        requirements
            .iter()
            .any(|requirement| requirement.evaluation == FormEvaluation::Value)
    );
}
