use quantitas::{Dimension, UnitRegistry};
use scientia::{
    ActiveSet, DERIVATIVE_REQUEST_SCHEMA, DerivativeConvention, DerivativeDependence,
    DerivativeLevel, DerivativeProductSpec, DerivativeRequest, DerivativeStateConvention,
    DesignVariable, DifferentiabilityDisposition, Objective, ObjectiveSense, ObservableFunctional,
    ScalarConvention, ShapeDerivativeConvention, compile_semantics, derive_verification_profiles,
};

fn poisson(reordered: bool) -> String {
    let declarations = if reordered {
        r#"
  observable energy { integrate(0.5 * k * dot(grad(u), grad(u))); }
  @mms(field = u);
  boundary walls on boundary("walls") { dirichlet u = exact_u(); }
  equation balance on Omega { -div(k * grad(u)) = f; }
"#
    } else {
        r#"
  equation balance on Omega { -div(k * grad(u)) = f; }
  boundary walls on boundary("walls") { dirichlet u = exact_u(); }
  @mms(field = u);
  observable energy { integrate(0.5 * k * dot(grad(u), grad(u))); }
"#
    };
    format!(
        r#"module sv0.poisson;
model Poisson {{
  domain Omega {{ dimension = 2; coordinates = cartesian; }}
  field u: unknown scalar H1(order=1) on Omega;
  property k = diffusivity(0);
  source f: VolumetricSource;
  {declarations}
}}"#
    )
}

#[test]
fn poisson_obligations_are_deterministic_across_declaration_order() {
    let registry = UnitRegistry::si_bootstrap();
    let first = compile_semantics(&poisson(false), &registry).unwrap();
    let second = compile_semantics(&poisson(true), &registry).unwrap();
    let first = derive_verification_profiles(&first);
    let second = derive_verification_profiles(&second);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].artifact_digest, second[0].artifact_digest);
    assert_eq!(
        first[0]
            .obligations
            .iter()
            .map(|obligation| &obligation.id)
            .collect::<Vec<_>>(),
        second[0]
            .obligations
            .iter()
            .map(|obligation| &obligation.id)
            .collect::<Vec<_>>()
    );
    assert!(first[0].obligations.len() >= 4);
    assert_eq!(first[0].observables[0].name, "energy");
    first[0].validate().unwrap();
    let mut tampered = first[0].clone();
    tampered.obligations[0].expected_relation = "changed".into();
    assert_eq!(
        tampered.validate().unwrap_err().code,
        "VERIFY_OBLIGATION_IDENTITY_MISMATCH"
    );
}

#[test]
fn shape_request_requires_revision_stratum_boundaries_and_matching_disposition() {
    let request = DerivativeRequest {
        schema: DERIVATIVE_REQUEST_SCHEMA.into(),
        parent_semantic_digest: "semantic".into(),
        objective: Objective {
            name: "thermal_compliance".into(),
            functional: ObservableFunctional {
                name: "energy".into(),
                semantic_expression: "integrate(energy_density)".into(),
                dimension: Dimension::LENGTH,
            },
            sense: ObjectiveSense::Minimize,
        },
        design_variables: vec![
            DesignVariable {
                name: "inner_radius".into(),
                dimension: Dimension::LENGTH,
                parameter_owner: "cadabra3".into(),
                admissible_set: "0 < inner_radius < outer_radius".into(),
            },
            DesignVariable {
                name: "outer_radius".into(),
                dimension: Dimension::LENGTH,
                parameter_owner: "cadabra3".into(),
                admissible_set: "outer_radius > inner_radius".into(),
            },
        ],
        controls: vec![],
        product: DerivativeProductSpec::Gradient,
        active_set: ActiveSet {
            active: vec!["inner_radius".into()],
            frozen: vec!["outer_radius".into()],
        },
        evaluation_state: "converged steady solution".into(),
        convention: DerivativeConvention {
            dependence: DerivativeDependence::Total,
            level: DerivativeLevel::Discrete,
            scalar: ScalarConvention::Real,
            state: DerivativeStateConvention::ConvergedState,
            disposition: DifferentiabilityDisposition::Smooth,
            event_or_refusal_basis: None,
        },
        shape: Some(ShapeDerivativeConvention {
            geometry_revision: "cadabra-revision-7".into(),
            fixed_topology_stratum: "annulus-positive-clearance".into(),
            boundary_selection: vec!["inner".into(), "outer".into()],
            include_normal_variation: true,
            include_measure_variation: true,
            disposition: DifferentiabilityDisposition::Smooth,
        }),
        identity: scientia::Digest::blake3(&[]),
    }
    .finish()
    .unwrap();
    request.validate().unwrap();

    let mut tampered = request.clone();
    tampered.evaluation_state = "different state".into();
    assert_eq!(
        tampered.validate().unwrap_err().code,
        "DERIVATIVE_IDENTITY_MISMATCH"
    );

    let mut missing_revision = request.clone();
    missing_revision
        .shape
        .as_mut()
        .unwrap()
        .geometry_revision
        .clear();
    assert_eq!(
        missing_revision.validate().unwrap_err().code,
        "SHAPE_DERIVATIVE_INCOMPLETE"
    );

    let mut event_without_basis = request;
    event_without_basis.convention.disposition = DifferentiabilityDisposition::TopologyEvent;
    event_without_basis.shape.as_mut().unwrap().disposition =
        DifferentiabilityDisposition::TopologyEvent;
    assert_eq!(
        event_without_basis.validate().unwrap_err().code,
        "DERIVATIVE_MISSING_DISPOSITION_BASIS"
    );

    let mut whitespace_basis = event_without_basis.clone();
    whitespace_basis.convention.event_or_refusal_basis = Some("   \t".into());
    assert_eq!(
        whitespace_basis.validate().unwrap_err().code,
        "DERIVATIVE_MISSING_DISPOSITION_BASIS"
    );

    let mut blank_boundary = event_without_basis;
    blank_boundary.convention.disposition = DifferentiabilityDisposition::Smooth;
    blank_boundary.convention.event_or_refusal_basis = None;
    blank_boundary.shape.as_mut().unwrap().disposition = DifferentiabilityDisposition::Smooth;
    blank_boundary.shape.as_mut().unwrap().boundary_selection = vec!["   ".into()];
    assert_eq!(
        blank_boundary.validate().unwrap_err().code,
        "SHAPE_DERIVATIVE_INCOMPLETE"
    );
}

#[test]
fn derivative_requests_refuse_unbound_colliding_and_incomplete_namespaces() {
    let base = DerivativeRequest {
        schema: DERIVATIVE_REQUEST_SCHEMA.into(),
        parent_semantic_digest: "semantic".into(),
        objective: Objective {
            name: "objective".into(),
            functional: ObservableFunctional {
                name: "response".into(),
                semantic_expression: "integrate(u)".into(),
                dimension: Dimension::LENGTH,
            },
            sense: ObjectiveSense::Measure,
        },
        design_variables: vec![DesignVariable {
            name: "width".into(),
            dimension: Dimension::LENGTH,
            parameter_owner: "cadabra3".into(),
            admissible_set: "width > 0".into(),
        }],
        controls: vec![],
        product: DerivativeProductSpec::Gradient,
        active_set: ActiveSet {
            active: vec!["missing".into()],
            frozen: vec![],
        },
        evaluation_state: "converged".into(),
        convention: DerivativeConvention {
            dependence: DerivativeDependence::Total,
            level: DerivativeLevel::Discrete,
            scalar: ScalarConvention::Real,
            state: DerivativeStateConvention::ConvergedState,
            disposition: DifferentiabilityDisposition::Smooth,
            event_or_refusal_basis: None,
        },
        shape: None,
        identity: scientia::Digest::blake3(&[]),
    };
    assert_eq!(
        base.clone().finish().unwrap_err().code,
        "DERIVATIVE_INPUT_PARTITION"
    );

    let mut collision = base.clone();
    collision.active_set.active = vec!["width".into()];
    collision.controls = vec![scientia::Control {
        name: "width".into(),
        dimension: Dimension::LENGTH,
        support: "boundary".into(),
    }];
    assert_eq!(
        collision.finish().unwrap_err().code,
        "DERIVATIVE_INPUT_NAMESPACE_COLLISION"
    );

    let mut incomplete = base;
    incomplete.parent_semantic_digest.clear();
    assert_eq!(
        incomplete.finish().unwrap_err().code,
        "DERIVATIVE_INCOMPLETE_OBJECTIVE"
    );

    let missing_partition = DerivativeRequest {
        schema: DERIVATIVE_REQUEST_SCHEMA.into(),
        parent_semantic_digest: "semantic".into(),
        objective: Objective {
            name: "objective".into(),
            functional: ObservableFunctional {
                name: "response".into(),
                semantic_expression: "integrate(u)".into(),
                dimension: Dimension::LENGTH,
            },
            sense: ObjectiveSense::Measure,
        },
        design_variables: vec![
            DesignVariable {
                name: "width".into(),
                dimension: Dimension::LENGTH,
                parameter_owner: "cadabra3".into(),
                admissible_set: "width > 0".into(),
            },
            DesignVariable {
                name: "height".into(),
                dimension: Dimension::LENGTH,
                parameter_owner: "cadabra3".into(),
                admissible_set: "height > 0".into(),
            },
        ],
        controls: vec![],
        product: DerivativeProductSpec::Gradient,
        active_set: ActiveSet {
            active: vec!["width".into()],
            frozen: vec![],
        },
        evaluation_state: "converged".into(),
        convention: DerivativeConvention {
            dependence: DerivativeDependence::Total,
            level: DerivativeLevel::Discrete,
            scalar: ScalarConvention::Real,
            state: DerivativeStateConvention::ConvergedState,
            disposition: DifferentiabilityDisposition::Smooth,
            event_or_refusal_basis: None,
        },
        shape: None,
        identity: scientia::Digest::blake3(&[]),
    };
    assert_eq!(
        missing_partition.finish().unwrap_err().code,
        "DERIVATIVE_INPUT_PARTITION"
    );
}

#[test]
fn shape_requests_require_an_active_design_variable() {
    let request = DerivativeRequest {
        schema: DERIVATIVE_REQUEST_SCHEMA.into(),
        parent_semantic_digest: "semantic".into(),
        objective: Objective {
            name: "objective".into(),
            functional: ObservableFunctional {
                name: "response".into(),
                semantic_expression: "integrate(u)".into(),
                dimension: Dimension::LENGTH,
            },
            sense: ObjectiveSense::Measure,
        },
        design_variables: vec![DesignVariable {
            name: "width".into(),
            dimension: Dimension::LENGTH,
            parameter_owner: "cadabra3".into(),
            admissible_set: "width > 0".into(),
        }],
        controls: vec![scientia::Control {
            name: "load".into(),
            dimension: Dimension::LENGTH,
            support: "boundary".into(),
        }],
        product: DerivativeProductSpec::Gradient,
        active_set: ActiveSet {
            active: vec!["load".into()],
            frozen: vec!["width".into()],
        },
        evaluation_state: "converged".into(),
        convention: DerivativeConvention {
            dependence: DerivativeDependence::Total,
            level: DerivativeLevel::Discrete,
            scalar: ScalarConvention::Real,
            state: DerivativeStateConvention::ConvergedState,
            disposition: DifferentiabilityDisposition::Smooth,
            event_or_refusal_basis: None,
        },
        shape: Some(ShapeDerivativeConvention {
            geometry_revision: "cadabra-revision-7".into(),
            fixed_topology_stratum: "positive-width".into(),
            boundary_selection: vec!["outer".into()],
            include_normal_variation: true,
            include_measure_variation: true,
            disposition: DifferentiabilityDisposition::Smooth,
        }),
        identity: scientia::Digest::blake3(&[]),
    };
    assert_eq!(
        request.finish().unwrap_err().code,
        "SHAPE_DERIVATIVE_NO_ACTIVE_DESIGN"
    );
}
