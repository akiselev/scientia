//! GX-A3: chain-rule property tangents in FC4. A repository-local nonlinear-heat-shaped fixture
//! with a declared `symbolic` provider verifies that the emitted JVP -- which now includes both
//! the ordinary directional term (`k * delta[grad(T)]`) and the chain-rule correction
//! (`grad(T) * d[k]/d[T] * delta[T]`) -- matches a finite difference of the primal QFunction
//! evaluated through the tensor interpreter, independent of the Sinbad corpus.

use quantitas::UnitRegistry;
use scientia::{
    DenseTensor, InputSourceRequirement, QFunctionProgram, TensorInputRole, compile_semantics,
    derive_variational_form, factor_operator, infer_form_requirements, interpret_qfunction,
};

const MODEL: &str = r#"
module gx_a3.chain_rule;

model Declared {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega {
    quantity = ThermodynamicTemperature;
    unit = K;
    time_role = differential;
  };

  provider thermal_conductivity(T: ThermodynamicTemperature) -> ThermalConductivity {
    shape = scalar;
    locality = pointwise;
    differentiability = symbolic;
    domain { T in [200 K, 2000 K]; }
  }

  property k = thermal_conductivity(T);

  source Q: VolumetricHeatSource;
  equation energy on Omega { -div(k * grad(T)) = Q; }

  boundary walls on boundary("walls") {
    dirichlet T = 300 K;
  }
}
"#;

/// A synthetic "state" for this fixture: `T`'s point value, its gradient, and the exact closed
/// form of the provider-backed property `k(T) = 0.5 + 0.001 * T` (known only to this test,
/// opaque to Scientia -- exactly the kind of runtime data contract C7's property kernel would
/// supply). Perturbing `t_value` along `direction` and re-deriving `grad_t`/`k_value`
/// consistently is what makes the finite difference a genuine check of `T`'s total effect on
/// the primal, through both its direct gradient use and its indirect use inside `k`.
struct State {
    t_value: f64,
    grad_t: [f64; 2],
    k_value: f64,
}

const DK_DT: f64 = 0.001;
fn k_of(t: f64) -> f64 {
    0.5 + 0.001 * t
}

fn state_at(t0: f64, grad_t0: [f64; 2], direction: [f64; 3], h: f64) -> State {
    let t_value = t0 + h * direction[0];
    State {
        t_value,
        grad_t: [grad_t0[0] + h * direction[1], grad_t0[1] + h * direction[2]],
        k_value: k_of(t_value),
    }
}

#[test]
fn jvp_inlines_the_symbolic_provider_and_matches_a_finite_difference() {
    let compilation = compile_semantics(MODEL, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "Declared", "energy").unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();
    let integral = &factorization.integrals[0];
    let primal = &integral.primal;
    let jvp = &integral.jvp;

    let receipt = jvp
        .derivative_receipt
        .as_ref()
        .expect("jvp has a derivative receipt");
    let model = &compilation.semantic.models[0];
    let k_symbol = model
        .symbols
        .iter()
        .find(|symbol| symbol.name == "k")
        .expect("property k has a symbol")
        .id;
    assert!(
        receipt.inlined_properties.contains(&k_symbol),
        "expected `k` in inlined_properties, got {:?}",
        receipt.inlined_properties
    );

    // Exactly one `PropertyTangent` input should have been synthesized for `k` w.r.t. `T`.
    let tangent_inputs: Vec<_> = jvp
        .inputs
        .iter()
        .filter(|input| matches!(input.role, TensorInputRole::PropertyTangent { .. }))
        .collect();
    assert_eq!(
        tangent_inputs.len(),
        1,
        "expected exactly one property-tangent input, got {tangent_inputs:?}"
    );

    let t0 = 320.0;
    let grad_t0 = [0.05, -0.02];
    // Perturb T's value and both gradient components together, matching the JVP's all-ones
    // direction below, so the finite difference exercises the *total* derivative through T.
    let direction = [1.0, 1.0, 1.0];

    let base = interpret_qfunction(
        primal,
        &bind_primal_inputs(primal, &state_at(t0, grad_t0, direction, 0.0)),
    )
    .unwrap();

    let h = 1e-4;
    let plus = interpret_qfunction(
        primal,
        &bind_primal_inputs(primal, &state_at(t0, grad_t0, direction, h)),
    )
    .unwrap();
    let minus = interpret_qfunction(
        primal,
        &bind_primal_inputs(primal, &state_at(t0, grad_t0, direction, -h)),
    )
    .unwrap();
    let finite_difference: Vec<f64> = plus
        .iter()
        .zip(minus.iter())
        .flat_map(|(p, m)| {
            p.data
                .iter()
                .zip(m.data.iter())
                .map(|(pv, mv)| (pv - mv) / (2.0 * h))
                .collect::<Vec<_>>()
        })
        .collect();

    // The JVP: primal inputs at the base state, every direction = 1 (matching the perturbation
    // above), and k's own tangent input = dk/dT.
    let jvp_out = interpret_qfunction(
        jvp,
        &bind_jvp_inputs(jvp, &state_at(t0, grad_t0, direction, 0.0), DK_DT),
    )
    .unwrap();
    let jvp_values: Vec<f64> = jvp_out
        .iter()
        .flat_map(|tensor| tensor.data.iter().copied())
        .collect();

    assert_eq!(finite_difference.len(), jvp_values.len());
    for (fd, jvp_value) in finite_difference.iter().zip(jvp_values.iter()) {
        assert!(
            (fd - jvp_value).abs() < 1e-3 * fd.abs().max(1.0),
            "finite difference {fd} vs analytic JVP {jvp_value} (base outputs: {base:?})"
        );
    }

    // A JVP with k's tangent forced to zero must NOT match the finite difference: the
    // regression check that the chain-rule term is actually load-bearing, not a coincidental
    // agreement (e.g. from `k` being frozen entirely, the pre-GX-A3 Picard behavior).
    let frozen_out = interpret_qfunction(
        jvp,
        &bind_jvp_inputs(jvp, &state_at(t0, grad_t0, direction, 0.0), 0.0),
    )
    .unwrap();
    let frozen_values: Vec<f64> = frozen_out
        .iter()
        .flat_map(|tensor| tensor.data.iter().copied())
        .collect();
    // A tight absolute tolerance here: `h`'s central-difference truncation error is O(h^2), far
    // below the chain-rule term's magnitude (`grad(T) * dk/dT`), so this only trips on a
    // genuine missing chain-rule contribution, not finite-difference noise.
    let mut any_differs = false;
    for (fd, frozen) in finite_difference.iter().zip(frozen_values.iter()) {
        if (fd - frozen).abs() > 1e-6 {
            any_differs = true;
        }
    }
    assert!(
        any_differs,
        "expected the chain-rule term to be load-bearing (zeroing dk/dT should NOT still match \
         the finite difference)"
    );
}

fn bind_primal_inputs(primal: &QFunctionProgram, state: &State) -> Vec<DenseTensor> {
    primal
        .inputs
        .iter()
        .map(|input| value_for(&input.source, &input.shape, state))
        .collect()
}

fn bind_jvp_inputs(jvp: &QFunctionProgram, state: &State, dk_dt: f64) -> Vec<DenseTensor> {
    jvp.inputs
        .iter()
        .map(|input| match input.role {
            TensorInputRole::Direction { .. } => direction_for(&input.shape),
            TensorInputRole::PropertyTangent { .. } => DenseTensor::scalar(dk_dt),
            _ => value_for(&input.source, &input.shape, state),
        })
        .collect()
}

fn value_for(source: &InputSourceRequirement, shape: &[usize], state: &State) -> DenseTensor {
    match source {
        InputSourceRequirement::ModelDefinedProperty { .. } => DenseTensor::scalar(state.k_value),
        InputSourceRequirement::Basis if shape.is_empty() => DenseTensor::scalar(state.t_value),
        InputSourceRequirement::Basis => {
            DenseTensor::new(shape.to_vec(), state.grad_t.to_vec()).unwrap()
        }
        InputSourceRequirement::ExternalValue => DenseTensor::scalar(1.5),
        other => panic!("fixture has no value binding for source {other:?}"),
    }
}

fn direction_for(shape: &[usize]) -> DenseTensor {
    if shape.is_empty() {
        DenseTensor::scalar(1.0)
    } else {
        let count: usize = shape.iter().product();
        DenseTensor::new(shape.to_vec(), vec![1.0; count]).unwrap()
    }
}
