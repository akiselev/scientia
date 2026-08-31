//! Regression test for the 1-D divergence shape arm: `div` of a concretely shaped one-vector
//! must contract to a scalar exactly as it does in higher dimensions. Before the fix,
//! `(Vector(_), divergence, spatial == 1)` kept the vector shape, so any 1-D model whose
//! coefficients carried declared (non-`Deferred`) shapes failed elaboration with
//! `TYPE_SHAPE_MISMATCH` — the `44-porous-electrode-battery.res` gap recorded in
//! GX-CONTRACTS C11.7/C11.8.

use quantitas::UnitRegistry;
use scientia::{compile_semantics, derive_variational_form, infer_form_requirements};

const ONE_DIMENSIONAL_DIFFUSION: &str = r#"
module tests.one_dimensional_divergence;

model OneDimensionalDiffusion {
    domain Line { dimension = 1; coordinates = cartesian; }

    field u: state scalar H1(order=1) on Line {
        time_role = differential;
    };

    provider transport_coefficient(u: Dimensionless) -> Diffusivity {
        shape = scalar;
        differentiability = symbolic;
    }

    property d = transport_coefficient(u);
    source f: VolumetricSource;

    equation balance on Line {
        dt(u) - div(d * grad(u)) = f;
    }

    boundary ends on boundary("ends") {
        dirichlet u = 0;
    }
}
"#;

#[test]
fn one_dimensional_divergence_of_a_declared_flux_elaborates_to_a_scalar_row() {
    let compilation = compile_semantics(ONE_DIMENSIONAL_DIFFUSION, &UnitRegistry::si_bootstrap())
        .expect("a concretely shaped 1-D div(d * grad(u)) elaborates");
    let form = derive_variational_form(&compilation.semantic, "OneDimensionalDiffusion", "balance")
        .expect("the 1-D balance equation derives a form");
    infer_form_requirements(&compilation.semantic, &form)
        .expect("the derived 1-D form infers requirements");
}
