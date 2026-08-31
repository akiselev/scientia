//! GX-facet: exterior-facet boundary integrals whose Neumann value is a bare provider call (no
//! wrapping symbol), e.g. `neumann u = prescribed_flux(t);` -- the shape `04-fick-diffusion` and
//! every other corpus model with a provider-backed flux boundary condition uses. Before this
//! change `factor_operator` refused such a boundary integrand with
//! `TENSOR_UNSUPPORTED: semantic expression N requires a later tensor primitive`, because the
//! call had no symbol of its own to bind a tensor input to (unlike a `property`/`source`-backed
//! value referenced by name). `derive_boundary_terms` now synthesizes a compiler symbol for a
//! bare top-level provider call and records it as a `FormCapture`, so it binds through the exact
//! same `ModelDefinedProperty` machinery a cell-measure property already uses: an External
//! input, never inlined.

use quantitas::UnitRegistry;
use scientia::{
    DenseTensor, InputSourceRequirement, SemanticMeasure, TensorInputRole, compile_semantics,
    derive_variational_form, factor_operator, infer_form_requirements, interpret_qfunction,
    lower_operator_kernels,
};

/// A repository-local model shaped exactly like `sinbad/physics/corpus/04-fick-diffusion.res`'s
/// `species_balance` equation and `walls` boundary condition (field name and role aside): a
/// `property`-backed diffusivity in the cell integral, and a bare provider call
/// (`prescribed_flux(t)`) as the sole Neumann boundary value, with no essential (Dirichlet)
/// condition at all.
const NEUMANN_PROVIDER_POISSON: &str = r#"
module gx_facet.provider_neumann;
model NeumannProviderPoisson {
    domain Omega { dimension = 2; coordinates = cartesian; }
    field u: unknown scalar H1(order=1) on Omega;
    provider diffusivity(u: Concentration) -> Diffusivity { differentiability = symbolic; }
    provider prescribed_flux(t: Time) -> SpeciesFlux { differentiability = analytic_provided; }
    property k = diffusivity(u);
    source f: VolumetricSource;
    equation balance on Omega { -div(k * grad(u)) = f; }
    boundary walls on boundary("walls") { neumann u = prescribed_flux(t); }
}
"#;

/// Same shape as `NEUMANN_PROVIDER_POISSON`, except the boundary value is a compound expression
/// that merely *contains* a provider call (`2 * prescribed_flux(t)`) rather than being a bare
/// top-level call. This is deliberately out of the bounded scope this change covers, and must
/// still refuse with the pre-existing `TENSOR_UNSUPPORTED` code, exactly as before this change.
const NEUMANN_COMPOUND_PROVIDER_POISSON: &str = r#"
module gx_facet.compound_provider_neumann;
model NeumannCompoundProviderPoisson {
    domain Omega { dimension = 2; coordinates = cartesian; }
    field u: unknown scalar H1(order=1) on Omega;
    provider diffusivity(u: Concentration) -> Diffusivity { differentiability = symbolic; }
    provider prescribed_flux(t: Time) -> SpeciesFlux { differentiability = analytic_provided; }
    property k = diffusivity(u);
    source f: VolumetricSource;
    equation balance on Omega { -div(k * grad(u)) = f; }
    boundary walls on boundary("walls") { neumann u = 2 * prescribed_flux(t); }
}
"#;

/// End-to-end: model -> form -> requirements -> factorization -> kernels succeeds, the
/// exterior-facet integral binds the boundary provider call as a single External
/// `ModelDefinedProperty` input (never inlined), and the tensor interpreter evaluates that
/// integral's primal QFunction against hand-built point data to the hand-computed value.
#[test]
fn facet_provider_call_becomes_an_external_input_and_evaluates_correctly() {
    let compilation =
        compile_semantics(NEUMANN_PROVIDER_POISSON, &UnitRegistry::si_bootstrap()).unwrap();
    let form = derive_variational_form(&compilation.semantic, "NeumannProviderPoisson", "balance")
        .unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let factorization = factor_operator(&form, &requirements).unwrap();

    let facet_integral = factorization
        .integrals
        .iter()
        .find(|integral| matches!(integral.measure, SemanticMeasure::ExteriorFacet { .. }))
        .expect("the Neumann boundary term derives an exterior-facet integral");

    let inputs = &facet_integral.primal.inputs;
    assert_eq!(
        inputs.len(),
        1,
        "the facet integrand is exactly `flux * v`; the test-function factor is a separate \
         BasisAdjoint stage, so the QFunction itself has one input: {inputs:?}"
    );
    let flux_input = &inputs[0];
    assert_eq!(flux_input.role, TensorInputRole::External);
    assert!(
        matches!(
            flux_input.source,
            InputSourceRequirement::ModelDefinedProperty { .. }
        ),
        "a boundary provider call must bind exactly like a cell-measure property call: \
         got {:?}",
        flux_input.source
    );
    assert!(flux_input.shape.is_empty(), "prescribed_flux is scalar");

    // FC5: the whole factorization -- including this facet integral -- lowers to Malleus
    // kernel bundles (primal/JVP/VJP/parameter), exactly as for a cell integral.
    let kernels = lower_operator_kernels(&factorization).unwrap();
    assert_eq!(kernels.bundles.len(), factorization.integrals.len());

    // Hand-computed check: the retained Neumann term for `-div(k * grad(u)) = f` is
    // `-flux * v` (the sign flip from moving the boundary term to the residual's other side),
    // so the primal QFunction with `flux` bound to a point value must return exactly `-flux`.
    for flux_value in [0.0, 1.0, -2.5, 3.75] {
        let outputs =
            interpret_qfunction(&facet_integral.primal, &[DenseTensor::scalar(flux_value)])
                .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].shape, Vec::<usize>::new());
        assert_eq!(
            outputs[0].data[0], -flux_value,
            "facet QFunction output for flux input {flux_value}"
        );
    }
}

/// Bounded-scope refusal: a provider call that is not the boundary value's own top-level
/// expression (here, nested inside a `2 * ...` multiplication) is not synthesized into a
/// capture, so it still reaches `factor_operator`'s tensor lowering as a raw, un-bound
/// `ProviderCall` node and refuses with the pre-existing typed error, unchanged by this package.
#[test]
fn a_provider_call_nested_inside_a_compound_boundary_expression_still_refuses() {
    let compilation = compile_semantics(
        NEUMANN_COMPOUND_PROVIDER_POISSON,
        &UnitRegistry::si_bootstrap(),
    )
    .unwrap();
    let form = derive_variational_form(
        &compilation.semantic,
        "NeumannCompoundProviderPoisson",
        "balance",
    )
    .unwrap();
    let requirements = infer_form_requirements(&compilation.semantic, &form).unwrap();
    let error = factor_operator(&form, &requirements).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("TENSOR_UNSUPPORTED") && message.contains("later tensor primitive"),
        "expected the pre-existing bounded-scope refusal, got: {message}"
    );
}

/// Deliberately reaches into the sinbad checkout, matching the pattern (and stated rationale) in
/// `tests/binding_slots.rs`'s `sinbad_corpus_dir`: an opt-in exception to the "no compile-time or
/// runtime path into Sinbad's product corpus" invariant, gated on `SINBAD_WORKSPACE` and skipped
/// otherwise. Sweeps every equation in the full 50-model corpus through
/// form -> requirements -> `factor_operator`, and reports how many now factor. The workspace
/// audit recorded a post-F5-corpus baseline of 41/128 operator factorizations; this package
/// targets exactly the shape that blocked further equations (a boundary condition whose value is
/// a bare provider call) and must not regress that count.
#[test]
fn corpus_sweep_reports_operator_factorization_counts() {
    let Some(dir) = sinbad_corpus_dir() else {
        return;
    };
    let mut total_equations = 0usize;
    let mut forms = 0usize;
    let mut requirements_ok = 0usize;
    let mut factored = 0usize;
    let mut factored_names = Vec::new();

    let mut files = std::fs::read_dir(&dir)
        .unwrap_or_else(|error| panic!("reading {}: {error}", dir.display()))
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "res"))
        .collect::<Vec<_>>();
    files.sort();
    assert_eq!(files.len(), 50, "expected the complete 50-model corpus");

    for path in files {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
        let compilation = compile_semantics(&source, &UnitRegistry::si_bootstrap()).unwrap_or_else(
            |diagnostics| panic!("{} failed to elaborate: {diagnostics:?}", path.display()),
        );
        for model in &compilation.semantic.models {
            for declaration in &model.declarations {
                if !matches!(
                    declaration.kind,
                    scientia::SemanticDeclarationKind::Equation { .. }
                ) {
                    continue;
                }
                total_equations += 1;
                let Ok(form) =
                    derive_variational_form(&compilation.semantic, &model.name, &declaration.name)
                else {
                    continue;
                };
                forms += 1;
                let Ok(requirements) = infer_form_requirements(&compilation.semantic, &form) else {
                    continue;
                };
                requirements_ok += 1;
                if factor_operator(&form, &requirements).is_ok() {
                    factored += 1;
                    factored_names.push(format!("{}:{}", model.name, declaration.name));
                }
            }
        }
    }

    eprintln!(
        "GX-facet corpus sweep: {factored}/{total_equations} operator factorizations \
         ({forms} forms, {requirements_ok} requirements) -- baseline was 41/128"
    );
    assert_eq!(
        total_equations, 128,
        "expected the complete 128-equation corpus"
    );
    assert!(
        factored >= 41,
        "GX-facet must not regress the post-F5-corpus baseline of 41/128 factorizations, got \
         {factored}/128: {factored_names:?}"
    );
    assert!(
        factored_names
            .iter()
            .any(|name| name == "FickDiffusion:species_balance"),
        "the corpus model this package targets must now factor: {factored_names:?}"
    );
}

/// The Sinbad corpus directory, only when the workspace coordinator opted in through
/// `SINBAD_WORKSPACE`; `None` skips the cross-repository sweep in hermetic runs. Mirrors
/// `tests/binding_slots.rs`'s helper of the same name exactly.
fn sinbad_corpus_dir() -> Option<std::path::PathBuf> {
    let Some(workspace) = std::env::var_os("SINBAD_WORKSPACE") else {
        eprintln!("skipping: SINBAD_WORKSPACE is not set; corpus sweep is opt-in");
        return None;
    };
    let dir = std::path::PathBuf::from(workspace).join("sinbad/physics/corpus");
    assert!(
        dir.is_dir(),
        "SINBAD_WORKSPACE is set but {} is not a directory",
        dir.display()
    );
    Some(dir)
}
