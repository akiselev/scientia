# Lane `scientia-diffusion` (repository: scientia, branch `claude/b13-scientia-diffusion`, base `84188ac`)

Work package: **SV2-C1c**: finite-volume lowering of diffusive terms and of the multidimensional
limiter intents. These are the two Scientia items the W12 capsule left open
("Scientia does not lower `div(-k grad u)` for finite volume: gate 4's κ is case data";
"simplex LSQ + Venkatakrishnan/Barth-Jespersen refused by Scientia lowering,
`FV_LIMITER_UNSUPPORTED`"). They are prerequisites for SV9 E5 stage 1 (conjugate heat transfer on
one FV mesh, D5), where conductivities are `.res` properties per region, not case data.

## Packages

1. **Diffusive flux semantics.** A `.res` balance `dt(u) + div(F(u)) - div(k * grad(u)) = S`
   (either term optional) is read as hyperbolic part `F` plus diffusive part with coefficient
   `k` (a `parameter`, a `property`, or a `constitutive` of `u`, i.e. `k(u)` allowed). Extend the
   conservation-law artifact versionfully (coordinate with the version the `scientia-systems`
   lane introduces: if both bump, the consolidation reconciles; name yours clearly) so the W12
   scalar identities in `tests/sv2c1_identity_pins.rs` stay unchanged.
2. **Diffusive face kernel.** Lower a facet-pair kernel for the diffusive normal flux
   `-k_f * (grad u)·n` with a new operand meaning for the face-normal gradient that the realizer
   supplies (`FaceNormalGradient`, the two-point `(u⁺ - u⁻)/(δ·n)` on orthogonal meshes) and
   `k` evaluated on both sides with a declared face averaging in the intent
   (`diffusion = { averaging = "harmonic" | "arithmetic" }`; harmonic is the default for
   piecewise-constant conductivity). Claims: consistency (the kernel equals `-k (grad u)·n` at
   samples, exact for constant `k`), antisymmetry, and dissipativity (`flux · (u⁺ - u⁻) ≤ 0`
   when `k > 0`). D6 applies: `k(u)` smooth keeps its JVP; an `abs`/`min`/`max` in `k` refuses.
3. **Multidimensional limiter intents.** Lower `LimiterFamily::BarthJespersen` and
   `Venkatakrishnan { k }` instead of refusing: the cell-wise limiter function
   `φ(Δ⁺, Δ⁻, h)` (Barth–Jespersen: `min(1, Δ⁺/Δ⁻)` form with the `min`; Venkatakrishnan: the
   smooth rational form with `ε² = (k h)³`) as a kernel Finitum applies per cell/face. D6: BJ
   refuses derivatives (`min`), Venkatakrishnan admits them (smooth).
4. **Obligations.** `DiscreteFluxBalance` already covers diffusion once realized; add
   `DiffusiveFluxDissipativity` (or fold into the flux claims) and a `GridPeclet` informational
   obligation (`|a| h / k`) if cheap; otherwise record as open.
5. **Fixture and tests.** Extend `tests/fixtures/scalar-conservation.res` with
   `AdvectionDiffusion1D` (`parameter a`, `parameter k`), `NonlinearDiffusion1D` (`k = k0 * (1 + u)`),
   and a two-region conduction model whose conductivity is a `property` bound per region (for E5
   stage 1), and add `tests/sv2c1c_diffusion.rs`:
   - the lowered diffusive kernel equals the hand formula bitwise (constant `k`, harmonic and
     arithmetic averaging at `k⁻ ≠ k⁺`);
   - dissipativity at random samples; antisymmetry exact;
   - `k(u)` JVP equals a finite-difference tangent to 1e-8 relative; the `abs` variant refuses;
   - BJ and Venkatakrishnan programs lower with stable digests, with the limiter kernel checked
     against hand values (`φ = 1` on smooth monotone data; BJ clips exactly to the bound);
   - every existing test and pin unchanged.

## Files you own

New `src/conservation_diffusion.rs` (put the diffusive lowering and limiter kernels there), the
minimal dispatch hunks in `src/conservation.rs` (intent fields, the limiter arm that today refuses
`FV_LIMITER_UNSUPPORTED`, the system artifact's diffusion field), `tests/sv2c1c_diffusion.rs`,
`tests/fixtures/scalar-conservation.res` (append models only), your STATUS section.

## Files you do not own

Lane `scientia-systems` extends `src/conservation.rs` for vector-valued state (component operands,
declared speeds, primitive map, admissibility). Keep your hunks local to the diffusion and limiter
code; do not restructure shared types. The coordinator merges your lane first.

## Done means

Packages 1–5 landed, tests green, identities unchanged, and report §4 giving the operand meaning,
intent spelling and kernel role names Finitum must bind (the Finitum realization of the diffusive
kernel is the next phase of this batch).

## Refuse rather than guess

Anisotropic/tensor conductivity, non-orthogonal correction terms, cross-diffusion between
components, and limiters acting on characteristic variables are out of scope: typed refusals or
carry-overs.
