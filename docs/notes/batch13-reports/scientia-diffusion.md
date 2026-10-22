# Lane `scientia-diffusion` (SV2-C1c) — handoff report (batch paused)

The coordinator paused batch 13 mid-lane (2026-10-22). This is a handoff, not a completion
report: packages 1–4 are coded but **not compiled against all targets and not tested**;
package 5 (the new test file) is not written. Nothing below is "verified" unless marked so.

## 1. Branch and final sha

`akiselev/scientia`, branch `claude/b13-scientia-diffusion`, base `84188ac`, head `f5e5919`
(this report is committed on top of it). Siblings cloned at the pinned shas: malleus
`7a43e7f`, quantitas `0d682ac`, resolvent `b604f93`. `Cargo.lock` untouched.

## 2. What landed (code exists; status per item)

- **Package 1, diffusive flux semantics** — `src/conservation_diffusion.rs`
  `classify_divergence`, called from `compile_conservation_law_system` (`src/conservation.rs`).
  Each top-level additive `div(...)` term is `Diffusive` when its argument is
  `k * grad(u)`, `grad(u) * k`, `grad(u)` or a negation/single defining symbol of those,
  else `Hyperbolic`. At most one of each; the FC10 `method.flux_divergence`/`method.flux` are
  re-pointed to the hyperbolic one when present; with none, `method` keeps the diffusive
  divergence and `ConservationLawSystem::hyperbolic_flux()` is `None`. New field
  `ConservationLawSystem.diffusion: Option<DiffusiveTerm>` (serde-omitted when absent).
  Refusals: `FV_DIFFUSION_SIGN` (`+div(k grad u)`), `FV_DIFFUSION_ANISOTROPIC` (non-scalar
  `k`), `FV_DIFFUSIVE_FLUX_UNSUPPORTED` (two diffusive terms; `k` holding a differential).
  The W12 `Diffusive1D` fixture (`div(a*u - grad(u))`) is still classified hyperbolic and
  still refuses at lowering with `FV_DIFFUSIVE_FLUX_UNSUPPORTED` (by design, W12 test kept).
  Status: compiles as library (`cargo check -p scientia` passed before the last test-file
  edits); semantics unverified.
- **Package 2, diffusive face kernel** — `lower_diffusion` (same module), called from
  `lower_conservation_law_checked`. Produces `DiffusionProgram { averaging, uniform,
  state_dependent, coefficient, face_flux, checks }` on `ConservationLawProgram.diffusion`
  (omitted when absent). Pure diffusion lowers a constant-zero `physical_flux_normal`
  (speed 0, face flux 0; authored flux refused `FV_FLUX_UNSUPPORTED`). `check_diffusive_claims`
  checks consistency / antisymmetry / dissipativity at `FvDiffusiveSample`s.
  Status: compiles; **never executed**.
- **Package 3, multidimensional limiters** — `multidimensional_limiter_kernel`,
  `validate_limiter` (`FV_LIMITER_PARAMETER_INVALID` for Venkatakrishnan `k ≤ 0`); the
  `FV_LIMITER_UNSUPPORTED` arm in `validate_intent` now accepts both families. TVD obligation
  restricted to Sweby limiters. Status: compiles; never executed.
- **Package 4, obligations** — `VerificationObligationKind::DiffusiveFluxDissipativity
  { equation }` and `GridPeclet { field }` (`src/verification.rs`), emitted by
  `conservation_diffusion::obligations` after the W12 list; the `CflBound` applicability gains
  a statement for the diffusive/pure-diffusion case. Status: compiles; never executed.
- **Package 5** — fixture models appended to `tests/fixtures/scalar-conservation.res`
  (`AdvectionDiffusion1D`, `NonlinearDiffusion1D`, `NonsmoothDiffusion1D`,
  `TwoRegionConduction2D`, `AntiDiffusion1D`, `DoubleDiffusion1D`); verified to elaborate with
  the CLI (`scientia check`) for the first, second and fourth. **`tests/sv2c1c_diffusion.rs`
  does not exist.**

## 3. Tests run

None of the lane's tests exist or ran. Baseline before edits: `cargo check --locked -p
scientia --all-targets` passed at `84188ac`. After the edits the library and all test targets
except the last-fixed `w11_symbolic_identity_pins` were checked; the final one-line fix to
that file was committed without re-running the check (coordinator instruction). First thing
to run next: `cargo check --locked -p scientia --all-targets`.

## 4. Contract shapes landed (for GX-CONTRACTS C12; all provisional until tests pass)

- Schemas unchanged (`scientia-conservation-law/1`, `…-program/1`, `…-fv-method-intent/1`,
  `…-fv-kernel/1`): the extension is **additive** — every new field is serde-omitted when
  absent/default, so W12 system, program, intent and kernel digests are byte-identical. If the
  `scientia-systems` lane bumps to `/2`, fold these optional fields into `/2`.
- Intent spelling: `diffusion = { averaging = "harmonic" | "arithmetic" }` on
  `FvMethodIntent.diffusion: DiffusionIntent` (default harmonic, omitted in JSON). **Sinbad's
  `[discretization]` parser and every literal `FvMethodIntent { .. }` must add the field**
  (see §6 for the Scientia test files edited for this).
- Operand meanings (`FvOperandMeaning`): `FaceNormalGradient` (odd; `(u⁺−u⁻)/(δ·n)`, ghost
  state and centre-to-face distance at a `dirichlet` face), `CellDatum { datum, side }`
  (per-cell data, `Cell` role with the opposite-side partner; boundary plus side = owner's
  value), `LimiterBound` (Δ⁺), `LimiterIncrement` (Δ⁻), `CellSize` (h).
- Kernel roles (`FvKernelRole`): `DiffusionCoefficient` (point `k(u)`), `DiffusiveFaceFlux`
  (facet pair `-k_f·g`, kernel name `diffusive_face_flux_{harmonic|arithmetic}`); the
  multidimensional limiters keep role `Limiter` with names `limiter_barth_jespersen`,
  `limiter_venkatakrishnan`, operands `delta_plus`, `delta_minus`, (`h`).
- Finitum binding rules intended: run `DiffusiveFaceFlux` on every interior and `dirichlet`
  face and add it to the hyperbolic face flux; a `neumann` (`PrescribedBoundaryFlux`) face
  prescribes the **total** flux and skips the diffusive kernel; Δ⁺ chosen by the sign of Δ⁻
  (`u_max−u_i` / `u_min−u_i`), cell limiter = min over faces; explicit step bound
  `λ_f → λ_f + k_f/δ_f`.
- Refusal codes: `FV_DIFFUSION_SIGN`, `FV_DIFFUSION_ANISOTROPIC`, `FV_DIFFUSION_ABSENT`,
  `FV_DIFFUSIVE_FLUX_INCONSISTENT`, `FV_DIFFUSIVE_FLUX_NOT_ANTISYMMETRIC`,
  `FV_DIFFUSIVE_FLUX_NOT_DISSIPATIVE`, `FV_LIMITER_PARAMETER_INVALID`.
- Obligation kinds: `DiffusiveFluxDissipativity { equation }`, `GridPeclet { field }`
  (Sinbad's exhaustive matches on `VerificationObligationKind` gain two variants).
- Public API: `lower_conservation_law_checked(module, system, intent, samples,
  diffusive_samples)`, `check_diffusive_claims(&program, &samples)`, `FvDiffusiveSample
  { minus, plus, distance, data, minus_data, plus_data }`.

## 5. Expected moves

None intended. No pinned identity should move (additive serde-omitted fields). Unverified:
`tests/sv2c1_identity_pins.rs`, `tests/w11_symbolic_identity_pins.rs` and Sinbad
`sv2c7_identity_pins` have not been run against this head.

## 6. Cross-lane edits

- `src/verification.rs`: two added `VerificationObligationKind` variants.
- `src/symbolic_import.rs`: the operand-meaning match gains `continue` arms for the new
  variants (no behaviour change).
- `tests/sv2c1_conservation.rs`: `intent()` helper gets `diffusion: Default::default()`; the
  `eval` match gets a `_ => 0.0` arm. `tests/w11_symbolic_import.rs`,
  `tests/w11_symbolic_identity_pins.rs`: `diffusion: Default::default()` in the one intent
  literal each. All needed to compile after the intent field the lane file assigns to me.
- `src/conservation.rs`: beyond the assigned hunks, `KernelBuilder`, `Lower`, `DataTable`,
  `StateContext` and the `add/half/select/compare/fv_error` helpers became `pub(crate)`;
  `Lower` gained `cell_data: bool` (provider product in a sided facet context → `CellDatum`,
  set only for the diffusion coefficient so W12 authored fluxes are unchanged).

## 7. Contradictions and blockers

- `.res` has no volume sub-regions (`region` is `boundary of <domain>` only), so "a property
  bound per region" is realized by the provider's product sampled per cell by the consumer
  (`TwoRegionConduction2D` uses `provider conductivity(material: selector)`); the region →
  material map is case data. Noted, not a blocker.
- `FvMethodIntent` gaining a field breaks every literal construction in sibling repositories
  (Sinbad, Finitum tests) — unavoidable given the lane's intent spelling; recorded in
  `docs/re-requests/batch13-scientia-diffusion.md`.
- Barth–Jespersen is not TVD (local maximum principle only); the W12 code claimed TVD for any
  limiter ≠ `None` with a monotone flux. I restricted the claim to Sweby families rather than
  claim what cannot be justified; no obligation kind exists yet for the maximum principle.

## 8. Carry-overs (next session, in order)

1. `cargo check --locked -p scientia --all-targets`; fix whatever the last edits broke.
2. Write `tests/sv2c1c_diffusion.rs` per the lane file: bitwise hand formula (constant `k`
   → `-(k*g)`; harmonic `((2*km)*kp)/(km+kp)` and arithmetic `0.5*(km+kp)` with `k⁻≠k⁺` via
   `TwoRegionConduction2D` `minus_data`/`plus_data`), dissipativity at random samples,
   exact antisymmetry, `k(u)` JVP vs finite differences (1e-8 rel) and `NonsmoothDiffusion1D`
   refusal `FV_NONSMOOTH_DERIVATIVE_REFUSED`, BJ/Venkatakrishnan digests + hand values
   (φ=1 for Δ⁺=2Δ⁻; BJ clips: Δ⁺=0.5, Δ⁻=2 → 0.25), `AntiDiffusion1D` → `FV_DIFFUSION_SIGN`,
   `DoubleDiffusion1D` → `FV_DIFFUSIVE_FLUX_UNSUPPORTED`, intent JSON with/without
   `diffusion`, and that every W12 test and pin passes.
3. Run `cargo test --locked -p scientia --test sv2c1c_diffusion --test sv2c1_conservation
   --test sv2c1_identity_pins --test w11_symbolic_identity_pins --test w11_symbolic_import
   --lib`, then clippy `-D warnings` and `cargo fmt -p scientia -- --check` (new module and
   patched hunks are **not yet formatted**).
4. STATUS.md section for the lane (not written; file is at 286/300 lines — condense the W11
   block first).
5. Open design points: explicit diffusive time-step bound has no obligation kind of its own
   (folded into the `CflBound` statement); a maximum-principle obligation for BJ;
   non-orthogonal correction, tensor `k`, cross-diffusion, characteristic-variable limiting
   remain out of scope (typed refusal only for tensor `k`).
