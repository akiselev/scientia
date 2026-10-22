# Lane `scientia-systems` (repository: scientia, branch `claude/b13-scientia-systems`, base `84188ac`)

Work package: **SV2-C1b** (systems of conservation laws), the Scientia half of SV9 F1 stage 1
(canonical compressible-flow building blocks). The Finitum realization and the Sinbad Euler
acceptance cases follow in later phases of this batch and will build against your branch, so your
public API must be stable and documented in the report.

## Goal

Extend the W12 scalar conservation-law semantics (`src/conservation.rs`: `ConservationLawSystem`,
`FvMethodIntent`, `lower_conservation_law`, `FvKernel`, the D6 `check_smooth_request`) to **systems**
`dt(U) + div(F(U)) = S(U)` with `m` conserved components, so that the corpus model
`sinbad/physics/corpus/28-compressible-euler.res` (ideal-gas Euler: `density` scalar, `momentum`
`vector(d)`, `total_energy` scalar, all `DG(order=0)`, three `equation`s, `pressure` constitutive)
reads, lowers and checks with **no Euler-specific Rust**. The law is whatever the `.res` says; the
generic flux families need only `F·n` and declared wave-speed information (D3).

## Packages

1. **Reading a system.** `compile_conservation_law_system` (or a new sibling entry point, keep
   the scalar one byte-identical) takes a *set* of equations/conserved fields of one model and
   produces one system artifact: component layout (field, component index, offset into the state
   vector), per-equation flux expressions (a scalar field's flux is a `vector(d)`; a `vector(d)`
   field's flux is a `tensor(d,d)`, row = component), per-equation signed source terms, boundary
   semantics per field (`dirichlet`/`neumann` ghost state or prescribed flux, per component), and
   the closure check. Version the schema (`scientia-conservation-law/2` or a new
   `scientia-conservation-system/1`); the `/1` scalar artifact and the pins in
   `tests/sv2c1_identity_pins.rs` and `tests/w11_symbolic_identity_pins.rs` must stay unchanged.
2. **Declared wave speeds.** Add `SpeedSource` variants that the `.res` declares rather than
   derives: a maximum signal speed expression (for LLF/Rusanov: e.g. `abs(dot(velocity, n)) +
   sqrt(gamma * pressure / density)`) and signed min/max speeds (for HLL). Decide the `.res`
   spelling (a `constitutive` or a dedicated declaration referencing the face normal `n`) and
   record it; the normal is an operand (`FvOperandMeaning::Normal{axis}`), never a global. For a
   scalar law the derived tangent stays the default. Refuse (typed) a generic flux on a system
   with no declared speed.
3. **Face kernels over state vectors.** `Central`, `LocalLaxFriedrichs`, `Hll` as facet-pair kernels
   whose operands are `State{side, component}` (extend `FvOperandMeaning` versionfully; the scalar
   operand shapes keep their meaning), outputs one flux per component; batched layout identical to
   the scalar case so Finitum's binding generalizes. Authored fluxes for systems (D4): a
   `constitutive` vector-valued two-point expression (this is how HLLC/Roe would be written in a
   `.res` catalog module); the existing consistency and antisymmetry claims checked per component
   at deterministic samples with a predeclared tolerance.
4. **Primitive reconstruction map and admissibility.** `ReconstructionVariables::Primitive` needs a
   declared map `U -> W` and its inverse (`.res` constitutive declarations, e.g. `primitive =
   [density, velocity, pressure]`): lower both directions as cell-local kernels so Finitum can
   reconstruct in `W` and convert back. Admissibility (`PositivityIntent` or a new
   `AdmissibilityIntent`): declared strictly-positive expressions (e.g. `density`, `pressure`)
   lowered as a cell-local kernel returning the minimum margin, for Methodus's `admissibility` hook.
5. **Obligations and D6 for systems.** `NumericalFluxConsistency` / `NumericalFluxAntisymmetry` per
   component; `CflBound` from the declared max speed; D6: derivative requests through `abs`/`min`/
   `max` in the speed or limiter kernels refuse with the existing `FV_NONSMOOTH_DERIVATIVE_REFUSED`
   semantics (LLF/HLL on Euler refused; `Central` on a smooth system keeps its JVP, checked
   numerically by finite differences at a sample).
6. **Fixture and tests.** `tests/fixtures/euler.res` (1-D and 2-D ideal-gas Euler with `parameter
   gamma`, periodic and Dirichlet/ghost variants, a declared max speed and HLL speeds, primitive map,
   admissibility) and `tests/sv2c1b_systems.rs`:
   - LLF on a 1-D Euler face equals a hand-written Rusanov formula bitwise at Sod left/right states
     (hand values computed in the test from `gamma = 1.4`);
   - HLL equals its hand formula; face-swap symmetry exact for Central/LLF/HLL on 2-D Euler;
   - the batched run equals the per-face run bitwise;
   - the declared speed at the Sod states equals `|u| + c` to 1 ulp;
   - primitive round trip `W(U(W)) = W` exact on rational data; admissibility margin negative for
     a negative-pressure state;
   - D6 refusals and the Central JVP check;
   - all existing tests and pins unchanged.

## Files you own

`src/conservation.rs` (the systems extension; split a new `src/conservation_systems.rs` if it keeps
the scalar code readable), `tests/sv2c1b_systems.rs`, `tests/fixtures/euler.res`, your STATUS
section. The language/semantic layers (`src/semantic.rs`, `src/scientific.rs`, `src/formulation.rs`)
only if a declaration form you add needs them; minimal, additive, listed in the report.

## Files you do not own

Lane `scientia-diffusion` edits `src/conservation.rs` too (diffusive flux lowering and the
Barth-Jespersen/Venkatakrishnan limiter intents) and adds `src/conservation_diffusion.rs`. Keep your
hunks away from the limiter-intent and diffusion code paths; do not touch `FV_LIMITER_UNSUPPORTED`.
The coordinator merges diffusion first, then your lane.

## Done means

Packages 1–6 landed with the tests above green, scalar identities pinned unchanged, the API
surface (types, functions, schema ids, refusal codes, the `.res` spelling of speeds/primitive
map/admissibility) written in the report §4 so the Finitum lane can bind it, and a worked
example of the Euler program's operand layout in the report.

## Refuse rather than guess

Entropy fixes, characteristic-variable limiting, non-ideal equations of state, exact Riemann
solvers, positivity-preserving limiters beyond the declared admissibility hook, and
three-dimensional meshes (the fixture may declare 3-D; Finitum has 1-D/2-D topologies) are out of
scope: typed refusals or recorded carry-overs, never approximations.
