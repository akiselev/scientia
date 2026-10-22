# Lane `scientia-systems` (SV2-C1b) — handoff report (batch paused by the coordinator)

Status: **half-done.** The coordinator paused batch 13 before the systems module was written.
This report is a handoff, not a completion report. Nothing below is consumer-accepted.

## 1. Branch and final sha

- Repository `akiselev/scientia`, branch `claude/b13-scientia-systems`, base `84188ac`.
- Final sha: `f411354` (plus the commit carrying this report). One code commit:
  "SV2-C1b: prepare the scalar FV lowering for systems (half-done, compiles)".
- Siblings used: malleus `7a43e7f`, quantitas `0d682ac`, resolvent `b604f93` (as pinned).
  `sinbad` was attached read-only (HEAD `fb11d4a`; the pinned `5341c48` is not fetchable as a
  shallow ref) only to read `physics/corpus/28-compressible-euler.res`.

## 2. What landed (groundwork only; `cargo check --all-targets` clean, one dead-code warning)

All in `src/conservation.rs` unless noted; every change is additive to the scalar path.

- `FvOperandMeaning` gains `StateComponent { side, component }`, `PrimitiveComponent
  { component }`, `OutputComponent { component }` (serde kinds `state_component`,
  `primitive_component`, `output_component`). Scalar variants unchanged.
- `FvKernelRole` gains `SignalSpeedMinimum`, `SignalSpeedMaximum`, `PrimitiveForward`,
  `PrimitiveInverse`, `Admissibility`.
- `SpeedSource` gains `DeclaredMaximum`, `DeclaredSigned`.
- `KernelBuilder::finish_outputs(role, Vec<(name, meaning, value)>, facet_roles)`: several
  write operands appended after the inputs; `finish` delegates to it (scalar bytes unchanged).
- `StateContext` gains `System { state: Vec<OperandId>, normal: Option<Vec<OperandId>> }` and
  `Primitive { variables: Vec<(SymbolId, OperandId)> }`; `Facet` now holds `Vec<OperandId>` per
  side (length 1 for the scalar law). Not yet constructed anywhere (the warning).
- `Lower` holds `components: Vec<(SymbolId, offset, count)>` instead of one `conserved`
  symbol; `symbol()` resolves any conserved field to its slots, refuses conserved reads inside
  the inverse primitive map (`FV_PRIMITIVE_MAP_INVALID`), and binds primitive variables.
- Expression lowering: vector literals of equal-extent vectors flatten row-major (tensor
  literal); `Index` with literal indices picks a row-major sub-block (`FV_FLUX_SHAPE` when the
  semantic axes do not match the components); `normal_component` works in any context that
  carries a normal and gives one `row·n` per row for a tensor operand.
- `normalize_law(model, equation, &method) -> (flux_sign, source_terms)` and
  `read_boundaries(model, conserved)` extracted from `compile_conservation_law_system`
  (behaviour identical; scalar tests and pins pass).
- `src/semantic.rs` (additive, 11 lines): a literal `[[..], [..]]` of equal-extent vectors is
  typed `Tensor { rows, cols }` instead of being mistyped as a vector. No `.res` in this
  repository or in `sinbad/physics` uses nested literals today.
- Visibility: `fv_error`, `empty_digest`, `additive_terms`, `is_zero_literal`,
  `contains_differential`, `limiter_kernel`, `authored_declaration`, `validate_intent`,
  `DataTable`, `KernelBuilder` (+ `meanings`), `Lower`, `StateContext`, `add`, `half`, `dot`,
  `compare`, `select` are `pub(crate)` for the planned `src/conservation_systems.rs`.

Not written: `src/conservation_systems.rs`, `tests/fixtures/euler.res`,
`tests/sv2c1b_systems.rs`, the STATUS section, lib re-exports of new items.

## 3. Tests run

`CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 cargo test --locked -p scientia --test
sv2c1_conservation --test sv2c1_identity_pins --test w11_symbolic_identity_pins --test
w11_symbolic_import --test fc10_methods`: 10 + 1 + 1 + 8 + 4 passed, 0 failed, 0 ignored.
The two pin targets prove the scalar `/1` artifacts and the LLF program bytes are unchanged.
Not run: clippy, fmt, rustdoc, the rest of the suite (paused before the push gate).

## 4. Contract shapes landed

Only the enum extensions in §2 (`FvOperandMeaning`, `FvKernelRole`, `SpeedSource`). No new
schema id exists yet. Planned (decided, not implemented; see §8):
`scientia-conservation-system/1`, `scientia-conservation-system-program/1`, refusal codes
`FV_SPEED_UNDECLARED`, `FV_SPEED_DECLARATION_INVALID`, `FV_PRIMITIVE_MAP_INVALID` (exists),
`FV_PRIMITIVE_MAP_SHAPE`, `FV_ADMISSIBILITY_INVALID`.

## 5. Expected moves

None. `sv2c1_identity_pins` and `w11_symbolic_identity_pins` pass unchanged.

## 6. Cross-lane edits

- `tests/sv2c1_conservation.rs`: two lines, a `_ => unreachable!` arm in the test's `eval`
  match (new `FvOperandMeaning` variants made the match non-exhaustive).
- `src/symbolic_import.rs`: one match arm extended with the three new variants (`continue`).
- `src/semantic.rs`: the tensor-literal typing above (allowed by the lane for declaration
  forms; listed here because it is outside `src/conservation.rs`).
- Consumers at pinned shas that match exhaustively on `FvOperandMeaning` (Finitum) will need
  the same arm; this is the "versionful extension" the lane asked for.

## 7. Contradictions and blockers

1. **The corpus Euler model cannot lower as written.** `sinbad/physics/corpus/28-compressible-euler.res`
   declares `outer`, `identity` and `heat_capacity_ratio` as *providers* (`provider outer(...)
   -> Pressure { shape = tensor(3,3); ... }`, `property gamma = heat_capacity_ratio(density,
   total_energy)`). Scientia cannot know a provider is the outer product, and the FV lowering
   refuses state-dependent provider calls (`FV_FLUX_PROVIDER_UNSUPPORTED`) by design. Making
   `outer`/`identity` language intrinsics would make the corpus fail to compile
   (`provider ... cannot reuse the reserved intrinsic function name`), breaking Sinbad's
   `tests/fc10_methods.rs`, which compiles this very file through the FC10 affine path. So the
   lane goal "the corpus model reads, lowers and checks" is reachable for *reading* only;
   lowering needs a corpus change (request filed in `docs/re-requests/batch13-scientia-systems.md`).
2. `sinbad` pinned sha `5341c48` is not fetchable as a shallow ref; HEAD `fb11d4a` was read.
   The corpus file content is what matters and is quoted in the request.
3. Unknown model annotations (`@wave_speeds`, `@primitive_map`, `@admissibility`, see §8)
   become `Unsupported` verification obligations (`VERIFY_UNSUPPORTED_ANNOTATION`) in
   `src/verification.rs`; harmless but noisy. A one-arm skip there is not in this lane's files.

## 8. Carry-overs (how to continue; decisions already taken)

Next session, in order:

1. Write `src/conservation_systems.rs` (`pub mod conservation_systems;` in `lib.rs`, re-export
   its public items) using the `pub(crate)` helpers:
   - `compile_conservation_system(module, model, members: &[(equation, conserved)])` →
     `ConservationSystem` (`scientia-conservation-system/1`): `layout: Vec<ConservedComponent
     { field, field_name, component: u8, offset }>`, `equations: Vec<SystemEquation { method,
     field_name, offset, components, flux_sign, source_terms, boundaries }>` (per equation via
     `select_conservation_law`, `normalize_law`, `read_boundaries`; one domain; scalar or
     `vector(k)` DG fields; duplicates refused `FV_SYSTEM_UNSUPPORTED`), `speeds:
     DeclaredSpeeds { absolute_maximum, minimum, maximum: Option<SpeedDeclaration { declaration,
     name, expression }> }`, `primitive: Option<PrimitiveMap { forward, inverse, variables }>`,
     `admissibility: Option<..{ positive: Vec<(name, ExprId)> }>`, `artifact_digest`.
     `check_boundary_closure(exterior, periodic)`: every equation must close every exterior
     non-periodic region.
   - `.res` spelling (model-level annotations, no parser change needed; arguments are symbols
     of `constitutive` declarations):
     `@wave_speeds(absolute_maximum = max_signal_speed, minimum = min_signal_speed, maximum =
     max_signed_speed);` with e.g. `constitutive max_signal_speed = abs(normal_component(velocity))
     + sqrt(gamma * pressure / density);` (1-D: `normal_component([velocity])`, a one-element
     literal, since `normal_component` needs a vector); the normal is the `Normal{axis}` operand.
     `@primitive_map(forward = primitive, inverse = conserved_from_primitive, variables =
     [w_density, w_velocity_x, w_velocity_y, w_pressure]);` where `variables` are valueless
     `parameter`s and `inverse` is a vector constitutive in them.
     `@admissibility(positive = [density, pressure]);`
     Momentum flux in 2-D as a tensor literal: `[[momentum[0]*velocity[0] + pressure,
     momentum[0]*velocity[1]], [momentum[1]*velocity[0], momentum[1]*velocity[1] + pressure]]`.
   - `lower_conservation_system(module, system, intent, samples: &[FvSystemSample { minus: Vec,
     plus: Vec, normal, data }])` → `ConservationSystemProgram`
     (`scientia-conservation-system-program/1`): operand layout per kernel is
     `StateComponent{Minus,0..m}`, `StateComponent{Plus,0..m}`, `Normal{0..d}`, data, then
     `OutputComponent{0..m}` (point kernels: `StateComponent{None,..}`); single-output kernels
     (speeds, admissibility) keep `Output`. Kernels: `physical_flux_normal` (m outputs, row
     `offset+r` = `flux_sign·Σ_j F[r·d+j]·n_j`), `characteristic_speed` (declared absolute
     maximum, role `CharacteristicSpeed`), `speed_minimum`/`speed_maximum`
     (`SignalSpeedMinimum/Maximum`), `face_flux` (Central ½(F⁻+F⁺); LLF with
     `λ = bind(max(λ⁻, λ⁺))` and the scalar tree `Sub(half(add(fm,fp)), Mul(half(λ), Sub(up,um)))`
     per component; HLL with `smin = bind(min(s⁻_min, s⁺_min))`, `smax = bind(max(..))` and the
     scalar select tree per component; authored: m-component constitutive in `Facet` context,
     `FV_AUTHORED_FLUX_SHAPE` otherwise), boundaries per equation (ghost / prescribed flux with
     k outputs at slots `offset..offset+k`), `source` (m outputs, 0 for equations without
     sources), `limiter` (reuse `limiter_kernel`), `primitive_forward`/`primitive_inverse`,
     `admissibility` (min of the positive expressions). Refuse every generic or authored flux
     without a declared absolute maximum (`FV_SPEED_UNDECLARED`); HLL additionally needs
     minimum and maximum. `ReconstructionVariables::Primitive` needs `system.primitive`
     (`FV_PRIMITIVE_MAP_MISSING`). Checks: consistency per component, antisymmetry via
     `check_facet_swap_symmetry` over all outputs; same tolerance rules as the scalar law.
     Obligations: consistency/antisymmetry/conservation per equation, `CflBound` per field,
     TVD only when m = 1. `run_point_components` returns all outputs in slot order.
2. `tests/fixtures/euler.res`: `Euler1D` (three scalar DG fields, `parameter gamma`),
   `Euler1DDirichlet` (one `boundary` block per field per region), `Euler2D` (`momentum:
   vector(2)`, tensor-literal flux, authored Rusanov as a 4-component constitutive using
   `average(normal_component(momentum_flux))[0]`, `jump(momentum)[0]`, `lambda =
   max(trace_minus(max_signal_speed), trace_plus(max_signal_speed))`), and a verbatim copy of
   the corpus model to prove it *reads* (m = 5) and lowering refuses with a typed code.
3. `tests/sv2c1b_systems.rs` per the lane list. Use `gamma = 1.5` for the exact primitive
   round trip (`gamma - 1` exact); the Sod bitwise LLF test must mirror the `.res` tree
   (`p = (gamma-1)*(E - ((0.5*m)*m)/rho)`), and the `|u|+c` check is to 1 ulp because `p`
   from `E` is one rounding away from the literal.
4. Then clippy/fmt/rustdoc, the STATUS section, and the full report.
