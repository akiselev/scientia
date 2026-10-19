# Scientia status

2026-10-19 W12 lane L1 (SV2-C1a, working clone; not yet integrated or consumer-accepted):
conservation-law semantics and finite-volume face kernels in `src/conservation.rs`.

## SV2-C1a conservation laws and FV face kernels

- `compile_conservation_law_system(module, model, equation, conserved)` ->
  `ConservationLawSystem` (`scientia-conservation-law/1`). It reuses the unchanged FC10
  selection (`select_conservation_law`, shared with `compile_conservation_law_method`), requires
  a scalar `DG` conserved field, normalizes `dt(u) + div(flux_sign*F) = S` from signed additive
  terms and records `dirichlet` boundaries as `GhostState` and `neumann` ones as
  `PrescribedFlux`. `check_boundary_closure(exterior, periodic)` refuses unclosed regions.
- `FvMethodIntent` (`scientia-fv-method-intent/1`, decision D1): numerical flux `central |
  local_lax_friedrichs | hll | authored{declaration, tolerance}`, reconstruction `constant |
  muscl{variables, gradient, limiter}`, source `pointwise`, positivity `none |
  zhang_shu_scaling`. JSON is pinned in `sv2c1_conservation.rs`.
- `lower_conservation_law(module, system, intent, samples)` -> `ConservationLawProgram`
  (`scientia-conservation-law-program/1`) with `FvKernel`s (`scientia-fv-kernel/1`):
  `physical_flux_normal` (lowered from the semantic F, not a coefficient table),
  `characteristic_speed` (`|d(F.n)/du|` from the Malleus JVP, inlined), `face_flux` (a Malleus
  facet pair: `u_minus`/`u_plus` partnered cells, normals `Odd`, data `Even`, flux `Odd`),
  boundary ghost/prescribed-flux kernels, `source`, and a Sweby limiter `phi(r)` (minmod, van
  Leer, MC, superbee). Every operand has an `FvOperandMeaning`; data are listed in `data`.
- Generic families only (D3). Authored fluxes are scalar `constitutive` two-point expressions
  over `trace_minus/trace_plus`, `jump` (minus - plus), `average`, `normal_component[_minus]`
  (`G.n`) and `normal_component_plus` (`G+.(-n)`). Consistency and antisymmetry
  (`malleus::check_facet_swap_symmetry`) are checked at caller samples; generic fluxes check at
  tolerance 0 when samples are given.
- D6 (FV path only): `FvKernel::differentiate` refuses a request whose independents reach the
  switch of `abs/min/max/floor/ceil` (`nonsmooth_sites`, dependency through locals). `select`
  is no kink. Smooth flux kernels and smooth directions (state direction of linear-advection LLF)
  keep their derivatives.
- Obligations (new `VerificationObligationKind` variants): `NumericalFluxConsistency`,
  `NumericalFluxAntisymmetry`, `CflBound{field, forward_euler_limit}` (1 constant, 1/2 MUSCL),
  `TotalVariationDiminishing` (LLF/HLL with constant or Sweby-limited MUSCL), and
  `Conservation{relation: DiscreteFluxBalance}` (new `ConservationRelation` variant).
- Refusal codes: `FV_METHOD_SELECTION`, `FV_SYSTEM_UNSUPPORTED`, `FV_STORAGE_SCALED`,
  `FV_FLUX_NOT_CONSERVATIVE`, `FV_FLUX_SHAPE`, `FV_FLUX_UNSUPPORTED`,
  `FV_FLUX_FIELD_UNSUPPORTED`, `FV_FLUX_PROVIDER_UNSUPPORTED`, `FV_DIFFUSIVE_FLUX_UNSUPPORTED`,
  `FV_BOUNDARY_UNSUPPORTED`, `FV_BOUNDARY_UNCLOSED`, `FV_SOURCE_UNSUPPORTED`,
  `FV_AUTHORED_FLUX_{MISSING,SHAPE,UNTRACED,UNCHECKED,INCONSISTENT,NOT_ANTISYMMETRIC}`,
  `FV_PRIMITIVE_MAP_MISSING`, `FV_LIMITER_UNSUPPORTED`, `FV_POSITIVITY_UNSUPPORTED`,
  `FV_NONSMOOTH_DERIVATIVE_REFUSED`, `FV_SAMPLE_INVALID`, `FV_KERNEL_INVALID`,
  `FV_DERIVATIVE_FAILED`, `FV_IDENTITY_MISMATCH`.
- Evidence: `tests/sv2c1_conservation.rs` (10 tests: LLF/HLL/central equal hand formulas
  bitwise on 64 samples incl. signed zero and ties; exact swap symmetry for 3 families x 2
  laws; authored Rusanov accepted, one-sided/doubled/untraced refused; D6 refusals and kept
  smooth JVP; boundary/source/limiter values; `run_batch` face block with stride-0 data equals
  per-face interpreter bitwise; obligations and identity). `tests/sv2c1_identity_pins.rs` pins
  the FC10 affine FV program digest, kernel digest and serialized bytes from before the change.
  Fixture: `tests/fixtures/scalar-conservation.res` (1-D/2-D linear advection, Burgers with a
  source, authored 2-D fluxes, a diffusive refusal case).
- Open (not in L1): systems/`SpectralBound`/declared speeds (C1b); Barth-Jespersen and
  Venkatakrishnan limiters; primitive maps; Zhang-Shu; diffusive TPFA; state-dependent providers
  and coefficient fields in fluxes; `RiemannProblem` for `@shock_tube`; a `.res` normal symbol
  (1-D authored fluxes cannot use `normal_component` on a scalar flux today).
- Consumers: Finitum (L4) binds face-flux operands by `FvOperandMeaning` (minus = owner cell,
  plus = neighbour or ghost from the region's `Ghost` kernel; `PrescribedBoundaryFlux` replaces
  the face flux), runs `face_flux` through `run_batch` (data stride 0), realizes
  `DiscreteFluxBalance`/`CflBound` from `characteristic_speed`, and uses `limiter` for MUSCL.
  Sinbad (L7) parses `[discretization]` into `FvMethodIntent` (serde shape above), supplies
  samples for authored fluxes, and must handle the new obligation kinds (its matches on
  `VerificationObligationKind` gain four variants). There is no nonsmooth policy field: FV
  derivatives through kinks are always refused.
- Lane gate: `cargo test -p scientia` 208 passed across 35 targets (also with
  `SINBAD_WORKSPACE=/projects/sinbad`), fmt check, all-target clippy `-D warnings`, rustdoc
  `-D warnings`.
- Build note: the W11 Malleus pin (`1585fc4`) adds `libm` to `Cargo.lock`. Lane validation ran
  with `CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0` (disk at 100%), opt-level 2.


2026-09-18 bounded implementation accepted:
SHOW-3: ports admit one explicitly signed flux divergence plus storage/source terms. Scaled, multiple, hidden and ambiguous divergences refuse. Transient thermal and species-diffusion compiler tests pass (5 focused tests); complete owner gate passed: 197 tests across 32 targets, formatting, check, strict clippy, rustdoc and doctests. Final consumer acceptance passed: 210 tests across 35 targets, with a documented external-fixture target retry and unchanged source. The explicit nonconjugating `frobenius(a,b)` intrinsic lowers through the existing Contraction IR; tensor-energy and vector-norm primal/VJP tests pass. Conjugating `inner` retains its typed refusal.


Historical September 17 checkpoint (superseded by the accepted extension above):
SC-W2 first matching-interface implementation: connector, region,
port, interface and connect declarations; explicit outward flux orientation;
source-located typed two-port ConnectionSet; quantity-kind/conservation checks;
Open boundary receipts. Unclosed/overlapping/wrong-kind/duplicate-argument and
unorientable declarations refuse. A standalone model with ports cannot run
without closure. General per-stratum matching, n-way merging, transient ports
and multiplier semantics remain planned; this is not all of SC-W2.
Owner gate: 193 tests across 32 targets, fmt, strict all-feature clippy, rustdoc
and doctests pass with unchanged source hashes. Consumer acceptance is pending.
Quantitas dependency: `c4c05b482a4fd8996dfdccac12316100a3b7f18d`.
Evidence: `docs/validation/2026-09-17-sc-w2/`.


Updated: 2026-10-19 (SV2-C1a section above); earlier entries 2026-09-17/18.
Branch: `master`.

## Ownership

Scientia owns source parsing, semantic identity, scientific/form meaning, symbolic lowering
and local executable artifacts. Quantitas owns quantities; Resolvent owns exact algebra;
Malleus owns local kernel IR/execution; Finitum owns discretization, sampling, quadrature,
accumulation and global operator actions; Krasis owns transactional coupled state; Methodus
owns physics-neutral numerical algorithms; Sinbad owns cases, runs and evidence.
Scientia contains no mesh, DOF, runtime solver or alternate scientific expression IR.

## Current capabilities

- Recovering `.res` parser/formatter and byte-precise spans; deterministic module closure,
  scoped imports, public declarations, `GlobalDeclId`, module digests and source locators.
- Canonical `scientia-semantic/5` arena with stable ids, dimensions/kinds/units through
  Quantitas, SI literals, frames, typed provider calls and presentation-invariant identity.
- `scientia-binding-slots/1`: case-bindable data and `Required | Unbound | ModelDefined`
  dispositions. Defined sources are model-defined; `input field` and `input value` are
  always external slots. `/2` prefixes slots by instance, except the implicit root.
- Verification profile/obligations: dimensions, invariants, manufactured solutions,
  convergence, conservation, patch/rigid-body, limiting cases, inf-sup and derivative Taylor.
- `scientia-derivative-request/2`: objective/observable expression identity and dependency
  closure, design/control slot links, active/frozen partitions, product/state convention and
  gradient dimensions. Its `/1` payload remains the derivative request vocabulary.
- Structural matching, SCC/BLT/tearing, dependency/coupling analysis and DAE planning.
- `scientia-operator-structure/2`: linearity, symmetry, block classes, saddle points,
  nullspaces, property/time dependence, pairwise transpose relations and a signed-graph
  residual gauge. Per-block facts stay distinct from global multi-block symmetry.
- Variational forms `/6`: generated tests, typed captures, integration by parts, provenance,
  provider-call lifting and natural boundary closure. Undeclared flux boundaries close with
  zero datum and retain a receipt; declared Neumann values become external facet inputs.
- FC3 requirements: spaces, pullbacks/orientations, evaluation/trace mappings, preprocessing,
  quadrature intent, constraints/boundary partitions and typed unsupported cases.
- FC4 tensor/QFunction factorization and symbolic JVPs. GX-A3 scalar differentiable property
  tangents remain explicitly represented; opaque constitutive inputs in the legacy residual
  path remain frozen and must not be described as complete Newton derivatives.
- FC5 Malleus primal/JVP/VJP/parameter bundles and property kernels through Resolvent.
- FC8 multi-equation `OperatorSystem` `/1`; FC10 separate FV/FD/network/particle/boundary
  integral method programs; artifact serialization/identity across the compiler chain.

## SC-W1 reusable composition

- `ScientificSystem` `/1`: models as implicit single instances; declared systems/instances,
  domain parameters, inputs/outputs, binds, global variable/residual identities and origin map.
- `/2` system operators reference reusable `/1` model artifacts and output kernel artifacts;
  bind chains use Malleus `KernelComposition`, without expression rewriting across instances.
- Direct kernel-input binds emit primal compositions and JVP identities. Provider-input
  binds expose the property path and require the runtime property tangent chain.
- Two instances of one model have disjoint prefixed slots and equal local artifact digests.
  Implicit models preserve `/1` artifacts and unprefixed slot identities. Implicit variable
  ids match `SystemIdMap::one_instance`; declared systems allocate dense instance ids.
- SC3a kernel evidence: composed electrical Joule output feeding thermal Q agrees with the
  monolithic 08 term at sampled points to 1e-12 relative. This is not solution/trajectory proof.
- Finitum consumes `SysVarId`/`SysResId`; local symbols and declaration ids remain in origins.
  Sinbad owns declared case/run/verify trees, which Scientia parses without product semantics.

## W8 S-EVAL: compiler-owned expressions (`d27780d`)

- Public `compile_point_expression(module, model, declaration, expression, domain)` and
  `compile_cell_functional(module, model, declaration_name)` produce
  `scientia-point-expression-kernels/1`. The latter accepts one `integrate(integrand)` on an
  unambiguous cell domain. Point values have no quadrature, geometry or synthetic test weight.
- Reuses `VariationalForm`, `FormRequirements`, `OperatorFactorization`,
  `StructuredOperatorKernels` and Malleus `DerivativeProduct`; no new universal expression IR.
  Existing `derive_output_form` follows its unchanged non-expanding path.
- Point lowering expands nested authored value/property/constitutive definitions in the form
  arena, memoized by expression id. Original canonical source and declaration attribution stay
  intact; unrelated expression/declaration attribution is refused.
- Provider calls remain captures with original expression/provider ids and explicit
  `ProviderValueAndProductsRequired` disposition. Each argument has a compiled point node in
  a shared `ExprId` DAG; nested or reused arguments are not copied exponentially.
- Bundles expose direct state JVP/VJP and captured-input JVP; each node also exposes a Malleus
  parameter VJP. These are explicitly partial products: total derivatives require the
  provider argument products and callback chain. External inputs may be active design/control
  data or explicitly held fixed by the consumer. Missing products never imply zero.
- Finitum must remap graph-local TensorInputIds through each input's semantic symbol and
  derivative evaluation; ids are not interchangeable between nodes. Parameter VJP consumers
  use the Malleus `primal_operands` remap, not assumed operand index preservation.
- Artifact identity covers support, source identity, all lowered payloads, argument DAG and
  capture dispositions. `validate_identity` detects mutation after transport, not source trust.
- `infer_expression_polynomial_degree` accepts explicit per-symbol and per-provider-call
  degree evidence; products add degrees, sums take maxima, and spatial derivatives reduce
  degree. Unknown data/providers remain unknown. Coordinates have no name-based shortcut;
  fixed-time spatial degree is distinct from a time-derivative degree bound.
- Supported evidence: inverse-Poisson-shaped misfit; elasticity vector squared norm;
  nested scalar and symmetric-gradient tensor constitutive expressions; provider argument
  forward/reverse chains, captured-data products, shared nested DAG and identity mutations.

## W8 property activity and tensor trace (working tree)

- `PropertyKernel::reads_input(name)` maps signature positions to generated primal operands
  and delegates output-dependency analysis to Malleus. It distinguishes declared unused
  arguments from actual reads, including through temporaries and multiple outputs. Invalid
  kernel/signature mappings refuse; no symbolic interpretation or numerical execution occurs.
- This derived query does not change property kernel artifacts or identities. Consumers must
  require a tangent for actually read arguments; unused analytic-provided arguments are not
  missing derivatives. Branch dependencies remain conservative; no cancellation proof.
- `TensorTrace` lowers to a diagonal finite sum over its authored axes. Structured lowering
  admits additive terms beside sums using positive-extent normalization, e.g.
  `sum(f)+g = sum(f+g/n)`. This defines floating-point grouping for previously unsupported
  point expressions; it is not a bitwise migration claim. No Malleus primitive or new IR added.
- Hermetic corpus-17 stress fixture checks primal, state JVP and accumulated state VJP;
  independent vector-gradient trace and additive/subtractive terms check dissimilar consumers.
  Existing supported kernel paths remain unchanged.

## Validation

- Focused property-kernel tests: 5 passed; point-expression tests: 13 passed.
- `cargo test -q -p scientia`: 191 passed (26 library, 165 integration), none ignored.
- Seven workspace-enabled suites with `SINBAD_WORKSPACE=/projects/sinbad`: 42 passed
  (binding slots, operator structure, verification, facets, sign gauge, composition, derivatives).
- All-target clippy and rustdoc with warnings denied, scoped formatting and diff checks passed.
- The stale negative reduction-plus-constant test now probes reciprocal-of-reduction, which
  remains unsupported; new additive tests establish the newly admitted behavior.
- Input activity prerequisite: Malleus `02164b3`, public validated primal-output dependency query.

## Live dependency snapshot

S-EVAL uses sibling paths at these committed heads:

- Malleus `9862a080499c3203992871402f4836bc00edb68b`.
- Resolvent `3185f229435f459beafcf3c65435617d8aaa8319`.
- Quantitas `69d3ac3e5b2a9be77dcdc0547d8c6f335827555c`.

Finitum and Sinbad are downstream consumers under concurrent W8 development. Exact integration
snapshots belong to workspace gates; a Scientia commit alone does not pin sibling sources.

## Supported/refused boundaries and next work

- Point functional lowering is Cartesian/cell-only. Boundary/nested integrals, ambiguous
  domains, nonlocal facet expressions and unsupported tensor constructs refuse with `POINT_*`.
  Explicit conjugating `inner` currently retains FC4's real64 contraction refusal; ordinary
  vector `dot`, nonconjugating vector/tensor `frobenius`, and tensor constitutive point values are demonstrated. No silent downgrade.
- Provider products admit Symbolic/Automatic/AnalyticProvided contracts; None, Piecewise and
  NumericalAllowed require a future explicit derivative policy and currently refuse.
- General complex kernels, non-Cartesian realization, history-dependent local updates and
  complete coupled/transient adjoints are not established here.
- Legacy residual kernels still have opaque constitutive/frozen coefficients; the new point
  API enables owning-layer execution, but does not by itself migrate residual consumers.
- Finitum must execute point graphs, sample/reconstruct fields, choose/record quadrature and
  accumulate state/design covectors. Sinbad must replace its interpreters and sampler copies.
- SC-W2 ports/connectors/interfaces, nonmatching transfers and system observables follow W8;
  SC-W1 kernel-level composition remains narrower evidence than end-to-end declared solves.
- Historical compiler implementations and removed compatibility layers remain in Git history,
  not as alternate authorities. Keep this ledger compact; workspace plans own lane ordering.

Final cross-repository evidence: [September 18 acceptance](../sinbad/docs/validation/2026-09-18-assembly/README.md).
