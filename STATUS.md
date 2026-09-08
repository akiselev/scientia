# Scientia status

Updated: 2026-09-08
Branch: `master`, committed base `2871d959d0cfe4a09270523823e2da699b6b7ba4`.
Milestone: W8 S-EVAL additive compiler prerequisite implemented in the working tree;
Finitum functional execution and Sinbad consumption remain separate gates.

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

## W8 S-EVAL: compiler-owned expressions (working tree)

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

## Validation

Current W8 working tree:

- `cargo check -q`: passed.
- `cargo test -q --test w8_point_expression`: 10 passed.
- `cargo test -q`: 186 passed (24 library, 162 integration). An initial full-suite
  attempt hit `ExecutableFileBusy` in CLI tests during concurrent executable relinking;
  the serial rerun passed without changing tests.
- Clippy all targets with warnings denied, rustdoc with warnings denied, formatting
  and diff checks passed.
- Coordinator reran all seven integration binaries containing `SINBAD_WORKSPACE`
  gates with `SINBAD_WORKSPACE=/projects/sinbad`: 42/42 passed, including complete
  corpus slot derivation and implicit-system compilation, structure/verification,
  factorization reporting, composition and derivative requests.

Prior committed evidence (2026-09-03, not rerun counts for this working tree):

- 176 tests passed (24 library, 152 integration); fmt, clippy and rustdoc passed.
- Corpus: 50/50 elaborate; 97/128 factorizations after natural boundary/provider lifting.
  Kernel-level SC-W1 evidence is described above; no product solve claim follows from it.

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
  vector `dot` and tensor constitutive point values are demonstrated. No silent downgrade.
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
