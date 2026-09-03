# Scientia status

Updated: 2026-09-03

Branch: `master`

Milestone: W7 Scientia lane (workspace `PLAN.md` §6 "W7"): prerequisite batch P landed;
runner-free SC packages, SV1-A `DerivativeRequest` production, and SC-W1 composition follow in
this order, each recorded here as it lands.

## Role

Scientia owns `.res` source and scientific/form semantics. It parses and resolves modules,
maintains one canonical model, performs semantic and structural analysis, derives variational and
local mathematical artifacts, and lowers local work into Malleus-owned structured kernel types.

Quantitas owns dimensions, quantity kinds, units, and registries. Resolvent owns consumer-neutral
exact algebra and symbolic differentiation. Malleus owns local kernel IR and execution. Finitum
owns concrete discretization/global operators. Krasis owns coupled runtime state. Methodus owns
numerical algorithms. Sinbad owns product orchestration. Solverang owns generalized constraint
solving over Methodus and is not part of the simulation execution dependency graph.

## Implemented

- Recovering `.res` parser, canonical formatter, byte-precise spans, deterministic module
  resolution (`ModuleSource`, the ordered `ModuleClosure`, scoped by-reference `use` imports
  with `GlobalDeclId`), and presentation-invariant source digest.
- One typed `SemanticModel` arena (`scientia-semantic/5`): stable domain/region/symbol/expression/
  declaration ids, resolved roles, shapes/axes, Quantitas dimensions/kinds/units with SI literal
  canonicalization, frames, provider signatures and typed provider calls (C1), and a
  presentation-invariant semantic digest. `compile_semantics` is the FC1 boundary.
- `scientia-binding-slots/1` (C2/C11.1): every case-bindable slot with typed identity and
  `Required | Unbound | ModelDefined` status; CLI `slots`. A defined `source x = expr;` is
  `ModelDefined`; `input field` / `input value` declarations are `input/<name>` slots (SC
  runner-free packages, below).
- `scientia-verification-profile/1` and `scientia-verification-obligation/2` (C6.1): typed
  dimension, invariant, manufactured-solution (exact `ExprId`), convergence, temporal-convergence,
  conservation, patch, rigid-body, limiting-case, inf-sup, and derivative-Taylor obligations;
  CLI `derive-verification`.
- `scientia-derivative-request/1` types objectives, observables, controls, design variables,
  active/frozen sets, products, conventions, and the fixed-topology shape slice.
  **`scientia-derivative-request/2`** (`LinkedDerivativeRequest`, SV1-A below) is the `.res`
  producer: `derive_derivative_request(compilation, spec)` from `objective`/`observable`
  declarations with `SymbolId`/`ExprId`/`DeclarationId`/slot links.
- `scientia-operator-structure/2` (C5.4 + SC §7): linearity, form symmetry, block
  coordinates/classes, saddle-point flag, nullspace candidates, property dependence, time
  structure, and the additive residual gauge (`block_symmetry`, `transpose_relation`,
  `sign_gauge` with a signed-graph proof); CLI `structure`.
- `ModuleDigest` and `SourceLocator { module, span }` (SC §2.1); `ResolvedModules` carries
  per-module digests.
- Structural incidence, matching, SCC/BLT, tearing, alias, and DAE planning; coupling graphs.
- `VariationalForm` (`scientia-variational-form/4`): authored forms and derived strong equations
  with generated typed test arguments, physical-field captures, space-aware integration by parts,
  and receipts. Boundary terms are `EliminatedByEssentialCondition`, `Substituted` (Neumann datum
  through the external-input path), or **`NaturallyClosed { flux }`** (batch P, below).
- FC3 `FormRequirements`: H1/L2/Hcurl/Hdiv/DG/trace spaces, pullbacks, orientations, evaluation
  sites and trace mappings, geometry preprocessing, quadrature intent, essential constraints,
  boundary partitions, canonically grouped integrals, and `REQ_*` refusals.
- FC4 `TensorProgram`/`QFunctionProgram`/`OperatorFactorization` with symbolic test
  differentiation, directional JVPs, and GX-A3 chain-rule property tangents; deterministic
  reference interpreters validated against an independent P1 Poisson fixture.
- FC5 `lower_operator_kernels`: complete Malleus primal/JVP/VJP/parameter bundles with receipts;
  GX-A2 property kernels (`scientia-property-kernel/1`) through the Resolvent projection.
- SC-W1 composition (below): `ScientificSystem` (`scientia-system/1`) with the system arena
  and `OriginMap`, `scientia-binding-slots/2` instance-prefixed slots, `model` as the
  implicit one-instance system, `system`/`instance`/`bind`, `output` declarations and their
  kernels, and `scientia-operator-system/2` with kernel-level `Composed` bind chains on
  Malleus `KernelComposition`.
- FC8 `OperatorSystem` (`scientia-operator-system/1`) for multi-equation systems; FC10
  `MethodProgram` compilers for FV/FD/network-DAE/particle/boundary-integral families; FC11 Serde
  round trips for every artifact.
- One `scientia` CLI; multi-model modules require explicit `Model:item` selection.

### Batch P (2026-09-01): natural boundaries, normal traces, provider calls in integrands

- **Natural closure.** A boundary region with no boundary condition for the equation's test
  field (including the GX-A6 implicit whole-boundary region) substitutes the zero flux datum:
  no exterior-facet integral is emitted, the receipt records
  `BoundaryTermDisposition::NaturallyClosed { flux: ExprId }` (the signed strong normal-flux
  expression) and `FormTransformation::SubstituteNaturalClosure { region }`. The former
  `Retained { integral_index }` state-computed facet integral is deleted: realizing it would
  cancel the integration by parts and enforce nothing, and Finitum's system path refused it
  anyway. A nonzero flux must be declared `neumann`.
- **Normal-trace redistribution (FC3 and FC4 agree).** A `Normal` trace mapping stays on the
  operand carrying the contracted axis and degenerates to a plain trace on a scalar factor
  (`n·(k grad T) = k (n·grad T)`); an opaque model-defined symbol under a normal trace is
  evaluated at the facet as one value and contracted with the normal, so its definition's leaves
  are plain traces. Inline products of two axis-carrying operands refuse
  `REQ_NORMAL_TRACE_NONLINEAR`; inline computed tensors (contraction, call, index, vector
  literal) under a normal trace refuse `REQ_NORMAL_TRACE_UNSUPPORTED` (name the flux in a
  `constitutive`). `TENSOR_SHAPE: normal trace requires a vector or rank-two tensor` no longer
  occurs on any corpus equation.
- **Provider-call lifting.** Every `ProviderCall` inside a derived residual term or boundary
  value becomes a compiler-synthesized property capture (`SymbolId::generated_for_expression`,
  `FormCapture.definition` = the call) bound exactly like a named `property`: External
  `ModelDefinedProperty` input, GX-A3 chain-rule tangent when scalar and differentiable, frozen
  coefficient otherwise. Generalizes the GX-facet bare-boundary-value rule. Authored `form`
  integrands are not lifted (they are compiled verbatim).
- Corpus effect (`tests/gx_facet_boundary.rs` sweep, `SINBAD_WORKSPACE` set): operator
  factorizations 45/128 → **97/128** (118 forms, 107 requirements, 56 equations with a naturally
  closed boundary). 08 (both), 16 (all three), 18 (both), 27 `energy`, 45 `fluid_energy`/
  `solid_energy`/`fluid_momentum` factor. Remaining failures are unrelated to this package:
  pressure rows with no test space (`incompressibility`), Robin laws (33), cross-domain measures
  (50), 4-vector integrands (38), rank-mismatched contractions (44, 49), `sym_grad` of a scalar
  momentum field (29).

### SC runner-free packages (2026-09-03): residual gauge, slot classification, locators

- **`scientia-operator-structure/2`** (`sinbad/ARCHITECTURE.md` §7, additive to C5.4):
  `block_symmetry: Vec<(SymbolId, FormSymmetry)>` per present diagonal block, decided by the
  evaluation-paired test/active exchange of every integral linear in its active inputs;
  `transpose_relation: Vec<TransposeRelation { row, column, sigma: Option<i8> }>` per
  two-sided off-diagonal pair, deciding `A_ij ≈ σ A_jiᵀ` by exchanging test/active roles,
  relabeling shared coefficient inputs by binding, renumbering axes, and splitting an overall
  sign; `sign_gauge: Option<SignGauge { signs: Vec<(SymbolId, i8)>, proof: SignedGraphBalance
  { edges, spanning_forest } }>` when every present diagonal block is `Symmetric`, none is
  `Convective`, every present coupling is two-sided with a decided `σ`, and the signed row
  graph is balanced (BFS spanning forest from each component's smallest row); otherwise
  `sign_gauge_reason` names the first failing condition (exactly one of the two is present,
  `STRUCTURE_INVALID` otherwise; ordering is `STRUCTURE_NONCANONICAL`). All four fields enter
  the identity digest. `form_symmetry` keeps its C5.4 meaning (still `Unknown` for every
  multi-block system). **Deviation from §7:** keys are per-model `SymbolId` row ids and the maps
  are sorted `Vec`s, because `SysResId` does not exist until SC-W1; SC-W1 re-keys them.
- **C5.4 block-class fix:** test evaluations are collected per column from the integrals in
  which that column is active, not pooled over the row. Corpus 13's flux mass block
  `K⁻¹ q · w` was `Convective` (the row's `-p div(w)` term contributed the test divergence);
  it is now `Reaction`, so Darcy gauges as `{flux: +1, pressure: -1}` with `σ = -1`.
- **Corpus effect:** 13 gauges. 25 does not: the momentum diagonal is `Unknown` because the
  viscous stress reaches the tensor program as one opaque `ModelDefinedConstitutive` input
  (its JVP is the frozen-coefficient zero), so no exchange is possible at the tensor level;
  `σ = -1` for both Stokes couplings is decided. Seeing through constitutive definitions
  (`2 μ sym_grad(u) : grad(v)`) needs expansion of the definition plus an index-symmetry
  normal form; recorded as the next `/2` refinement, not silently guessed.
- **Slots (§3.3):** `source x = expr;` is `ModelDefined` with its expression recorded (08's
  `joule`); valueless `source` stays `Required`. New declarations `input field x: Kind on D;`
  and `input value x: Kind;` (soft keyword `input`, only before `field`/`value`; a definition or
  a missing domain is `PARSE_SYNTAX`) elaborate as `SemanticDeclarationKind::InputField {
  domain }` with role `Source` and `InputValue` with role `Parameter`, so the form path treats
  them like today's external data; slots are `input/<name>`, always `Required`, kinds
  `ExternalValue` / `Parameter` (the frozen C2 `SlotKind` enum is not extended: Sinbad matches
  it exhaustively). `ScientificModel.inputs` is skipped from the digest projection when empty,
  so every existing module digest is unchanged. Schema ids stay `scientia-binding-slots/1` and
  `scientia-semantic/5`: both additions are purely additive and only appear when authored.
- **`SourceLocator { module: ModuleDigest, span: SourceSpan }`** (§2.1). `ModuleDigest` is the
  typed `semantic_digest` of one module; `ResolvedModules.module_digests` lists it per module;
  `SemanticCompilation::module_digest()` / `locate(span)` produce locators for the root module.
  `SourceSpan` gained `Ord`/`Hash`.
- **Sinbad consumer note:** `SlotKind::ExternalValue` with `status == ModelDefined` must not
  demand a case binding (`plan.rs` treats every `ExternalValue` as required today); `input/…`
  ids bind like `source/…` (field) and `parameter/…` (value).

### SV1-A (2026-09-03): `DerivativeRequest` production from `.res` objectives

- **Grammar:** `objective NAME { minimize|maximize|measure EXPR; }`. Elaborates as
  `SemanticDeclarationKind::Objective { value, sense }` with role `Observable`, so an objective
  is also an observable: it gets the `observable/<name>` slot (`ModelDefined`, expression
  recorded) and the verification profile's `ObservableDefinition`. A missing sense is
  `PARSE_SYNTAX`. `ScientificModel.objectives` is skipped from the digest projection when empty.
- **Producer:** `derive_derivative_request(&SemanticCompilation, &DerivativeRequestSpec)
  -> Result<LinkedDerivativeRequest, DerivativeRefusal>`. The spec names the model, an
  `objective` or `observable` (an observable is requested with sense `Measure`), active and
  frozen C2.1 slot ids, the product (`Jvp | Vjp | Gradient`), the state convention, and the
  level (`Discrete` default). The result (`scientia-derivative-request/2`) carries the
  unchanged `/1` `DerivativeRequest` record (`request`; Sinbad's own construction sites keep
  compiling), `ObjectiveLink { name, slot, declaration, symbol, expression, sense, dimension,
  quantity_kind, depends_on }` (the dependency set is closed over property, constitutive, and
  model-defined value definitions, so the misfit names `u` and `u_obs`), and
  `inputs: Vec<ActiveInputLink { name, role: DesignVariable | Control, active, kind, symbol,
  declaration, provider, expression, dimension, quantity_kind, gradient_dimension }>` sorted by
  slot id and equal to the `/1` active ∪ frozen partition; `identity` covers all of it and
  `validate()` checks the `/1` record, the partition (`DERIVATIVE_LINK_PARTITION`), the parent
  digest (`DERIVATIVE_PARENT_MISMATCH`), and the identity (`DERIVATIVE_IDENTITY_MISMATCH`).
- **Slot classification (GX decision 8: design variables bound to case property slots):**
  declared `provider/…`, valueless `parameter/…`, and `input value` slots are `/1`
  `DesignVariable`s (`parameter_owner = "case slot <id>"`, `dimension` from the signature);
  valueless `source/…`, `input field`, `boundary/…`, and `initial/…` slots are `/1`
  `Control`s with `support` = domain, region, or initial state. `gradient_dimension` is
  `dim(objective) / dim(input)` (inverse Poisson: dimensionless / Diffusivity = s/m²). A
  symbol with no declared quantity kind is the language's dimensionless scalar (elaboration
  already types its arithmetic so); a declared but unresolvable kind refuses.
- **Refusals (`DerivativeRefusal.code`):** `DERIVATIVE_UNKNOWN_MODEL`,
  `DERIVATIVE_UNKNOWN_OBJECTIVE`, `DERIVATIVE_OBJECTIVE_NOT_SCALAR`,
  `DERIVATIVE_OBJECTIVE_INVALID`, `DERIVATIVE_OBJECTIVE_DIMENSION_UNKNOWN`,
  `DERIVATIVE_NO_ACTIVE_INPUT`, `DERIVATIVE_DUPLICATE_INPUT`, `DERIVATIVE_UNKNOWN_SLOT`,
  `DERIVATIVE_UNBOUND_SLOT` (undeclared provider), `DERIVATIVE_MODEL_DEFINED_SLOT` (e.g.
  `property/k`; the message names the case slots its definition calls, `provider/diffusivity`),
  `DERIVATIVE_SLOT_NOT_DIFFERENTIABLE` (domain, region, observable: shape derivatives are
  SV1-G), `DERIVATIVE_SLOT_DIMENSION_UNKNOWN`, `DERIVATIVE_SCHEMA_MISMATCH`, plus the `/1`
  codes. Conventions: `Real`, `Smooth`, `shape: None`; dependence is `Partial` for
  `FixedState` and `Total` for `ConvergedState`/`AcceptedTrajectory`.
- **Tests:** `tests/sv1_a_derivative_request.rs` on the inverse-Poisson model (corpus 01 plus
  `input field u_obs: Dimensionless on Omega;` and `objective misfit { minimize
  integrate(0.5 * (u - u_obs) * (u - u_obs)); }`) and, opt-in, on the unmodified corpus
  `01-poisson.res` (`energy` as a `Measure` objective, VJP with respect to
  `provider/diffusivity`, `source/f` frozen).
- **Sinbad consumer note (E7):** call `derive_derivative_request` from the compiled case with
  the case's chosen slot ids; read `inputs[*].provider`/`symbol` to route the parameter VJP,
  `objective.expression` to evaluate `J` through the observable evaluator, and
  `gradient_dimension` for the gradient artifact's units. `derivative_campaign.rs`'s own
  `DerivativeRequest` literal should become a `LinkedDerivativeRequest` from this producer
  (decision 8). The `/1` `semantic_expression` is the formatter's canonical spelling of the
  authored expression. Not landed: a `derivative-request` CLI subcommand.

### SC-W1 (2026-09-03): scoped imports, systems, bind chains

- **Imports (§2.1, §3.2).** `use a.b.c;` (alias `c`), `use a.b.c as x;`, `use a.b.c.{A, B as
  C};`. Declarations are `pub model`, `pub system`, and module-level `pub provider`; an import
  resolves each name to `GlobalDeclId { module: ModuleDigest, kind: Model | System |
  Provider, name }` of a `pub` declaration of the target module and the arena records it as
  `SemanticModule.imports: Vec<SemanticImport { name, target, span }>` (skipped when empty:
  `scientia-semantic/5` digests of every existing module are unchanged). Only provider
  signatures enter model scope (selective items under their name, alias imports under
  `alias.name`); models and systems are instanced by reference. GX-F4 flatten-by-name is
  removed; `tests/gx_f4_module_resolution.rs` fixtures moved to `pub provider` catalogs. New
  refusals: `RESOLVE_UNKNOWN_IMPORT`, `RESOLVE_PRIVATE_DECLARATION`, `RESOLVE_AMBIGUOUS_PATH`
  (import name collides with a local declaration), `RESOLVE_DUPLICATE_IMPORT`.
  **Deviation from §3.2:** `.` is not a separate token; the lexer already folds a dotted path
  into one identifier (`0.5` lexes as a number first), and `a.b.{` is recognized by the
  trailing separator. `<-` is a new two-character operator. Region parameters (`region r:
  boundary of D;`) and `remainder(D)` are not landed; regions still come from `boundary("…")`
  and map per instance.
- **Module closure (§2.4).** `resolve_module_closure(source, &ModuleSource) -> ModuleClosure
  { schema: scientia-module-closure/1, root, modules: Vec<ClosureModule { name, digest, source,
  module }>, identity }`, dependencies before dependents, root last;
  `SemanticCompilation.closure` carries it; `compile_module_in_closure(closure, name,
  registries)` elaborates any member.
- **Model additions (§3.3).** `output NAME: Kind on Domain = expr;` (ascription optional only
  for a bare owned field) elaborates as `SemanticDeclarationKind::Output { value, domain }`
  with the new `SemanticRole::Output`; `TYPE_OUTPUT_DIMENSION` / `RESOLVE_OUTPUT_DOMAIN`
  refusals. `derive_output_form(module, model, output)` derives the output kernel through the
  unchanged FC2–FC5 chain as the synthetic functional `∫ G · w` against a generated
  `L2(order=0)` test argument: the test-dual primal output is `G` at the point and the JVP is
  `dG/dx`. FC5 gained `hoist_reductions` (sum distributivity: `a · Σ f = Σ a · f` when `a` is
  axis-free) so `σ |∇V|²` lowers; nothing else in FC5 changed.
- **`scientia-system/1` (`ScientificSystem`, §1–§3, §5).** `compile_system(closure,
  registries, name)` for a declared `pub? system { domain …; instance a: Model(param =
  domain, …); bind a.x <- b.y; }`, `compile_model_system(closure, registries, model)` for the
  implicit one-instance system (instance 0, empty prefix, identity domain map). Dense ids
  `InstanceId`, `SysVarId` (owned `unknown`/`state` fields), `SysResId` (equations, with
  `orientation ±1` and `OrientationBasis::{Accumulation, Unoriented}` from the `dt` term),
  `SysDomainId`, `SysRegionId` (one per instance region, display `a.region`), `OutputId`;
  `OriginMap { variables: Vec<VariableOrigin { variable, instance, model: GlobalDeclId, symbol,
  locator }>, residuals, regions, domains }`; `SystemSlotManifest` (`scientia-binding-slots/2`)
  = every instance's `/1` manifest under `<instance>/` with `SlotBinding::{Open, Bound {
  bind }}`; `SystemBind { consumer, consumer_slot, consumer_symbol, producer: OutputId, chain:
  BoundChain::Composed, locator }`; `SystemDependency { edges, components (Tarjan SCCs in
  condensation order), sequential }`; `require_closed(case_bound)` / `open_inputs()` for
  acceptance test 4. Bind targets are `Required` `input/`, `source/`, or `parameter/` slots;
  producers are `output`s. Refusals: `SYSTEM_UNKNOWN_SYSTEM`, `SYSTEM_UNKNOWN_MODEL`,
  `SYSTEM_DUPLICATE_INSTANCE`, `SYSTEM_ELABORATION`, `SYSTEM_UNKNOWN_PARAMETER`,
  `SYSTEM_UNKNOWN_DOMAIN`, `SYSTEM_DOMAIN_UNMAPPED`, `SYSTEM_DOMAIN_MISMATCH`,
  `SYSTEM_UNKNOWN_INSTANCE`, `SYSTEM_UNKNOWN_INPUT`, `SYSTEM_UNKNOWN_OUTPUT`,
  `SYSTEM_PRIVATE_SYMBOL`, `SYSTEM_DUPLICATE_BINDING`, `SYSTEM_BIND_KIND_MISMATCH`
  (dimension, quantity kind, or shape), `SYSTEM_BIND_SUPPORT_MISMATCH` (input value ← field
  output), `SYSTEM_BIND_CROSS_DOMAIN` (relations are SC-W2), `SYSTEM_OPEN_INPUT`. System-level
  observables, `connect`/`port`/`connector`, `oriented by`, and domain relations are SC-W2.
- **`scientia-operator-system/2` (`SystemOperator`, §2.6, §6).**
  `compile_system_operator(&SystemCompilation) -> SystemOperatorCompilation { system, operator,
  model_systems: Vec<(InstanceId, OperatorSystem /1)>, output_kernels, compositions }`. Per
  model, one unchanged `/1` artifact over all its equations (two instances of one model carry
  equal digests); rows `SysResBlock { id, origin, orientation, model_system: Digest, block:
  Digest }` reference per-model blocks by the new `block_digest`; `SysBlock { row: SysResId,
  column: SysVarId, construction: Local { kernels } | Composed { bind, consumer_slot,
  producer, output_kernels, path } }`. `ComposedPath::KernelInput { compositions,
  jvp_compositions }` when a consumer residual kernel reads the bound input as an operand:
  one Malleus `KernelComposition` per consumer bundle (stage 0 = producer output primal
  kernel, stage 1 = consumer primal kernel, one `SharedBuffer` from the output operand to
  every access operand of the input), validated by `malleus::validate_composition`, digested
  by `composition_digest`, and differentiated by `differentiate_composition` (producer active
  evaluations → consumer output) for the cross-block tangent. `ComposedPath::ProviderInput {
  properties: Vec<PropertyPath { property, slot, providers }> }` when the input is read only
  as a provider-call argument inside model-defined properties/constitutive laws (`sigma =
  electrical_conductivity(temperature)`): the value feeds the property's provider evaluation
  through Finitum's `FieldSource`; the tangent chains the consumer's frozen-input parameter
  JVP, the property's own GX-A2 tangent, and the output JVP, and Scientia emits no fused
  composition. **Deviation from §2.6:** the `/2` type is `SystemOperator`, not a rename of
  `OperatorSystem` (Sinbad constructs and consumes the `/1` type); `Transferred` and
  `Relation` constructions wait for SC-W2. Refusals: `SYSTEM_OPERATOR_NO_RESIDUALS`,
  `SYSTEM_OPERATOR_MODEL`, `SYSTEM_OPERATOR_OUTPUT`, `SYSTEM_OPERATOR_UNBOUND_COLUMN`,
  `SYSTEM_OPERATOR_BIND_UNUSED`, `SYSTEM_OPERATOR_COMPOSITION`.
- **Acceptance (`tests/sc_w1_composition.rs`).** Test 1: two `HeatConduction` instances have
  disjoint `a/…`/`b/…` slots and byte-equal `/1` artifacts. Test 2: `HeatConduction` direct
  and as the implicit system: equal `/1` artifact, equal block digests, identical unprefixed
  slot ids; opt-in, all 50 corpus models compile as implicit systems with exactly their `/1`
  slot ids and 08 compiles the full `/2` artifact monolithically. Test 3 (Scientia half): the
  composed electrothermal (`physics.electrical` + `physics.thermal` + `systems.electrothermal`,
  the §3.8 example in today's grammar) elaborates with one SCC `{electrical, thermal}`, both
  binds `Composed`, `thermal/input/Q` on the kernel-input path with validated compositions and
  JVPs, `electrical/input/temperature` on the provider-input path through `sigma`; identities
  are deterministic and round-trip. Value agreement with monolithic 08 at sampled states
  (SC3a) is not yet executed. Test 4: dropping `bind thermal.Q` opens `thermal/input/Q`
  (`SYSTEM_OPEN_INPUT`), re-binding it to case data yields a DAG scheduled thermal → electrical.
  Test 5 (W1 part): duplicate producers, private symbols, kind/support/cross-domain
  mismatches, and domain-parameter errors are typed refusals; port items are SC-W2.
- **Consumer notes.** Finitum/Sinbad key by `SysVarId`/`SysResId`/`SysRegionId` from
  `ScientificSystem`; `SysVar.local`/`owner` and `OriginMap` give the per-model coordinates;
  slot ids are `<instance>/<local id>` with the implicit root unprefixed, so every existing
  case file is unchanged. A `Composed` block's kernels are the referenced per-model bundles
  plus the `KernelComposition`s in `SystemOperatorCompilation.compositions` (never fused).

## Removed

The pre-form pipeline and its frontends, duplicate form/discrete/operator/backend types,
reference FEM implementations, bridge layers, comparison tooling, runtime plans, the internal
quantity crate, and the exact-CAS ADR corpus. Git history is the archive; none of it is an
acceptance oracle.

## Validation

Verified locally on 2026-09-03 (SC runner-free tree):

- `cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings`,
  `RUSTDOCFLAGS='-D warnings' cargo doc --no-deps` -- passed.
- `cargo test` (lib + every integration binary, run per binary with `SINBAD_WORKSPACE` set) --
  passed: 174 tests, 0 failed (24 lib, 150 integration across 28 binaries including
  `tests/sc_sign_gauge.rs` (5), `tests/sc_slots_inputs.rs` (4),
  `tests/sv1_a_derivative_request.rs` (5), and `tests/sc_w1_composition.rs` (8)).
- `cargo check` of the Sinbad and Finitum checkouts against this working tree passed
  (2026-09-03).
- Corpus sweep: 50/50 elaborate; 97/128 operator factorizations (see batch P above).
- The GX exit gate passed on 2026-08-31 (Sinbad `a1402f2`); C5.4/C6.1 shapes are corpus-verified
  against `25-stokes.res`/`13-mixed-darcy.res` (`207bb2e`).

## Cross-repository contract

- Resolvent path `../resolvent` (RV0 `f686190b`); Quantitas path `../quantitas` (`734d78cd`;
  `Dimension`, `Quantity`, `QuantityLiteral`, `QuantityKindId`, `UnitId`, `UnitRegistry`);
  Malleus path `../malleus` (Scientia constructs Malleus modules, operands, affine maps,
  expressions, derivative requests, statements, and numeric policies directly; the lockfile
  follows Malleus's W7 dependency additions).
- Public downstream sequence: `compile_variational_form`/`derive_variational_form` ->
  `infer_form_requirements` -> `factor_operator` -> `lower_operator_kernels`; sibling families
  start at `compile_*_method` and produce `MethodProgram`.
- Finitum maps `LocalIterationContract::QuadraturePoint` across elements and quadrature points;
  fixed-axis batching stays realization-owned (`ITERATION-OWNERSHIP.md`).
- **Batch P consumer note (Finitum, Sinbad):** a naturally closed boundary produces no facet
  integral, so 08-shaped models realize with cell integrals only; the `BoundaryPartitionRequirement`
  for the implicit region is still emitted and must still be discharged against topology. A
  `QFunctionInput` whose `binding.symbol.is_generated()` and whose `source` is
  `ModelDefinedProperty { definition }` is a lifted provider call; its data binding is the
  provider's `provider/<name>` slot, unchanged.

## Known limits

- `evidence.rs` has no consumer. `derive_derivative_request` has no CLI subcommand yet;
  shape design variables (`shape: None`) are SV1-G.
- FC4 JVPs inline only scalar properties whose definition wraps differentiable provider calls;
  vector-valued provider outputs (`convect`, `gravity_vector`) and constitutive laws stay frozen
  (Picard) coefficients, named truthfully in the derivative receipt.
- Generated test-argument and lifted-capture `SymbolId`s are per-declaration/per-expression
  (`GENERATED_BASE` high bit) and never index `model.symbols`.
- `form_symmetry` is `Unknown` for every multi-block `OperatorSystem`; the `/2` `sign_gauge`
  is the gauged claim. A diagonal block whose active dependence is hidden behind an opaque
  constitutive input (Stokes viscous stress) is `Unknown` and blocks the gauge with a reason.
  `[system].equation_sign` is deleted from `sinbad-case/2` on Sinbad's side once it consumes
  the gauge.
- Non-Cartesian coordinate systems validate and are then ignored downstream. `si_bootstrap`
  coverage is what the corpus needs, no more.
- SC-W1 does not execute the composed-vs-monolithic value comparison (SC3a) or land
  `connector`/`port`/`connect`, region parameters, domain relations, system observables, or
  `Transferred` chains (SC-W2). There is no `system` CLI subcommand yet.

## Next compiler work (W7 lane order)

1. **Done:** batch P (above).
2. **Done:** runner-free SC packages (above).
3. **Done:** SV1-A (above).
4. **Done:** SC-W1 (above). Next: SC3a value agreement of the composed electrothermal
   kernels with monolithic 08 through the Malleus reference interpreter; SC-W2 (`connector`/
   `port`/`connect`, region parameters, relations, `Transferred` chains); a `system` CLI
   subcommand; the `/2` sign gauge through opaque constitutive inputs (Stokes).

Deviations from `sinbad/ARCHITECTURE.md` are recorded in the batch/package sections above with
their reasons; the coordinator folds them into GX-CONTRACTS C12. Scientia remains execution-free.
