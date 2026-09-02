# Scientia status

Updated: 2026-09-01

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
  resolution (`ModuleSource`/`resolve_modules`, GX-F4 provider-only `use` import), and
  presentation-invariant source digest.
- One typed `SemanticModel` arena (`scientia-semantic/5`): stable domain/region/symbol/expression/
  declaration ids, resolved roles, shapes/axes, Quantitas dimensions/kinds/units with SI literal
  canonicalization, frames, provider signatures and typed provider calls (C1), and a
  presentation-invariant semantic digest. `compile_semantics` is the FC1 boundary.
- `scientia-binding-slots/1` (C2/C11.1): every case-bindable slot with typed identity and
  `Required | Unbound | ModelDefined` status; CLI `slots`.
- `scientia-verification-profile/1` and `scientia-verification-obligation/2` (C6.1): typed
  dimension, invariant, manufactured-solution (exact `ExprId`), convergence, temporal-convergence,
  conservation, patch, rigid-body, limiting-case, inf-sup, and derivative-Taylor obligations;
  CLI `derive-verification`.
- `scientia-derivative-request/1` types objectives, observables, controls, design variables,
  active/frozen sets, products, conventions, and the fixed-topology shape slice; there is no
  producer from `.res` yet (SV1-A, this wave).
- `scientia-operator-structure/1` (C5.4): linearity, form symmetry, block coordinates/classes,
  saddle-point flag, nullspace candidates, property dependence, time structure; CLI `structure`.
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

## Removed

The pre-form pipeline and its frontends, duplicate form/discrete/operator/backend types,
reference FEM implementations, bridge layers, comparison tooling, runtime plans, the internal
quantity crate, and the exact-CAS ADR corpus. Git history is the archive; none of it is an
acceptance oracle.

## Validation

Verified locally on 2026-09-01 (batch P tree):

- `cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings`,
  `RUSTDOCFLAGS='-D warnings' cargo doc --no-deps` -- passed.
- `cargo test` (lib + every integration binary, run per binary with `SINBAD_WORKSPACE` set) --
  passed: 152 tests, 0 failed (24 lib, 128 integration across 24 binaries including the new
  `tests/p_natural_closure.rs`, 5 tests).
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

- `DerivativeRequest` has no `.res` producer (SV1-A, this wave); `evidence.rs` has no consumer.
- FC4 JVPs inline only scalar properties whose definition wraps differentiable provider calls;
  vector-valued provider outputs (`convect`, `gravity_vector`) and constitutive laws stay frozen
  (Picard) coefficients, named truthfully in the derivative receipt.
- Generated test-argument and lifted-capture `SymbolId`s are per-declaration/per-expression
  (`GENERATED_BASE` high bit) and never index `model.symbols`.
- `form_symmetry` is `Unknown` for every multi-block `OperatorSystem` (the `/2` sign gauge is the
  next package). `[system].equation_sign` remains Sinbad case data until then.
- Non-Cartesian coordinate systems validate and are then ignored downstream. `si_bootstrap`
  coverage is what the corpus needs, no more.
- `use` imports resolve provider declarations only (GX-F4 flatten-by-name); models are not
  importable until SC-W1's `GlobalDeclId` import lands.

## Next compiler work (W7 lane order)

1. **Done:** batch P (above).
2. Runner-free SC packages (`sinbad/ARCHITECTURE.md` §7, §3.3, §2.1): `scientia-operator-structure/2`
   with per-block `block_symmetry`, per-pair `transpose_relation`, and a signed-graph
   `sign_gauge`; defined `source x = expr;` classified `ModelDefined`; `input field`/`input
   value` slots; `SourceLocator { module, span }`.
3. SV1-A (E7): `DerivativeRequest` producer from `observable`/`objective` declarations with
   `SymbolId`/`ExprId` links and design variables bound to case property slots; inverse-Poisson
   corpus test and a public API Sinbad can call from a compiled case.
4. SC-W1 (§2, §3): scoped by-reference imports (`GlobalDeclId`, `use` aliases/selective lists,
   `pub`), `model` as the implicit one-instance system, `system`/`instance`/`bind`, the system
   arena (`SysVarId`/`SysResId`, `OriginMap`), `scientia-system/1`, `scientia-operator-system/2`,
   kernel-level `Composed` bind chains, ordered module closure; ARCHITECTURE §11 tests 1–5
   (Scientia-local parts). `connector`/`port`/`Open` stay SC-W2.

Deviations from `sinbad/ARCHITECTURE.md` are recorded in the batch/package sections above with
their reasons; the coordinator folds them into GX-CONTRACTS C12. Scientia remains execution-free.
