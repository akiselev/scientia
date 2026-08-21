# Scientia status

Updated: 2026-08-21

Branch: `master`

Milestone: R1 standalone scientific compiler extraction

## Role

Scientia owns `.res` source and scientific/form semantics. It parses and resolves modules,
maintains one canonical model, performs semantic and structural analysis, derives variational and
local mathematical artifacts, and lowers local work into Malleus-owned structured kernel types.

Quantitas owns dimensions, quantity kinds, units, and registries. Resolvent owns consumer-neutral
exact algebra and symbolic differentiation. Malleus owns local kernel IR and execution. Finitum
owns concrete discretization/global operators. Krasis owns coupled runtime state. Solverang owns
numerical algorithms. Sinbad owns product orchestration.

## Implemented

- Recovering `.res` parser, canonical formatter, byte-precise expression/reference spans,
  deterministic module resolution, and presentation-invariant source digest.
- One typed `SemanticModel` arena with stable domain, region, symbol, expression, and declaration
  identities; resolved roles and references; shapes and axes; Quantitas dimensions/kinds/units;
  domain frames; typed declaration payloads; and presentation-invariant semantic digest. The
  typed-declaration wire shape is `scientia-semantic/3`; variational forms are
  `scientia-variational-form/4` and record the physical source of generated test spaces.
- Stable structured diagnostics with exact spans for malformed syntax, imports, domains, names,
  units, quantity kinds, roles, shapes, axes, dimensions, and frames.
- `compile_semantics` is the complete FC1 library boundary. CLI `check`, `inspect`, `freeze`, and
  all analysis/form commands require it; `elaborate` emits the typed arena directly.
- Direct Quantitas quantity/unit types and validation; no internal quantity representation exists.
- Property tables/expressions, derivative contracts, constitutive semantics, coupling graphs,
  time/state semantics, and evidence profiles.
- Scientific property differentiation delegates exact algebra to Resolvent and explicitly refuses
  unsupported scientific expression forms; Scientia retains scientific spans and diagnostics.
- Structural incidence, matching, SCC/BLT, tearing, alias, and DAE planning projected directly
  from `ScientificModel`; incidence follows model-defined value, property, and constitutive chains
  to the same transitive field dependencies reported by coupling analysis.
- `VariationalForm` compilation consumes only `SemanticModule`; arguments, captures, measures, and
  integrands retain arena IDs and typed roles instead of source-name keys.
- Strong equations derive into residual forms with generated typed test arguments, physical-field
  captures, space-aware integration by parts, and stable capability refusals when test selection,
  boundary partitions, curl orientation, or Robin flux meaning is not available.
- Cell, exterior/interior-facet, interface, and point sides are explicit. Differential,
  contraction, tensor/facet trace, jump, average, normal-component, and conjugation operations are
  typed semantic nodes; ambiguous two-sided field access is rejected.
- Form receipts record residualization, test multiplication, integration by parts, boundary-data
  substitution, an explicit complex convention, typed boundary-partition/test-trace assumptions,
  and retained/substituted/eliminated boundary terms. Multiple flux terms cannot consume the same
  Neumann datum without explicit correspondence.
- Mixed strong equations select test spaces from typed result shape and differentiated dependency
  structure. Gradient terms integrate against H(div) divergence, curl terms integrate against
  H(curl) curl with oriented tangential traces, and deferred source/flux shapes are refined from
  the selected equation row without changing the canonical semantic arena.
- The deterministic form interpreter evaluates scalar, complex, vector/tensor, differential,
  contraction, side/trace, indexing, and common scalar operations from caller-supplied point data.
  Caller-supplied weights are accumulated without taking ownership of quadrature traversal.
  `required_evaluations` exposes the expression/evaluation bindings needed by an integral.
- Realization-neutral `LocalFormProgram` factorization preserves test/trial/field/coefficient/value
  roles and required value/gradient/time-derivative/trace evaluations.
- Form, local-program, and kernel-lowering artifacts have schema-versioned, span-independent
  digests and receipts linking each artifact to its parent.
- Mesh-free `FormRequirements` inference covers H1, L2, product/mixed, Hcurl, Hdiv, DG, and trace
  spaces; abstract element family/order/shape; H1/L2/broken and Piola pullbacks; tangential,
  normal, and two-sided orientation; basis evaluation sites; geometry preprocessing; conservative
  quadrature intent; essential constraints; and exterior-boundary partition requirements.
- Requirement inference expands model-defined value/property/constitutive dependency chains to
  recover hidden physical-field spaces and derivative evaluations. Inputs explicitly distinguish
  basis-backed values, external values, model-defined values/properties, and model-defined
  constitutive preprocessing, so non-space symbols are never presented as basis data.
- Integral expression signatures are normalized without arena IDs or spans and grouped only when
  measure/domain/region, output type, inputs/evaluations, geometry, quadrature, and integrand all
  match. The mathematical requirement digest is invariant to integral, domain, and field
  declaration order, while its receipt retains the exact parent form digest.
- Stable `REQ_*` refusals cover non-scalar axes, cross-domain measures, invalid region kinds,
  incompatible space continuity/value shapes/differentials/normal traces, and incompatible
  essential boundary data.
- Scalar point-QFunction lowering returns Malleus IR with a lowering receipt. Finitum owns
  quadrature selection/traversal; the empty Malleus iteration domain means one point invocation.
- `factor_operator` consumes a typed form and its digest-linked `FormRequirements`, retaining one
  indexed `TensorProgram` per integral with explicit shapes, cell/facet sides, real scalar
  semantics, free component axes, and canonical sum-reduction axes.
- Symbolic differentiation with respect to test evaluations produces basis-dual
  `QFunctionProgram` outputs. Operator artifacts factor gather/restriction, basis actions,
  external/model-defined preprocessing, geometry, point functions, quadrature weight, basis
  transpose, scatter, and essential-constraint stages without concrete mesh or DOF types.
- Symbolic directional differentiation produces JVP QFunctions with receipts naming the primal,
  active and frozen inputs, runtime evaluation point, complex convention, stateless semantics,
  construction method, and algebraic evidence.
- Tensor input shapes are resolved by the complete symbol/evaluation/site/mapping binding rather
  than evaluation declaration order. Invalid differential base shapes and active non-basis
  inputs are refused before artifact generation and again at the interpreter boundary.
- Deterministic QFunction and caller-supplied element-factorization interpreters implement the FC4
  reference semantics. An independent P1 triangle fixture validates generated Poisson element
  residuals and JVPs against its analytic tensor and a directional finite difference.
- `lower_operator_kernels` lowers each accepted FC4 QFunction output into a digest-linked Malleus
  `StructuredModule` containing primal, state-JVP, state-VJP, and frozen-input parameter-JVP
  kernels. Tensor free/reduction axes become fixed structured iteration domains, tensor input
  indices become affine maps, and reduction outputs carry explicit additive effects.
- Each FC5 bundle records QFunction input-to-operand bindings, derivative modes/purposes/evidence,
  numeric policy, source factorization/primal/symbolic-JVP digests, integral/output identity, and
  the exact kernel index for each product. Malformed shapes, unsupported non-enclosing reduction
  structure, precision mismatch, and broken derivative receipt chains are refused.
- One logical QFunction input may bind multiple Malleus operands when tensor contraction accesses
  it through distinct affine index vectors; derivative contracts bind every access while receipts
  retain each logical input once.
- FC5 bundles retain their direct typed handoff into Finitum, and FC11 now also gives complete
  `StructuredModule`, structured bundle, `MethodProgram`, and `OperatorSystem` artifacts stable
  Serde round trips. Decoded Malleus modules are structurally revalidated before execution, while
  executable schedules remain rebuilt downstream data.
- All four products execute with Malleus's deterministic interpreter. The FC5 Poisson gate covers
  three triangle geometries against independent analytic element tensors, Malleus-vs-FC4 JVP
  agreement, directional finite differences, a VJP adjoint dot product, and property/source
  parameter derivatives. No named-physics operation exists in the lowering or Malleus.
- `OperatorSystem` compiles derived equations or authored forms through requirements,
  factorization, and complete structured-kernel bundles. Typed test-space receipts own block rows,
  active QFunction bindings own block columns, and the system digest covers every artifact link.
  Repository-local FC8 gates compile elasticity, Stokes, Darcy, split-complex Maxwell, and a
  two-sided DG facet form through the same contract.
- Five digest-linked `MethodProgram` compilers consume the existing typed semantic arena for
  conservation-law/FV, structured-stencil/FD, network DAE, particle, and boundary-integral
  families. They retain typed source identities, Quantitas-backed state types, and expression
  arenas while their receipts explicitly bypass `VariationalForm`.
- FV numerical flux and FD stencil requests lower to validated affine Malleus point kernels;
  concrete topology, matrices, pairs, and boundary quadrature remain downstream-owned. Stable
  `METHOD_*` errors refuse incompatible domains, equation structure, shapes, and kernel requests.
- One `scientia` CLI for check, format, parse, inspect, freeze, explain, coupling, structural
  analysis, forms, requirements, and operators. Multi-model modules require explicit selection;
  form/equation commands accept `Model:item` and model-wide commands accept `Model`.

## Removed

- The pre-form expression/context/system pipeline and its RSL, LaTeX, and Lean frontends.
- Form/discrete/operator/backend types that duplicated the new compiler direction.
- Reference FEM implementations and scientific bridge layers.
- Old comparison tooling, runtime plans, diagnostic logs, and the internal quantity crate.
- The exact-CAS roadmap/research/ADR corpus that no longer described this product.

Git history is the archive. None of the removed implementation is an acceptance oracle.

## Validation

Verified locally on 2026-08-21:

- `cargo fmt --all -- --check` -- passed.
- `cargo check --locked --workspace --all-targets` -- passed.
- `cargo clippy --locked --workspace --all-targets -- -D warnings` -- passed.
- `cargo test --locked --workspace --all-targets` -- passed: 84 tests, 0 failed.
- `RUSTDOCFLAGS='-D warnings' cargo doc --locked --workspace --no-deps` -- passed.
- `cargo test --locked --workspace --doc` -- passed.
- `git diff --check` -- passed.
- `cargo run --quiet --bin scientia -- check` over all 50 Sinbad corpus models -- passed: 50 of
  50 parsed and elaborated.
- `derive-form` passed for Poisson, transient diffusion, nonlinear heat, linear elasticity, and
  both Stokes equations; the same set is covered by the FC2 integration gate.
- `derive-requirements` passed for Poisson, transient diffusion, nonlinear heat, linear
  elasticity, and both Stokes equations from the Sinbad corpus; the same set is covered by the FC3
  integration gate.
- Stokes momentum requirements include velocity H1/order-2 space and basis-backed
  `symmetric_gradient`, with viscosity/strain/stress typed as model-defined preprocessing.
- The electrothermal corpus model's coupling graph includes Joule-source and constitutive
  dependencies, and structural analysis returns a nonsingular coupled 2x2 schedule.
- A three-model CLI fixture passes qualified form, requirement, coupling, structural, and explain
  selection plus authored/derived operator selection, and refuses ambiguous unqualified item
  selection.
- `derive-operator` passes on Sinbad's Poisson corpus model and emits digest-linked tensor,
  primal-QFunction, JVP-QFunction, and operator-factorization artifacts.
- The repository-local FC4 Poisson gate passes an independent analytic P1 triangle residual and
  element-matrix JVP comparison plus a directional finite-difference JVP check.
- The repository-local FC5 gate validates complete four-kernel Malleus modules and executes
  generated Poisson primal/JVP/VJP/parameter products across three element geometries; analytic
  tensors, FC4 symbolic JVPs, finite differences, and an adjoint identity agree.
- The repository-local FC8 gate compiles elasticity, Stokes, Darcy, and split-complex Maxwell
  operator systems plus a two-sided DG facet form into validated Malleus bundles. Stokes exposes
  three nonzero block coordinates, Darcy exposes H(div)-L2 rows, Maxwell exposes four coupled
  split-complex coordinates, and minus/plus trace inputs remain explicit.
- The repository-local FC10 gate compiles all five method families from independent local source
  fixtures, proves distinct artifact identities and nonvariational receipts, executes FV/FD affine
  kernels with Malleus, and checks domain/stencil refusals.
- FC11 round-trip gates serialize and deserialize a complete FV method program, a Poisson
  primal/JVP/VJP/parameter bundle, and a mixed Stokes operator system, then revalidate every nested
  Malleus module.
- Scientia tests contain no compile-time or runtime path into Sinbad's product corpus; standalone
  validation uses only repository-local fixtures plus the declared Quantitas/Malleus dependencies.
- `cargo run --quiet --bin scientia -- check examples/nonlinear_heat.res` -- passed.
- `cargo run --quiet --bin scientia -- structural examples/nonlinear_heat.res` -- passed with
  one explicit structural block.

## Cross-repository contract

- Quantitas path: `../quantitas`, validated at
  `734d78cd6ff516afee54201bc70cd59fd34e67e3`; API types used directly include `Dimension`,
  `Quantity`, `QuantityLiteral`, `QuantityKindId`, `UnitId`, and `UnitRegistry`.
- Malleus path: `../malleus`, validated at
  `09e27a6a23a6a5eab6f881ac0bec9db23046d58e`; Scientia constructs Malleus modules, operands,
  affine maps, expressions, derivative requests, statements, and numeric policies directly.
- Public downstream sequence: `compile_variational_form`/`derive_variational_form` ->
  `infer_form_requirements` -> `factor_operator` -> `lower_operator_kernels`. The existing narrow
  scalar path remains `factor_local_integral` -> `lower_local_program`, with
  `LoweredKernel { kernel, receipt }` rather than a bare `StructuredKernel`.
- Sibling downstream sequences start with the family-specific `compile_*_method` functions and
  produce `MethodProgram` directly; Finitum consumes that artifact without form requirements or
  FEM operator factorization.
- Finitum maps `LocalIterationContract::QuadraturePoint` across selected elements and quadrature
  points in its landed FC6 reference realization. Any later fixed-axis batching remains
  realization-owned and must preserve point-QFunction semantics; see `ITERATION-OWNERSHIP.md`.

## Next compiler work

1. Keep optimized topology traversal realization-owned and preserve local-kernel semantics.
2. Evolve serialized schemas only with explicit versioning and receipt-chain validation.
