# Scientia scientific compiler

## Purpose

Scientia turns authored scientific meaning into inspectable mathematical artifacts. It stops at
the local structured-kernel boundary. Concrete discretization, coupled runtime state, and solver
policy are separate concerns with separate owners.

The intended artifact flow is:

```text
.res source
  -> ScientificModule / ScientificModel
  -> SemanticModule / SemanticModel arena
  -> formulation and structural analysis
  -> VariationalForm
  -> indexed tensor and local-operator factorization
  -> Malleus StructuredKernel
```

Each arrow is a compiler pass with diagnostics, source provenance, and an evidence receipt when it
changes mathematical form. These are successive artifacts, not interchangeable frontends or
temporary representations.

Planned, not landed: the SC composition layer designed in `sinbad/ARCHITECTURE.md` inserts a
`ScientificSystem` between the per-model semantic arena and the operator artifacts — instances of
imported models, `bind`/`connect`, connector ports as open boundary unknowns, system-level ids with
an origin map — while every per-model FC2–FC5 artifact stays as it is and is reused by digest. A
bare `model` becomes the implicit one-instance system, so there is one compile path.

## Repository boundaries

### Quantitas

Owns exact dimensions, rational exponents, quantity-kind identity, units, registry provenance, and
canonical quantities. `.res` declarations carry Quantitas types directly. Scientia does not wrap
or mirror them.

### Scientia

Owns:

- the `.res` grammar, source spans, diagnostics, formatting, module resolution, and semantic hash;
- fields, equations, authored forms, measures, properties, constitutive laws, events, observables,
  and verification annotations;
- dimension/kind checking using Quantitas;
- formulation derivation and variational semantics;
- dependency/coupling, incidence, alias, matching, SCC/BLT, tearing, and DAE analysis;
- tensor/QFunction/local-operator factorization; and
- evidence for symbolic and semantic transformations.

Scientia does not own meshes, global degrees of freedom, finite-element tables, assembly, time
integration, nonlinear solves, device code generation, or product orchestration.

### Malleus

Owns the schedule-independent local kernel IR, validation, derivative products, schedule choices,
and executable backends. Scientia constructs Malleus types directly; Malleus never imports
Scientia and has no scientific vocabulary.

### Downstream

Finitum binds form requirements and kernels to concrete meshes, spaces, basis data, quadrature,
constraints, and global operators. Krasis combines those operators with transactional coupled
state. Methodus consumes physics-neutral residual/operator traits. Sinbad selects cases, runs,
policies, and artifacts.

## Source and canonical semantic model

`ScientificModule`, its contained source models, and `Expr` retain authored syntax and provenance.
They are parser output, not a second type system. `compile_semantics` resolves that syntax into one
`SemanticModule` / `SemanticModel` arena. Every semantic expression has an arena identity, precise
span, resolved symbol identity, value shape and axes, a Quantitas `Dimension` and
`QuantityKindId` when constrained, a domain frame, and a distinct scientific role.

Unspecified external scientific-function signatures remain explicitly `deferred`. Known facts are
still enforced around them; no scalar, unit, axis, or frame meaning is invented. Stable diagnostic
codes and byte spans cover parsing, module imports, names, domains, units, quantity kinds, roles,
axes, shapes, dimensions, and frames. Semantic arena digests exclude presentation spans.

Authored forms and derived strong equations compile to `VariationalForm`, which retains canonical
expressions and adds only form-specific organization:

- the selected semantic declaration and parent semantic digest;
- test/trial arguments and captures keyed by `SymbolId`, with roles, types, spaces, and domains;
- cell, exterior/interior-facet, interface, and point measures keyed by `DomainId` or `RegionId`;
- integrands keyed by `ExprId`; and
- an artifact digest and receipt recording the source declaration and transformation history.

Strong-form derivation residualizes equations, selects or accepts an explicit physical field for
the test space, applies integration by parts only when that space supports the resulting
derivative, and records boundary terms as retained, substituted, or eliminated by an essential
condition. The receipt explicitly assumes that its resolved exterior regions partition the domain
boundary; Finitum must discharge that assumption against topology. A Neumann value may substitute
only one integrated flux per region and field until the language can express per-term
correspondence. Ambiguous test spaces or fluxes, missing boundary partitions, invalid derived
differentials, curl orientation requirements, and unsupported Robin flux laws return stable
capability errors rather than guessed forms.

Differential, contraction, tensor/facet trace, jump, average, normal-component, and conjugation
operations are typed semantic nodes rather than string opcodes. Two-sided facets require explicit
minus/plus restrictions (or jump/average). The deterministic form interpreter evaluates these
nodes over caller-supplied point values, derivatives, traces, normals, and weights; constructing
quadrature remains a Finitum responsibility. `required_evaluations` exposes the expression and
evaluation-site bindings without requiring consumers to traverse the semantic arena. Form receipts
record an explicit-conjugation-only convention; derivation inserts no implicit complex conjugate.

## FC3 requirements boundary

`infer_form_requirements` produces a mesh-free `FormRequirements` artifact before any concrete
realization. It records per-binding H1, L2, Hcurl, Hdiv, DG, and trace needs; single or product
argument-space composition; abstract element family/order/value shape; H1/L2/broken and Piola
pullbacks; tangential, normal, and two-sided orientation; basis evaluation sites; geometry
preprocessing; conservative quadrature intent; essential constraints; and boundary-partition
obligations.

Requirement inference follows model-defined value, property, and constitutive expressions to
their physical-field dependencies. A Stokes stress chain therefore emits the velocity space and
`sym_grad` basis evaluation rather than treating stress as an opaque coefficient. Inputs also say
whether they are basis-backed, externally supplied, or computed by a model-defined value,
property, or constitutive expression, so FC4 cannot confuse preprocessing outputs with basis data.

Integral expressions are normalized without arena IDs or source spans and grouped only when the
complete kernel signature matches: measure/domain/region, output type, input evaluations,
geometry, quadrature intent, and typed integrand. The mathematical requirement digest is invariant
to integral, domain, and field declaration ordering, while a separate receipt retains the exact
parent form digest. Stable `REQ_*` diagnostics refuse non-scalar axes, cross-domain measures,
incompatible continuity/differential/trace spaces, region kinds, and essential boundary data.

This artifact does not select reference cells, meshes, basis tables, quadrature rules, DOFs, or
assembly strategy. Those choices remain Finitum responsibilities.

Structural incidence uses the same transitive field-dependency meaning as coupling analysis;
indirection through model-defined values, properties, and constitutive laws cannot make an
otherwise matched coupled system appear structurally singular.

## FC4 tensor and operator boundary

`factor_operator` consumes a form and its digest-linked `FormRequirements`. Each integral retains
an indexed `TensorProgram` for the scalar integrand. Shapes, cell/facet side, real scalar
semantics, free component axes, and canonical sum-reduction axes are explicit. Differentiating
the integrand with respect to each test evaluation creates basis-dual `QFunctionProgram` outputs;
the test basis itself is therefore not smuggled into the point function.

The resulting `OperatorFactorization` orders restriction/gather, basis evaluation, model-defined
or external preprocessing, geometry, QFunction, quadrature weighting, basis transpose, scatter,
and essential-constraint stages without choosing a mesh, basis table, quadrature rule, or global
map. JVP point programs are obtained by symbolic directional differentiation of active
trial/unknown/state evaluations. Their receipts identify the primal, active and frozen inputs,
runtime evaluation point, complex convention, stateless transaction meaning, construction method,
and algebraic evidence.

The deterministic reference interpreters execute indexed QFunctions and caller-supplied element
factorizations. The FC4 gate uses a repository-local P1 triangle fixture to compare generated
Poisson residuals with the independent analytic element tensor and generated JVPs with both that
tensor action and directional finite differences. No Poisson operation exists in the compiler or
interpreter.

## Malleus boundary

`factor_local_integral` produces a realization-neutral `LocalFormProgram`; `lower_local_program`
is the first narrow implementation of the Malleus boundary. It accepts scalar pointwise arithmetic
and common scalar functions, then constructs `malleus::StructuredKernel` directly. Local inputs
retain test-basis, trial-basis, physical-field, parameter, constant, source, property, and
constitutive-law roles plus required value/gradient/time-derivative/trace evaluations. It rejects
`grad`, `div`, `curl`, `dot`, indexed tensors, and vector expressions until the basis/tensor pass
has expanded them.

The local artifact is explicitly a one-quadrature-point QFunction. Finitum owns quadrature choice
and traversal; Malleus's empty iteration domain therefore means one invocation, never an omitted
symbolic loop. See [ITERATION-OWNERSHIP.md](ITERATION-OWNERSHIP.md).

FC5 lowers only FC4's local indexed numerical regions to Malleus structured modules. Each indexed
output becomes a validated primal, state-JVP, state-VJP, and frozen-input parameter-JVP bundle.
Free/reduction axes, affine maps, dense layouts, access effects, finite-precision policy, and
source/derivative receipts are explicit. Malleus constructs derivatives as structured IR-to-IR
passes and its deterministic interpreter executes every product before any optimized backend.

Each distinct affine access to a logical QFunction input receives its own read operand. Derivative
contracts bind all access operands, while receipts retain each logical input once. Direct typed
handoff remains the execution boundary. FC11 additionally serializes and deserializes complete
modules and bundles for product archives; decoded modules must pass Malleus validation before use.

## FC8 mixed and facet systems

`OperatorSystem` compiles selected derived equations or authored forms through the ordinary
requirements, tensor factorization, and structured-kernel path. Generated test-space provenance
owns each block row and active typed QFunction bindings own its columns; the system digest covers
every child receipt. Test-field selection uses result shape and differentiated dependency depth,
so Stokes, Darcy, and split-complex Maxwell rows are selected without source-name heuristics.

Space-aware derivation now integrates gradients against H(div) divergence and curls against
H(curl) curl with oriented tangential traces. Facet sites and natural value/tangential/normal
trace mappings remain explicit through QFunction inputs. The repository-local gate compiles
elasticity, Stokes, Darcy, Maxwell, and a two-sided DG form into complete Malleus bundles. Concrete
facets, Piola maps, compatible DOFs, exact-sequence checks, and condensation remain Finitum-owned.

No named-physics opcode is permitted. A heat, elasticity, Maxwell, or flow form must decompose into
general mathematical operations and explicit data dependencies.

## FC10 sibling method programs

Five method compilers consume the same typed `SemanticModule` as the form path and produce
digest-linked `MethodProgram` artifacts for conservation-law finite volume, structured-stencil
finite differences, network DAEs, particle systems, and boundary integrals. Their receipts retain
typed declarations, Quantitas-backed state types, domains, regions, and the source expression
arena while explicitly recording that no variational form was constructed. Stable `METHOD_*`
refusals reject incompatible domains, equation structure, state shapes, or local-kernel requests.

FV numerical fluxes and FD stencils lower to validated affine Malleus point kernels. Concrete
cells, stencil neighborhoods, network matrices, particle pairs, boundary quadrature, and global
actions remain Finitum-owned. The repository-local gate uses five independent source fixtures and
executes the FV/FD kernels with Malleus reference semantics.

## Evidence and validation

The compiler treats these as different claims:

- source accepted and resolved;
- semantic model internally valid;
- quantity and kind constraints satisfied;
- structural schedule valid;
- mathematical transformation justified;
- local kernel structurally valid;
- numerical realization verified downstream.

An earlier implementation is not an oracle. Compiler tests use grammar invariants, independent
small exhaustive oracles, analytical identities, manufactured solutions, and external fixtures as
appropriate. Removed code remains available in Git history but has no active runtime or acceptance
role.

## Immediate work

FC11 complete-artifact serialization is landed for method programs, structured kernel bundles,
and operator systems. Keep realization choices downstream and do not merge sibling programs into
variational-form types; future wire-schema changes must be explicit and retain receipt validation.
