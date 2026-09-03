//! `scientia-operator-structure/1` (GX-A4, contract C5.4): solver-relevant structural facts
//! about a compiled operator, derived read-only from existing FC3/FC4/FC8 artifacts. Nothing
//! here executes a solver, chooses a linear algebra strategy, or inspects concrete mesh/DOF
//! data; every fact is either decided structurally from the typed tensor program or explicitly
//! `Unknown`/absent when it cannot be decided without guessing.

use crate::formulation::{FormArgumentRole, FormCaptureRole, VariationalForm};
use crate::id::{Digest, span_independent_digest};
use crate::requirements::{DerivativeEvaluation, FormRequirements, InputSourceRequirement};
use crate::scientific::{DerivativeContract, FieldRole, TimeRole, ValueShape};
use crate::semantic::{ExprId, SemanticExpr, SemanticExprKind, SemanticProvider, SymbolId};
use crate::structural::IndexReductionPlan;
use crate::system::OperatorSystem;
use crate::tensor::{
    OperatorFactorization, TensorAxis, TensorAxisId, TensorBinaryOp, TensorInputId,
    TensorProgramInputRole, TensorScalarExpr, TensorUnaryOp, collect_direct_symbols,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

/// `/2` (SC, `sinbad/ARCHITECTURE.md` §7, contract C12): additive `block_symmetry`,
/// `transpose_relation`, and `sign_gauge` over the `/1` (C5.4) fields. Keys are the per-model
/// row `SymbolId`s of `blocks` until SC-W1's system-level `SysResId` re-keys them.
pub const OPERATOR_STRUCTURE_SCHEMA: &str = "scientia-operator-structure/2";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "spec", rename_all = "snake_case")]
pub enum Linearity {
    Linear,
    Nonlinear { active: Vec<SymbolId> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormSymmetry {
    Symmetric,
    Nonsymmetric,
    Unknown,
}

/// Contract C5.4 lists this enum verbatim, including `Convective`; the task's own prose summary
/// dropped it, but the frozen contract text is authoritative (§C5.4 "implement as written").
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockClass {
    Mass,
    Diffusive,
    Convective,
    Constraint,
    Coupling,
    Reaction,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BlockStructure {
    pub row: SymbolId,
    pub column: SymbolId,
    pub present: bool,
    pub class: BlockClass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NullspaceKind {
    Constant,
    RigidBody { dimension: u8 },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NullspaceCandidate {
    pub field: SymbolId,
    pub kind: NullspaceKind,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PropertyTangent {
    Inlined,
    External,
    Frozen,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropertyDependence {
    pub property: SymbolId,
    pub depends_on: Vec<SymbolId>,
    pub tangent: PropertyTangent,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeStructure {
    pub transient: bool,
    pub roles: Vec<(SymbolId, TimeRole)>,
    pub dae_index_lower_bound: Option<u8>,
}

/// `A_ij ≈ σ_ij · A_jiᵀ` decided structurally (§7): the two off-diagonal kernels are compared
/// after exchanging test/active roles, relabeling shared coefficient inputs by binding, and
/// splitting an overall sign; `None` when the relation is undecidable (kernels with several
/// integrals, non-matching evaluations, or structure beyond sign-normalized equivalence).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TransposeRelation {
    pub row: SymbolId,
    pub column: SymbolId,
    pub sigma: Option<i8>,
}

/// One signed edge of the row graph: `sigma` is the transpose relation between the two rows'
/// off-diagonal blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SignedEdge {
    pub row: SymbolId,
    pub column: SymbolId,
    pub sigma: i8,
}

/// Proof that the signed row graph is balanced: signs were propagated from each component's
/// smallest row (`+1`) along `spanning_forest`, and every edge in `edges` then satisfies
/// `sign[row] · sign[column] == sigma`, so multiplying each row by its sign makes every
/// off-diagonal pair an exact transpose pair. Structure proposes; Finitum's `prove_symmetry`
/// remains the runtime gate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedGraphBalance {
    pub edges: Vec<SignedEdge>,
    pub spanning_forest: Vec<SignedEdge>,
}

/// The compiler-owned residual gauge (§7): a `±1` row orientation per block row under which the
/// whole system is structurally symmetric. Replaces case-data `[system].equation_sign`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignGauge {
    /// Sorted by row.
    pub signs: Vec<(SymbolId, i8)>,
    pub proof: SignedGraphBalance,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperatorStructure {
    pub schema: String,
    pub model: String,
    pub source_digest: Digest,
    pub trial_linearity: Linearity,
    /// The unsigned authored form/system's symmetry (C5.4): decided for a single linear block,
    /// `Unknown` for every multi-block system (see `sign_gauge` for the gauged claim).
    pub form_symmetry: FormSymmetry,
    pub blocks: Vec<BlockStructure>,
    pub saddle_point: bool,
    pub nullspace_candidates: Vec<NullspaceCandidate>,
    pub property_dependence: Vec<PropertyDependence>,
    pub time: TimeStructure,
    /// `/2`: symmetry of each *present* diagonal block, keyed by row, sorted; decided by the
    /// evaluation-paired test/active exchange of every integral linear in its active inputs.
    pub block_symmetry: Vec<(SymbolId, FormSymmetry)>,
    /// `/2`: one entry per ordered off-diagonal pair `(row, column)` whose block and whose
    /// mirror block are both present; sorted.
    pub transpose_relation: Vec<TransposeRelation>,
    /// `/2`: present iff every present diagonal block is `Symmetric`, no diagonal block is
    /// `Convective`, every present off-diagonal pair is two-sided with a decided `sigma`, and
    /// the signed row graph is balanced.
    pub sign_gauge: Option<SignGauge>,
    /// `/2`: exactly when `sign_gauge` is `None`, why (the first failing condition).
    pub sign_gauge_reason: Option<String>,
    pub identity: Digest,
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("{code}: {message}")]
pub struct StructureError {
    pub code: String,
    pub message: String,
}

fn structure_error(code: &str, message: impl Into<String>) -> StructureError {
    StructureError {
        code: code.into(),
        message: message.into(),
    }
}

#[derive(Serialize)]
struct StructureIdentity<'a> {
    schema: &'static str,
    model: &'a str,
    source_digest: &'a Digest,
    trial_linearity: &'a Linearity,
    form_symmetry: FormSymmetry,
    blocks: &'a [BlockStructure],
    saddle_point: bool,
    nullspace_candidates: &'a [NullspaceCandidate],
    property_dependence: &'a [PropertyDependence],
    time: &'a TimeStructure,
    block_symmetry: &'a [(SymbolId, FormSymmetry)],
    transpose_relation: &'a [TransposeRelation],
    sign_gauge: &'a Option<SignGauge>,
    sign_gauge_reason: &'a Option<String>,
}

struct BlockView<'a> {
    row: SymbolId,
    form: &'a VariationalForm,
    requirements: &'a FormRequirements,
    factorization: &'a OperatorFactorization,
}

/// Derive `OperatorStructure` for a single derived form/factorization pair (contract C5.4).
/// `dae_plan` is caller-supplied because Pantelides planning operates on the original
/// `ScientificModel` (structural.rs), a different source than the FC4 tensor artifacts this
/// function otherwise reads; pass `None` when no plan is available and
/// `dae_index_lower_bound` is left absent rather than guessed.
pub fn derive_operator_structure(
    form: &VariationalForm,
    requirements: &FormRequirements,
    factorization: &OperatorFactorization,
    dae_plan: Option<&IndexReductionPlan>,
) -> Result<OperatorStructure, StructureError> {
    let row = block_row(form)?;
    let block = BlockView {
        row,
        form,
        requirements,
        factorization,
    };
    build_structure(
        &form.model,
        factorization.artifact_digest.clone(),
        std::slice::from_ref(&block),
        dae_plan,
    )
}

/// Derive `OperatorStructure` for a compiled `OperatorSystem` (contract C5.4): block coordinates
/// come from the system's own row/column receipts rather than display names.
pub fn derive_operator_structure_for_system(
    system: &OperatorSystem,
    dae_plan: Option<&IndexReductionPlan>,
) -> Result<OperatorStructure, StructureError> {
    if system.blocks.is_empty() {
        return Err(structure_error(
            "STRUCTURE_UNDECIDABLE",
            format!("operator system `{}` has no blocks", system.model),
        ));
    }
    let blocks = system
        .blocks
        .iter()
        .map(|block| BlockView {
            row: block.row,
            form: &block.form,
            requirements: &block.requirements,
            factorization: &block.factorization,
        })
        .collect::<Vec<_>>();
    build_structure(
        &system.model,
        system.artifact_digest.clone(),
        &blocks,
        dae_plan,
    )
}

fn block_row(form: &VariationalForm) -> Result<SymbolId, StructureError> {
    if let Some(symbol) = form.receipt.test_space_source {
        return Ok(symbol);
    }
    let tests = form
        .arguments
        .iter()
        .filter(|argument| argument.role == FormArgumentRole::Test)
        .map(|argument| argument.symbol)
        .collect::<Vec<_>>();
    match tests.as_slice() {
        [symbol] => Ok(*symbol),
        _ => Err(structure_error(
            "STRUCTURE_UNDECIDABLE",
            format!(
                "form `{}::{}` has no unique test field to serve as a block row",
                form.model, form.name
            ),
        )),
    }
}

fn active_symbols(form: &VariationalForm) -> BTreeSet<SymbolId> {
    let mut symbols = BTreeSet::new();
    for argument in &form.arguments {
        if argument.role == FormArgumentRole::Trial {
            symbols.insert(argument.symbol);
        }
    }
    for capture in &form.captures {
        if matches!(
            capture.role,
            FormCaptureRole::PhysicalField(
                FieldRole::Unknown | FieldRole::State | FieldRole::Trial
            )
        ) {
            symbols.insert(capture.symbol);
        }
    }
    symbols
}

fn collect_provider_differentiabilities(
    expressions: &[SemanticExpr],
    providers: &[SemanticProvider],
    id: ExprId,
    out: &mut Vec<DerivativeContract>,
) {
    let Some(expression) = expressions.get(id.index()) else {
        return;
    };
    match &expression.kind {
        SemanticExprKind::ProviderCall { provider, args } => {
            if let Some(found) = providers.iter().find(|candidate| candidate.id == *provider) {
                out.push(found.differentiability.clone());
            }
            for arg in args {
                collect_provider_differentiabilities(expressions, providers, *arg, out);
            }
        }
        SemanticExprKind::Unary { arg, .. } | SemanticExprKind::Differential { arg, .. } => {
            collect_provider_differentiabilities(expressions, providers, *arg, out);
        }
        SemanticExprKind::Binary { lhs, rhs, .. } => {
            collect_provider_differentiabilities(expressions, providers, *lhs, out);
            collect_provider_differentiabilities(expressions, providers, *rhs, out);
        }
        SemanticExprKind::Call { args, .. } | SemanticExprKind::Vector { elements: args } => {
            for arg in args {
                collect_provider_differentiabilities(expressions, providers, *arg, out);
            }
        }
        _ => {}
    }
}

fn tangent_rank(tangent: PropertyTangent) -> u8 {
    match tangent {
        PropertyTangent::Inlined => 2,
        PropertyTangent::External => 1,
        PropertyTangent::Frozen => 0,
    }
}

fn merge_tangent(a: PropertyTangent, b: PropertyTangent) -> PropertyTangent {
    if tangent_rank(a) >= tangent_rank(b) {
        a
    } else {
        b
    }
}

fn derive_property_dependence(blocks: &[BlockView]) -> BTreeMap<SymbolId, PropertyDependence> {
    let mut properties = BTreeMap::new();
    for block in blocks {
        let block_active = active_symbols(block.form);
        for integral in &block.factorization.integrals {
            for input in &integral.primal.inputs {
                let InputSourceRequirement::ModelDefinedProperty { definition } = input.source
                else {
                    continue;
                };
                let property_symbol = input.binding.symbol;
                let mut direct = BTreeSet::new();
                let _ = collect_direct_symbols(&block.form.expressions, definition, &mut direct);
                let depends_on = direct
                    .intersection(&block_active)
                    .copied()
                    .collect::<BTreeSet<_>>();
                let mut differentiabilities = Vec::new();
                collect_provider_differentiabilities(
                    &block.form.expressions,
                    &block.form.providers,
                    definition,
                    &mut differentiabilities,
                );
                let tangent = if depends_on.is_empty() {
                    PropertyTangent::Frozen
                } else if !differentiabilities.is_empty()
                    && differentiabilities.iter().all(|contract| {
                        matches!(
                            contract,
                            DerivativeContract::Symbolic | DerivativeContract::Automatic
                        )
                    })
                {
                    PropertyTangent::Inlined
                } else if !differentiabilities.is_empty() {
                    PropertyTangent::External
                } else {
                    PropertyTangent::Frozen
                };
                let entry =
                    properties
                        .entry(property_symbol)
                        .or_insert_with(|| PropertyDependence {
                            property: property_symbol,
                            depends_on: Vec::new(),
                            tangent,
                        });
                let mut merged = entry.depends_on.iter().copied().collect::<BTreeSet<_>>();
                merged.extend(depends_on.iter().copied());
                entry.depends_on = merged.into_iter().collect();
                entry.tangent = merge_tangent(entry.tangent, tangent);
            }
        }
    }
    properties
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Degree {
    Zero,
    One,
    Unknown,
}

fn degree_of(
    expr: &TensorScalarExpr,
    active: &BTreeSet<TensorInputId>,
    offending: &mut BTreeSet<TensorInputId>,
) -> Degree {
    match expr {
        TensorScalarExpr::Constant { .. } => Degree::Zero,
        TensorScalarExpr::IndexEqual { .. } => Degree::Zero,
        TensorScalarExpr::Input { input, .. } => {
            if active.contains(input) {
                Degree::One
            } else {
                Degree::Zero
            }
        }
        TensorScalarExpr::Unary { op, arg } => {
            let inner = degree_of(arg, active, offending);
            match op {
                TensorUnaryOp::Neg => inner,
                _ => {
                    if inner != Degree::Zero {
                        collect_active_inputs(arg, active, offending);
                        Degree::Unknown
                    } else {
                        Degree::Zero
                    }
                }
            }
        }
        TensorScalarExpr::Binary { op, lhs, rhs } => {
            let left = degree_of(lhs, active, offending);
            let right = degree_of(rhs, active, offending);
            match op {
                TensorBinaryOp::Add | TensorBinaryOp::Sub => left.max(right),
                TensorBinaryOp::Mul => match (left, right) {
                    (Degree::Zero, other) | (other, Degree::Zero) => other,
                    _ => {
                        collect_active_inputs(lhs, active, offending);
                        collect_active_inputs(rhs, active, offending);
                        Degree::Unknown
                    }
                },
                TensorBinaryOp::Div => match right {
                    Degree::Zero => left,
                    _ => {
                        collect_active_inputs(lhs, active, offending);
                        collect_active_inputs(rhs, active, offending);
                        Degree::Unknown
                    }
                },
                TensorBinaryOp::Pow
                | TensorBinaryOp::Min
                | TensorBinaryOp::Max
                | TensorBinaryOp::Atan2 => {
                    if left == Degree::Zero && right == Degree::Zero {
                        Degree::Zero
                    } else {
                        collect_active_inputs(lhs, active, offending);
                        collect_active_inputs(rhs, active, offending);
                        Degree::Unknown
                    }
                }
            }
        }
        TensorScalarExpr::Reduction { expression, .. } => degree_of(expression, active, offending),
    }
}

fn collect_active_inputs(
    expr: &TensorScalarExpr,
    active: &BTreeSet<TensorInputId>,
    out: &mut BTreeSet<TensorInputId>,
) {
    match expr {
        TensorScalarExpr::Constant { .. } | TensorScalarExpr::IndexEqual { .. } => {}
        TensorScalarExpr::Input { input, .. } => {
            if active.contains(input) {
                out.insert(*input);
            }
        }
        TensorScalarExpr::Unary { arg, .. } => collect_active_inputs(arg, active, out),
        TensorScalarExpr::Binary { lhs, rhs, .. } => {
            collect_active_inputs(lhs, active, out);
            collect_active_inputs(rhs, active, out);
        }
        TensorScalarExpr::Reduction { expression, .. } => {
            collect_active_inputs(expression, active, out);
        }
    }
}

fn derive_trial_linearity(
    blocks: &[BlockView],
    properties: &BTreeMap<SymbolId, PropertyDependence>,
) -> Linearity {
    let mut nonlinear_symbols = BTreeSet::new();
    for block in blocks {
        for integral in &block.factorization.integrals {
            let mut active_ids = integral
                .tensor_program
                .inputs
                .iter()
                .filter(|input| input.role == TensorProgramInputRole::Active)
                .map(|input| input.id)
                .collect::<BTreeSet<_>>();
            for input in &integral.tensor_program.inputs {
                if matches!(
                    input.source,
                    InputSourceRequirement::ModelDefinedProperty { .. }
                ) && properties
                    .get(&input.binding.symbol)
                    .is_some_and(|dependence| dependence.tangent != PropertyTangent::Frozen)
                {
                    active_ids.insert(input.id);
                }
            }
            let mut offending = BTreeSet::new();
            let degree = degree_of(
                &integral.tensor_program.output.expression,
                &active_ids,
                &mut offending,
            );
            if degree == Degree::Unknown {
                for id in &offending {
                    if let Some(input) = integral
                        .tensor_program
                        .inputs
                        .iter()
                        .find(|input| input.id == *id)
                    {
                        nonlinear_symbols.insert(input.binding.symbol);
                    }
                }
            }
        }
    }
    if nonlinear_symbols.is_empty() {
        Linearity::Linear
    } else {
        Linearity::Nonlinear {
            active: nonlinear_symbols.into_iter().collect(),
        }
    }
}

fn swap_ids(expr: &TensorScalarExpr, a: TensorInputId, b: TensorInputId) -> TensorScalarExpr {
    match expr {
        TensorScalarExpr::Constant { value } => TensorScalarExpr::Constant { value: *value },
        TensorScalarExpr::Input { input, indices } => TensorScalarExpr::Input {
            input: if *input == a {
                b
            } else if *input == b {
                a
            } else {
                *input
            },
            indices: indices.clone(),
        },
        TensorScalarExpr::Unary { op, arg } => TensorScalarExpr::Unary {
            op: *op,
            arg: Box::new(swap_ids(arg, a, b)),
        },
        TensorScalarExpr::Binary { op, lhs, rhs } => TensorScalarExpr::Binary {
            op: *op,
            lhs: Box::new(swap_ids(lhs, a, b)),
            rhs: Box::new(swap_ids(rhs, a, b)),
        },
        TensorScalarExpr::IndexEqual { lhs, rhs } => TensorScalarExpr::IndexEqual {
            lhs: *lhs,
            rhs: *rhs,
        },
        TensorScalarExpr::Reduction {
            op,
            axis,
            expression,
        } => TensorScalarExpr::Reduction {
            op: *op,
            axis: *axis,
            expression: Box::new(swap_ids(expression, a, b)),
        },
    }
}

/// Structural equality up to commutativity of `Add`/`Mul`/`Min`/`Max`; every other operator
/// (`Sub`, `Div`, `Pow`, `Atan2`) is order-sensitive.
/// Flatten a chain of the same associative/commutative binary operator into its leaf operands
/// (any subexpression that is not itself a `Binary` node with the same `op`).
fn flatten_chain<'a>(
    expr: &'a TensorScalarExpr,
    op: TensorBinaryOp,
    out: &mut Vec<&'a TensorScalarExpr>,
) {
    if let TensorScalarExpr::Binary {
        op: inner,
        lhs,
        rhs,
    } = expr
        && *inner == op
    {
        flatten_chain(lhs, op, out);
        flatten_chain(rhs, op, out);
        return;
    }
    out.push(expr);
}

/// Two operand multisets are equivalent when every element of one has a distinct structurally
/// equivalent partner in the other (bipartite matching by greedy search, sufficient for the
/// small operand counts a single integrand ever produces).
fn multiset_equivalent(left: &[&TensorScalarExpr], right: &[&TensorScalarExpr]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut used = vec![false; right.len()];
    'outer: for candidate in left {
        for (index, other) in right.iter().enumerate() {
            if !used[index] && expr_equivalent(candidate, other) {
                used[index] = true;
                continue 'outer;
            }
        }
        return false;
    }
    true
}

fn expr_equivalent(a: &TensorScalarExpr, b: &TensorScalarExpr) -> bool {
    match (a, b) {
        (TensorScalarExpr::Constant { value: x }, TensorScalarExpr::Constant { value: y }) => {
            x.to_bits() == y.to_bits()
        }
        (
            TensorScalarExpr::Input {
                input: i1,
                indices: idx1,
            },
            TensorScalarExpr::Input {
                input: i2,
                indices: idx2,
            },
        ) => i1 == i2 && idx1 == idx2,
        (
            TensorScalarExpr::Unary { op: o1, arg: a1 },
            TensorScalarExpr::Unary { op: o2, arg: a2 },
        ) => o1 == o2 && expr_equivalent(a1, a2),
        (TensorScalarExpr::Binary { op: o1, .. }, TensorScalarExpr::Binary { op: o2, .. }) => {
            if o1 != o2 {
                return false;
            }
            match o1 {
                // Add/Mul are associative as well as commutative: `(k*u)*v` and `(k*v)*u` are
                // the same tree shape only after flattening the whole chain into a multiset, not
                // after a single pairwise swap at the outermost node.
                TensorBinaryOp::Add | TensorBinaryOp::Mul => {
                    let mut left = Vec::new();
                    let mut right = Vec::new();
                    flatten_chain(a, *o1, &mut left);
                    flatten_chain(b, *o1, &mut right);
                    multiset_equivalent(&left, &right)
                }
                TensorBinaryOp::Min | TensorBinaryOp::Max => {
                    let (
                        TensorScalarExpr::Binary {
                            lhs: l1, rhs: r1, ..
                        },
                        TensorScalarExpr::Binary {
                            lhs: l2, rhs: r2, ..
                        },
                    ) = (a, b)
                    else {
                        unreachable!("both sides matched Binary above");
                    };
                    (expr_equivalent(l1, l2) && expr_equivalent(r1, r2))
                        || (expr_equivalent(l1, r2) && expr_equivalent(r1, l2))
                }
                TensorBinaryOp::Sub
                | TensorBinaryOp::Div
                | TensorBinaryOp::Pow
                | TensorBinaryOp::Atan2 => {
                    let (
                        TensorScalarExpr::Binary {
                            lhs: l1, rhs: r1, ..
                        },
                        TensorScalarExpr::Binary {
                            lhs: l2, rhs: r2, ..
                        },
                    ) = (a, b)
                    else {
                        unreachable!("both sides matched Binary above");
                    };
                    expr_equivalent(l1, l2) && expr_equivalent(r1, r2)
                }
            }
        }
        (
            TensorScalarExpr::IndexEqual { lhs: l1, rhs: r1 },
            TensorScalarExpr::IndexEqual { lhs: l2, rhs: r2 },
        ) => l1 == l2 && r1 == r2,
        (
            TensorScalarExpr::Reduction {
                op: o1,
                axis: ax1,
                expression: e1,
            },
            TensorScalarExpr::Reduction {
                op: o2,
                axis: ax2,
                expression: e2,
            },
        ) => o1 == o2 && ax1 == ax2 && expr_equivalent(e1, e2),
        _ => false,
    }
}

/// Decide `form_symmetry` structurally by test/trial exchange (contract C5.4). Only attempted
/// for a single diagonal (row == active symbol) block whose every integral has exactly one test
/// and one active input; anything else is `Unknown` rather than guessed.
fn classify_symmetry(block: &BlockView) -> FormSymmetry {
    let mut any_integral = false;
    let mut symmetric = true;
    for integral in &block.factorization.integrals {
        let test_inputs = integral
            .tensor_program
            .inputs
            .iter()
            .filter(|input| input.role == TensorProgramInputRole::Test)
            .collect::<Vec<_>>();
        let active_inputs = integral
            .tensor_program
            .inputs
            .iter()
            .filter(|input| input.role == TensorProgramInputRole::Active)
            .collect::<Vec<_>>();
        if active_inputs.is_empty() {
            // A pure source/RHS term has no trial dependence at all and so cannot break the
            // bilinear form's symmetry; skip it rather than declaring the block undecidable.
            continue;
        }
        let [test] = test_inputs.as_slice() else {
            return FormSymmetry::Unknown;
        };
        let [active] = active_inputs.as_slice() else {
            return FormSymmetry::Unknown;
        };
        any_integral = true;
        let swapped = swap_ids(
            &integral.tensor_program.output.expression,
            test.id,
            active.id,
        );
        if !expr_equivalent(&integral.tensor_program.output.expression, &swapped) {
            symmetric = false;
        }
    }
    if !any_integral {
        FormSymmetry::Unknown
    } else if symmetric {
        FormSymmetry::Symmetric
    } else {
        FormSymmetry::Nonsymmetric
    }
}

/// Relabel every `Input` id through `map` (ids absent from `map` are kept).
fn relabel_inputs(
    expr: &TensorScalarExpr,
    map: &BTreeMap<TensorInputId, TensorInputId>,
) -> TensorScalarExpr {
    match expr {
        TensorScalarExpr::Constant { value } => TensorScalarExpr::Constant { value: *value },
        TensorScalarExpr::Input { input, indices } => TensorScalarExpr::Input {
            input: map.get(input).copied().unwrap_or(*input),
            indices: indices.clone(),
        },
        TensorScalarExpr::Unary { op, arg } => TensorScalarExpr::Unary {
            op: *op,
            arg: Box::new(relabel_inputs(arg, map)),
        },
        TensorScalarExpr::Binary { op, lhs, rhs } => TensorScalarExpr::Binary {
            op: *op,
            lhs: Box::new(relabel_inputs(lhs, map)),
            rhs: Box::new(relabel_inputs(rhs, map)),
        },
        TensorScalarExpr::IndexEqual { lhs, rhs } => TensorScalarExpr::IndexEqual {
            lhs: *lhs,
            rhs: *rhs,
        },
        TensorScalarExpr::Reduction {
            op,
            axis,
            expression,
        } => TensorScalarExpr::Reduction {
            op: *op,
            axis: *axis,
            expression: Box::new(relabel_inputs(expression, map)),
        },
    }
}

/// Relabel tensor axes by first appearance in a pre-order walk, so two programs that allocated
/// axis ids in different orders compare equal when their index structure agrees.
fn relabel_axes(
    expr: &TensorScalarExpr,
    map: &mut BTreeMap<TensorAxisId, TensorAxisId>,
) -> TensorScalarExpr {
    let canonical = |axis: TensorAxisId, map: &mut BTreeMap<TensorAxisId, TensorAxisId>| {
        let next = TensorAxisId(map.len() as u32);
        *map.entry(axis).or_insert(next)
    };
    match expr {
        TensorScalarExpr::Constant { value } => TensorScalarExpr::Constant { value: *value },
        TensorScalarExpr::Input { input, indices } => TensorScalarExpr::Input {
            input: *input,
            indices: indices.iter().map(|axis| canonical(*axis, map)).collect(),
        },
        TensorScalarExpr::Unary { op, arg } => TensorScalarExpr::Unary {
            op: *op,
            arg: Box::new(relabel_axes(arg, map)),
        },
        TensorScalarExpr::Binary { op, lhs, rhs } => {
            let lhs = relabel_axes(lhs, map);
            let rhs = relabel_axes(rhs, map);
            TensorScalarExpr::Binary {
                op: *op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            }
        }
        TensorScalarExpr::IndexEqual { lhs, rhs } => {
            let lhs = canonical(*lhs, map);
            let rhs = canonical(*rhs, map);
            TensorScalarExpr::IndexEqual { lhs, rhs }
        }
        TensorScalarExpr::Reduction {
            op,
            axis,
            expression,
        } => {
            let id = canonical(axis.id, map);
            TensorScalarExpr::Reduction {
                op: *op,
                axis: TensorAxis { id, ..*axis },
                expression: Box::new(relabel_axes(expression, map)),
            }
        }
    }
}

/// Split an overall `±1` factor off a product/negation tree: `-(a*b)`, `(-a)*b`, `a*(-2*b)`
/// all yield `-1` and a sign-free core. Sums are left alone (a sign inside a `Sub` is not an
/// overall sign), so an undetected relation stays `None` rather than being guessed.
fn split_sign(expr: &TensorScalarExpr) -> (i8, TensorScalarExpr) {
    match expr {
        TensorScalarExpr::Constant { value } if *value < 0.0 => {
            (-1, TensorScalarExpr::Constant { value: -value })
        }
        TensorScalarExpr::Unary {
            op: TensorUnaryOp::Neg,
            arg,
        } => {
            let (sign, core) = split_sign(arg);
            (-sign, core)
        }
        TensorScalarExpr::Binary {
            op: op @ (TensorBinaryOp::Mul | TensorBinaryOp::Div),
            lhs,
            rhs,
        } => {
            let (left, lhs) = split_sign(lhs);
            let (right, rhs) = split_sign(rhs);
            (
                left * right,
                TensorScalarExpr::Binary {
                    op: *op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            )
        }
        TensorScalarExpr::Reduction {
            op,
            axis,
            expression,
        } => {
            let (sign, core) = split_sign(expression);
            (
                sign,
                TensorScalarExpr::Reduction {
                    op: *op,
                    axis: *axis,
                    expression: Box::new(core),
                },
            )
        }
        other => (1, other.clone()),
    }
}

/// Binding identity of a coefficient input, independent of program-local ids.
type CoefficientKey = (
    SymbolId,
    crate::requirements::BasisEvaluationRequirement,
    Vec<usize>,
);

fn coefficient_key(input: &crate::tensor::TensorProgramInput) -> CoefficientKey {
    (
        input.binding.symbol,
        input.binding.evaluation.clone(),
        input.shape.clone(),
    )
}

const CANONICAL_TEST: TensorInputId = TensorInputId(0);
const CANONICAL_ACTIVE: TensorInputId = TensorInputId(1);

/// Canonical form of an integrand for cross-program comparison: the (single) test input becomes
/// `test_key`, the (single) active input `active_key`, every other input its rank in
/// `coefficients`, and axes are renumbered by first appearance.
fn canonical_integrand(
    program: &crate::tensor::TensorProgram,
    test: TensorInputId,
    active: TensorInputId,
    test_key: TensorInputId,
    active_key: TensorInputId,
    coefficients: &BTreeMap<CoefficientKey, TensorInputId>,
) -> TensorScalarExpr {
    let mut map = BTreeMap::new();
    map.insert(test, test_key);
    map.insert(active, active_key);
    for input in &program.inputs {
        if input.id != test
            && input.id != active
            && let Some(key) = coefficients.get(&coefficient_key(input))
        {
            map.insert(input.id, *key);
        }
    }
    let relabeled = relabel_inputs(&program.output.expression, &map);
    relabel_axes(&relabeled, &mut BTreeMap::new())
}

/// The integrals of `block` whose active inputs all belong to `column`, skipping pure source
/// terms (no active input). `None` when some integral mixes `column` with another active
/// symbol, which no single block coordinate describes.
fn block_integrals<'a>(
    block: &'a BlockView<'a>,
    column: SymbolId,
) -> Option<Vec<&'a crate::tensor::IntegralOperatorFactorization>> {
    let mut selected = Vec::new();
    for integral in &block.factorization.integrals {
        let active = integral
            .tensor_program
            .inputs
            .iter()
            .filter(|input| input.role == TensorProgramInputRole::Active)
            .map(|input| input.binding.symbol)
            .collect::<BTreeSet<_>>();
        if active.is_empty() {
            continue;
        }
        if active.len() > 1 {
            return None;
        }
        if active.contains(&column) {
            selected.push(integral);
        }
    }
    Some(selected)
}

/// Does the integrand of `integral` depend only affinely on its active inputs (and on the
/// properties `properties` says are not frozen)? Symmetry claims are made about bilinear forms
/// only; a nonlinear diagonal block is `Unknown`.
fn integral_is_linear(
    integral: &crate::tensor::IntegralOperatorFactorization,
    properties: &BTreeMap<SymbolId, PropertyDependence>,
) -> bool {
    let mut active_ids = integral
        .tensor_program
        .inputs
        .iter()
        .filter(|input| input.role == TensorProgramInputRole::Active)
        .map(|input| input.id)
        .collect::<BTreeSet<_>>();
    for input in &integral.tensor_program.inputs {
        if matches!(
            input.source,
            InputSourceRequirement::ModelDefinedProperty { .. }
        ) && properties
            .get(&input.binding.symbol)
            .is_some_and(|dependence| dependence.tangent != PropertyTangent::Frozen)
        {
            active_ids.insert(input.id);
        }
    }
    let mut offending = BTreeSet::new();
    degree_of(
        &integral.tensor_program.output.expression,
        &active_ids,
        &mut offending,
    ) != Degree::Unknown
}

/// Symmetry of one diagonal block by evaluation-paired test/active exchange: every test input
/// is swapped with the active input carrying the same evaluation and shape (so
/// `u v + k grad u · grad v` pairs `(v, u)` and `(grad v, grad u)` at once); an unpaired input
/// makes the block `Unknown` rather than guessed.
fn classify_diagonal_symmetry(
    block: &BlockView,
    properties: &BTreeMap<SymbolId, PropertyDependence>,
) -> FormSymmetry {
    let Some(integrals) = block_integrals(block, block.row) else {
        return FormSymmetry::Unknown;
    };
    if integrals.is_empty() {
        return FormSymmetry::Unknown;
    }
    let mut symmetric = true;
    for integral in integrals {
        if !integral_is_linear(integral, properties) {
            return FormSymmetry::Unknown;
        }
        let program = &integral.tensor_program;
        let tests = program
            .inputs
            .iter()
            .filter(|input| input.role == TensorProgramInputRole::Test)
            .collect::<Vec<_>>();
        let actives = program
            .inputs
            .iter()
            .filter(|input| input.role == TensorProgramInputRole::Active)
            .collect::<Vec<_>>();
        if tests.len() != actives.len() {
            return FormSymmetry::Unknown;
        }
        let mut map = BTreeMap::new();
        for test in &tests {
            let Some(partner) = actives.iter().find(|active| {
                active.binding.evaluation == test.binding.evaluation
                    && active.shape == test.shape
                    && !map.contains_key(&active.id)
            }) else {
                return FormSymmetry::Unknown;
            };
            map.insert(test.id, partner.id);
            map.insert(partner.id, test.id);
        }
        let swapped = relabel_inputs(&program.output.expression, &map);
        if !expr_equivalent(&program.output.expression, &swapped) {
            symmetric = false;
        }
    }
    if symmetric {
        FormSymmetry::Symmetric
    } else {
        FormSymmetry::Nonsymmetric
    }
}

fn derive_block_symmetry(
    blocks: &[BlockView],
    properties: &BTreeMap<SymbolId, PropertyDependence>,
) -> Vec<(SymbolId, FormSymmetry)> {
    let mut result = BTreeMap::new();
    for block in blocks {
        let has_diagonal = block.factorization.integrals.iter().any(|integral| {
            integral.tensor_program.inputs.iter().any(|input| {
                input.role == TensorProgramInputRole::Active && input.binding.symbol == block.row
            })
        });
        if !has_diagonal {
            continue;
        }
        let symmetry = classify_diagonal_symmetry(block, properties);
        let entry = result.entry(block.row).or_insert(symmetry);
        if *entry != symmetry {
            *entry = FormSymmetry::Unknown;
        }
    }
    result.into_iter().collect()
}

/// `σ` with `A_ij ≈ σ A_jiᵀ` for one ordered pair, or `None` when undecidable.
fn transpose_sigma(row_block: &BlockView, column_block: &BlockView) -> Option<i8> {
    let (i, j) = (row_block.row, column_block.row);
    let a_ij = block_integrals(row_block, j)?;
    let a_ji = block_integrals(column_block, i)?;
    let ([ij], [ji]) = (a_ij.as_slice(), a_ji.as_slice()) else {
        return None;
    };
    let single = |program: &crate::tensor::TensorProgram| {
        let tests = program
            .inputs
            .iter()
            .filter(|input| input.role == TensorProgramInputRole::Test)
            .collect::<Vec<_>>();
        let actives = program
            .inputs
            .iter()
            .filter(|input| input.role == TensorProgramInputRole::Active)
            .collect::<Vec<_>>();
        match (tests.as_slice(), actives.as_slice()) {
            ([test], [active]) => Some(((*test).clone(), (*active).clone())),
            _ => None,
        }
    };
    let (test_ij, active_ij) = single(&ij.tensor_program)?;
    let (test_ji, active_ji) = single(&ji.tensor_program)?;
    // The transpose pairs A_ij's test with A_ji's active (both row-`i` evaluations) and
    // A_ij's active with A_ji's test (both row-`j` evaluations).
    if test_ij.binding.evaluation != active_ji.binding.evaluation
        || test_ij.shape != active_ji.shape
        || active_ij.binding.evaluation != test_ji.binding.evaluation
        || active_ij.shape != test_ji.shape
    {
        return None;
    }
    let mut coefficients = BTreeMap::new();
    for program in [&ij.tensor_program, &ji.tensor_program] {
        for input in &program.inputs {
            if !matches!(
                input.role,
                TensorProgramInputRole::Test | TensorProgramInputRole::Active
            ) {
                coefficients.entry(coefficient_key(input)).or_insert(());
            }
        }
    }
    let coefficients = coefficients
        .into_keys()
        .enumerate()
        .map(|(rank, key)| (key, TensorInputId(2 + rank as u32)))
        .collect::<BTreeMap<_, _>>();
    let (sign_ij, core_ij) = split_sign(&canonical_integrand(
        &ij.tensor_program,
        test_ij.id,
        active_ij.id,
        CANONICAL_TEST,
        CANONICAL_ACTIVE,
        &coefficients,
    ));
    let (sign_ji, core_ji) = split_sign(&canonical_integrand(
        &ji.tensor_program,
        test_ji.id,
        active_ji.id,
        CANONICAL_ACTIVE,
        CANONICAL_TEST,
        &coefficients,
    ));
    expr_equivalent(&core_ij, &core_ji).then_some(sign_ij * sign_ji)
}

fn derive_transpose_relations(
    blocks: &[BlockView],
    structure: &[BlockStructure],
) -> Vec<TransposeRelation> {
    let present = structure
        .iter()
        .filter(|block| block.present)
        .map(|block| (block.row, block.column))
        .collect::<BTreeSet<_>>();
    let mut result = Vec::new();
    for (row, column) in &present {
        if row == column || !present.contains(&(*column, *row)) {
            continue;
        }
        let row_blocks = blocks
            .iter()
            .filter(|block| block.row == *row)
            .collect::<Vec<_>>();
        let column_blocks = blocks
            .iter()
            .filter(|block| block.row == *column)
            .collect::<Vec<_>>();
        let sigma = match (row_blocks.as_slice(), column_blocks.as_slice()) {
            ([row_block], [column_block]) => transpose_sigma(row_block, column_block),
            _ => None,
        };
        result.push(TransposeRelation {
            row: *row,
            column: *column,
            sigma,
        });
    }
    result.sort();
    result
}

/// The signed-graph gauge (§7): rows are nodes, decided two-sided off-diagonal pairs are edges
/// signed by `σ`; balanced iff a `±1` per row reproduces every edge sign as a product.
fn derive_sign_gauge(
    structure: &[BlockStructure],
    block_symmetry: &[(SymbolId, FormSymmetry)],
    transpose_relation: &[TransposeRelation],
) -> (Option<SignGauge>, Option<String>) {
    let rows = structure
        .iter()
        .map(|block| block.row)
        .collect::<BTreeSet<_>>();
    for block in structure {
        if block.present && block.row == block.column && block.class == BlockClass::Convective {
            return (
                None,
                Some(format!(
                    "diagonal block of row {} is convective; no row orientation makes it \
                     symmetric",
                    block.row
                )),
            );
        }
    }
    for (row, symmetry) in block_symmetry {
        if *symmetry != FormSymmetry::Symmetric {
            return (
                None,
                Some(format!(
                    "diagonal block of row {row} is {symmetry:?}; a sign gauge needs every \
                     present diagonal block Symmetric"
                )),
            );
        }
    }
    let present = structure
        .iter()
        .filter(|block| block.present)
        .map(|block| (block.row, block.column))
        .collect::<BTreeSet<_>>();
    for (row, column) in &present {
        if row != column && !present.contains(&(*column, *row)) {
            return (
                None,
                Some(format!(
                    "coupling from row {row} to column {column} is one-sided; no row \
                     orientation makes it symmetric"
                )),
            );
        }
    }
    let mut edges = Vec::new();
    for relation in transpose_relation {
        if relation.row >= relation.column {
            continue;
        }
        let Some(sigma) = relation.sigma else {
            return (
                None,
                Some(format!(
                    "transpose relation between rows {} and {} is undecided",
                    relation.row, relation.column
                )),
            );
        };
        edges.push(SignedEdge {
            row: relation.row,
            column: relation.column,
            sigma,
        });
    }
    // Propagate from each component's smallest row along a BFS spanning forest.
    let mut signs: BTreeMap<SymbolId, i8> = BTreeMap::new();
    let mut forest = Vec::new();
    for root in &rows {
        if signs.contains_key(root) {
            continue;
        }
        signs.insert(*root, 1);
        let mut queue = std::collections::VecDeque::from([*root]);
        while let Some(node) = queue.pop_front() {
            let node_sign = signs[&node];
            for edge in &edges {
                let other = if edge.row == node {
                    edge.column
                } else if edge.column == node {
                    edge.row
                } else {
                    continue;
                };
                if signs.contains_key(&other) {
                    continue;
                }
                signs.insert(other, node_sign * edge.sigma);
                forest.push(*edge);
                queue.push_back(other);
            }
        }
    }
    for edge in &edges {
        if signs[&edge.row] * signs[&edge.column] != edge.sigma {
            return (
                None,
                Some(format!(
                    "signed row graph is unbalanced: rows {} and {} need σ = {} but the \
                     spanning-forest signs give {}",
                    edge.row,
                    edge.column,
                    edge.sigma,
                    signs[&edge.row] * signs[&edge.column]
                )),
            );
        }
    }
    forest.sort();
    (
        Some(SignGauge {
            signs: signs.into_iter().collect(),
            proof: SignedGraphBalance {
                edges,
                spanning_forest: forest,
            },
        }),
        None,
    )
}

fn classify_block_class(
    row: SymbolId,
    column: SymbolId,
    test_kinds: &BTreeSet<DerivativeEvaluation>,
    active_kinds: &BTreeSet<DerivativeEvaluation>,
) -> BlockClass {
    use DerivativeEvaluation::{Divergence, Gradient, SymmetricGradient, TimeDerivative, Value};
    if active_kinds.contains(&TimeDerivative) {
        return BlockClass::Mass;
    }
    if row != column && (active_kinds.contains(&Divergence) || test_kinds.contains(&Divergence)) {
        return BlockClass::Constraint;
    }
    let active_diffusive = active_kinds
        .iter()
        .any(|kind| matches!(kind, Gradient | SymmetricGradient));
    let test_diffusive = test_kinds
        .iter()
        .any(|kind| matches!(kind, Gradient | SymmetricGradient));
    if active_diffusive && test_diffusive {
        return BlockClass::Diffusive;
    }
    let is_derivative = |kind: &DerivativeEvaluation| !matches!(kind, Value);
    let active_has_derivative = active_kinds.iter().any(is_derivative);
    let test_has_derivative = test_kinds.iter().any(is_derivative);
    if active_has_derivative != test_has_derivative {
        return BlockClass::Convective;
    }
    if row == column
        && !active_has_derivative
        && !test_has_derivative
        && active_kinds.contains(&Value)
        && test_kinds.contains(&Value)
    {
        return BlockClass::Reaction;
    }
    if row != column {
        return BlockClass::Coupling;
    }
    BlockClass::Unknown
}

fn derive_blocks(blocks: &[BlockView]) -> (Vec<BlockStructure>, bool) {
    let mut coordinates: BTreeMap<SymbolId, BTreeSet<SymbolId>> = BTreeMap::new();
    let mut classes: BTreeMap<(SymbolId, SymbolId), BlockClass> = BTreeMap::new();
    let mut field_order: BTreeSet<SymbolId> = BTreeSet::new();
    for block in blocks {
        field_order.insert(block.row);
        // Test evaluations are collected per column, from the integrals in which that column
        // is active: pooling a row's test evaluations across all of its integrals made a mixed
        // row's mass block (`K⁻¹ q · w`) look convective because the same row's constraint term
        // (`-p div(w)`) evaluates the test divergence (`/2`, corpus 13).
        let mut test_kinds: BTreeMap<SymbolId, BTreeSet<DerivativeEvaluation>> = BTreeMap::new();
        let mut active_kinds: BTreeMap<SymbolId, BTreeSet<DerivativeEvaluation>> = BTreeMap::new();
        for integral in &block.factorization.integrals {
            let mut integral_tests: BTreeSet<DerivativeEvaluation> = BTreeSet::new();
            let mut integral_columns: BTreeSet<SymbolId> = BTreeSet::new();
            for input in &integral.tensor_program.inputs {
                match input.role {
                    TensorProgramInputRole::Test => {
                        integral_tests.insert(input.binding.evaluation.derivative);
                    }
                    TensorProgramInputRole::Active => {
                        integral_columns.insert(input.binding.symbol);
                        active_kinds
                            .entry(input.binding.symbol)
                            .or_default()
                            .insert(input.binding.evaluation.derivative);
                    }
                    _ => {}
                }
            }
            for column in integral_columns {
                test_kinds
                    .entry(column)
                    .or_default()
                    .extend(integral_tests.iter().copied());
            }
        }
        let row_coordinates = coordinates.entry(block.row).or_default();
        for (column, kinds) in &active_kinds {
            field_order.insert(*column);
            row_coordinates.insert(*column);
            classes.insert(
                (block.row, *column),
                classify_block_class(block.row, *column, &test_kinds[column], kinds),
            );
        }
    }
    let mut saddle_point = false;
    for (row, columns) in &coordinates {
        if columns.contains(row) {
            continue;
        }
        let off_diagonal_as_row = columns.iter().any(|column| column != row);
        let off_diagonal_as_column = coordinates
            .iter()
            .any(|(other_row, other_columns)| other_row != row && other_columns.contains(row));
        if off_diagonal_as_row || off_diagonal_as_column {
            saddle_point = true;
        }
    }
    let mut result = Vec::new();
    for row in coordinates.keys() {
        for column in &field_order {
            let present = coordinates[row].contains(column);
            let class = if present {
                classes[&(*row, *column)]
            } else {
                BlockClass::Unknown
            };
            result.push(BlockStructure {
                row: *row,
                column: *column,
                present,
                class,
            });
        }
    }
    result.sort();
    (result, saddle_point)
}

fn derive_nullspace_candidates(blocks: &[BlockView]) -> Vec<NullspaceCandidate> {
    let mut kinds_by_field: BTreeMap<SymbolId, BTreeSet<DerivativeEvaluation>> = BTreeMap::new();
    let mut constrained: BTreeSet<SymbolId> = BTreeSet::new();
    let mut dimension_by_field: BTreeMap<SymbolId, u8> = BTreeMap::new();
    let mut shape_by_field: BTreeMap<SymbolId, ValueShape> = BTreeMap::new();
    let mut active_fields: BTreeSet<SymbolId> = BTreeSet::new();
    let mut rows: BTreeSet<SymbolId> = BTreeSet::new();
    let mut has_diagonal: BTreeSet<SymbolId> = BTreeSet::new();
    for block in blocks {
        active_fields.extend(active_symbols(block.form));
        rows.insert(block.row);
        for integral in &block.factorization.integrals {
            if integral.tensor_program.inputs.iter().any(|input| {
                input.role == TensorProgramInputRole::Active && input.binding.symbol == block.row
            }) {
                has_diagonal.insert(block.row);
            }
        }
        // `EssentialConstraintRequirement::argument` names the block's own generated test
        // symbol (contract C2's residual-argument bookkeeping), not the physical field's own
        // `SymbolId` -- correlate it back to `block.row` via that block's own test inputs
        // rather than comparing it directly against a field symbol.
        let test_symbols = block
            .factorization
            .integrals
            .iter()
            .flat_map(|integral| &integral.tensor_program.inputs)
            .filter(|input| input.role == TensorProgramInputRole::Test)
            .map(|input| input.binding.symbol)
            .collect::<BTreeSet<_>>();
        if block
            .requirements
            .essential_constraints
            .iter()
            .any(|constraint| test_symbols.contains(&constraint.argument))
        {
            constrained.insert(block.row);
        }
        for element in &block.requirements.elements {
            dimension_by_field
                .entry(element.symbol)
                .or_insert(element.topological_dimension);
            shape_by_field
                .entry(element.symbol)
                .or_insert_with(|| element.value_shape.clone());
        }
        for integral in &block.factorization.integrals {
            for input in &integral.tensor_program.inputs {
                if input.role != TensorProgramInputRole::Active {
                    continue;
                }
                kinds_by_field
                    .entry(input.binding.symbol)
                    .or_default()
                    .insert(input.binding.evaluation.derivative);
            }
        }
    }
    let empty_kinds: BTreeSet<DerivativeEvaluation> = BTreeSet::new();
    let mut candidates = Vec::new();
    for field in &active_fields {
        if constrained.contains(field) {
            continue;
        }
        let kinds = kinds_by_field.get(field).unwrap_or(&empty_kinds);
        let only_derivative_evaluations =
            !kinds.is_empty() && !kinds.contains(&DerivativeEvaluation::Value);
        // A field this contract's own `saddle_point` rule already flags as having no diagonal
        // block -- it never appears as its own equation's active input at all, only through
        // off-diagonal coupling (the classic Stokes/Darcy pressure slot) -- is a nullspace
        // candidate by the same reasoning even when its only active evaluation is `Value`
        // (integration by parts commonly moves the differential operator onto the test
        // function instead, e.g. `grad(pressure)` becoming `-pressure * div(v)`).
        let no_diagonal_block = rows.contains(field) && !has_diagonal.contains(field);
        if !only_derivative_evaluations && !no_diagonal_block {
            continue;
        }
        let dimension = dimension_by_field.get(field).copied();
        let is_rigid_body_vector = matches!(
            (shape_by_field.get(field), dimension),
            (Some(ValueShape::Vector(extent)), Some(dim)) if *extent == dim
        );
        if only_derivative_evaluations
            && is_rigid_body_vector
            && kinds.len() == 1
            && kinds.contains(&DerivativeEvaluation::SymmetricGradient)
        {
            candidates.push(NullspaceCandidate {
                field: *field,
                kind: NullspaceKind::RigidBody {
                    dimension: dimension.expect("checked above"),
                },
                reason: format!(
                    "field {field} appears only via symmetric-gradient evaluations with no essential constraint"
                ),
            });
        } else if only_derivative_evaluations {
            candidates.push(NullspaceCandidate {
                field: *field,
                kind: NullspaceKind::Constant,
                reason: format!(
                    "field {field} appears only via derivative evaluations with no essential constraint"
                ),
            });
        } else {
            candidates.push(NullspaceCandidate {
                field: *field,
                kind: NullspaceKind::Constant,
                reason: format!(
                    "field {field} has no diagonal block and no essential constraint (saddle-point coupling only)"
                ),
            });
        }
    }
    candidates.sort_by_key(|candidate| candidate.field);
    candidates
}

fn derive_time_structure(
    blocks: &[BlockView],
    dae_plan: Option<&IndexReductionPlan>,
) -> TimeStructure {
    let mut roles: BTreeMap<SymbolId, TimeRole> = BTreeMap::new();
    for block in blocks {
        for integral in &block.factorization.integrals {
            for input in &integral.tensor_program.inputs {
                if !matches!(
                    input.role,
                    TensorProgramInputRole::Test | TensorProgramInputRole::Active
                ) {
                    continue;
                }
                let role = if input.binding.evaluation.derivative
                    == DerivativeEvaluation::TimeDerivative
                {
                    TimeRole::Differential
                } else {
                    TimeRole::Algebraic
                };
                let entry = roles.entry(input.binding.symbol).or_insert(role);
                if role == TimeRole::Differential {
                    *entry = TimeRole::Differential;
                }
            }
        }
    }
    let transient = roles.values().any(|role| *role == TimeRole::Differential);
    TimeStructure {
        transient,
        roles: roles.into_iter().collect(),
        dae_index_lower_bound: dae_plan.map(|plan| plan.structural_index_lower_bound),
    }
}

fn build_structure(
    model: &str,
    source_digest: Digest,
    blocks: &[BlockView],
    dae_plan: Option<&IndexReductionPlan>,
) -> Result<OperatorStructure, StructureError> {
    if blocks.is_empty() {
        return Err(structure_error(
            "STRUCTURE_UNDECIDABLE",
            format!("model `{model}` has no blocks to derive a structure from"),
        ));
    }
    let properties = derive_property_dependence(blocks);
    let trial_linearity = derive_trial_linearity(blocks, &properties);
    let form_symmetry = if blocks.len() == 1 && trial_linearity == Linearity::Linear {
        classify_symmetry(&blocks[0])
    } else {
        FormSymmetry::Unknown
    };
    let (block_structures, saddle_point) = derive_blocks(blocks);
    let nullspace_candidates = derive_nullspace_candidates(blocks);
    let block_symmetry = derive_block_symmetry(blocks, &properties);
    let transpose_relation = derive_transpose_relations(blocks, &block_structures);
    let (sign_gauge, sign_gauge_reason) =
        derive_sign_gauge(&block_structures, &block_symmetry, &transpose_relation);
    let property_dependence = properties.into_values().collect::<Vec<_>>();
    let time = derive_time_structure(blocks, dae_plan);

    let identity = span_independent_digest(&StructureIdentity {
        schema: OPERATOR_STRUCTURE_SCHEMA,
        model,
        source_digest: &source_digest,
        trial_linearity: &trial_linearity,
        form_symmetry,
        blocks: &block_structures,
        saddle_point,
        nullspace_candidates: &nullspace_candidates,
        property_dependence: &property_dependence,
        time: &time,
        block_symmetry: &block_symmetry,
        transpose_relation: &transpose_relation,
        sign_gauge: &sign_gauge,
        sign_gauge_reason: &sign_gauge_reason,
    });
    let structure = OperatorStructure {
        schema: OPERATOR_STRUCTURE_SCHEMA.into(),
        model: model.to_owned(),
        source_digest,
        trial_linearity,
        form_symmetry,
        blocks: block_structures,
        saddle_point,
        nullspace_candidates,
        property_dependence,
        time,
        block_symmetry,
        transpose_relation,
        sign_gauge,
        sign_gauge_reason,
        identity,
    };
    structure.validate()?;
    Ok(structure)
}

impl OperatorStructure {
    pub fn validate(&self) -> Result<(), StructureError> {
        if self.schema != OPERATOR_STRUCTURE_SCHEMA || self.model.trim().is_empty() {
            return Err(structure_error(
                "STRUCTURE_INVALID",
                "operator structure is incomplete or has an unsupported schema",
            ));
        }
        if self.blocks.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(structure_error(
                "STRUCTURE_NONCANONICAL",
                "blocks must be uniquely sorted by (row, column)",
            ));
        }
        if self
            .block_symmetry
            .windows(2)
            .any(|pair| pair[0].0 >= pair[1].0)
            || self
                .transpose_relation
                .windows(2)
                .any(|pair| (pair[0].row, pair[0].column) >= (pair[1].row, pair[1].column))
            || self.sign_gauge.as_ref().is_some_and(|gauge| {
                gauge.signs.windows(2).any(|pair| pair[0].0 >= pair[1].0)
                    || gauge.signs.iter().any(|(_, sign)| sign.abs() != 1)
            })
        {
            return Err(structure_error(
                "STRUCTURE_NONCANONICAL",
                "block symmetry, transpose relations, and gauge signs must be uniquely sorted \
                 by row with unit signs",
            ));
        }
        if self.sign_gauge.is_some() == self.sign_gauge_reason.is_some() {
            return Err(structure_error(
                "STRUCTURE_INVALID",
                "exactly one of sign_gauge and sign_gauge_reason must be present",
            ));
        }
        if self
            .nullspace_candidates
            .windows(2)
            .any(|pair| pair[0].field >= pair[1].field)
        {
            return Err(structure_error(
                "STRUCTURE_NONCANONICAL",
                "nullspace candidates must be uniquely sorted by field",
            ));
        }
        if self
            .property_dependence
            .windows(2)
            .any(|pair| pair[0].property >= pair[1].property)
        {
            return Err(structure_error(
                "STRUCTURE_NONCANONICAL",
                "property dependence entries must be uniquely sorted by property",
            ));
        }
        if self
            .time
            .roles
            .windows(2)
            .any(|pair| pair[0].0 >= pair[1].0)
        {
            return Err(structure_error(
                "STRUCTURE_NONCANONICAL",
                "time roles must be uniquely sorted by symbol",
            ));
        }
        let expected = span_independent_digest(&StructureIdentity {
            schema: OPERATOR_STRUCTURE_SCHEMA,
            model: &self.model,
            source_digest: &self.source_digest,
            trial_linearity: &self.trial_linearity,
            form_symmetry: self.form_symmetry,
            blocks: &self.blocks,
            saddle_point: self.saddle_point,
            nullspace_candidates: &self.nullspace_candidates,
            property_dependence: &self.property_dependence,
            time: &self.time,
            block_symmetry: &self.block_symmetry,
            transpose_relation: &self.transpose_relation,
            sign_gauge: &self.sign_gauge,
            sign_gauge_reason: &self.sign_gauge_reason,
        });
        if self.identity != expected {
            return Err(structure_error(
                "STRUCTURE_IDENTITY_MISMATCH",
                "operator structure identity does not match its structural contents",
            ));
        }
        Ok(())
    }
}
