//! W11 S1: import Malleus symbolic exports (`malleus-symbolic/1`) into Resolvent terms and
//! cross-check them against the Scientia source they were lowered from.
//!
//! Scientia owns the provenance of its kernels, so it maps every exported input element back to
//! the Resolvent symbol of the Scientia input that produced it ([`SymbolMap`]) and rebuilds the
//! *source* meaning of every output independently of the Malleus lowering ([`KernelSources`]):
//! property definitions, point expressions and the generic FV flux families (central, local
//! Lax-Friedrichs, HLL) composed from the semantic flux `F`, with the characteristic speed taken
//! from Resolvent's own symbolic derivative, not from the Malleus JVP.
//!
//! # Import ([`import_symbolic`])
//!
//! - Constants are exact rationals (the export's `mantissa·2^exponent`, checked against `bits`).
//! - `fma` is the exact real `a·b + c` (its single rounding is a floating-point property only).
//! - `select` is a Resolvent `Piecewise` with one case and an `otherwise`; `abs`, `min` and
//!   `max` are encoded as the equivalent one-case piecewise terms (Resolvent has no head for
//!   them); `tan x` is `sin x / cos x`; `pow` needs an integral constant exponent.
//! - `floor`, `ceil`, `atan2` and non-integral `pow` have no Resolvent representation and refuse
//!   with `SYMBOLIC_OP_UNREPRESENTABLE`; an exported input with no Scientia provenance refuses
//!   with `SYMBOLIC_INPUT_UNMAPPED`.
//!
//! # Equivalence ([`check_equivalence`]) — method and limits
//!
//! 1. `CanonicalEquality`: both terms simplify (Resolvent `Simplify`) to the same canonical
//!    digest. This is a proof relative to the soundness of Resolvent's canonicalizer, which only
//!    flattens, sorts and folds literals (it does not expand products), so it is incomplete.
//! 2. Otherwise `ExactSamples`: both terms are evaluated with exact rational arithmetic at a
//!    deterministic Halton point set inside the [`SampleDomain`]; partial operations (division by
//!    zero, square roots of negatives, piecewise terms with no taken case) must be partial on
//!    both sides. A transcendental value at a sample is decided by substituting the sample into
//!    both terms and comparing canonical digests; if that does not decide it, the check refuses
//!    with `SYMBOLIC_EQUIVALENCE_UNDECIDED`. **Sampling is evidence, not proof**: two different
//!    rational functions agree on finitely many points only by coincidence, but nothing here
//!    bounds that probability.
//!
//! Source literals are taken at the binary64 value the kernel lowering uses
//! ([`KernelSources::rounded_literals`] counts those that are not exactly their decimal), so the
//! checks state the kernel's real semantics, not the authored decimal.
//!
//! # Derivatives and enclosures
//!
//! Malleus JVP/VJP kernels are exported and imported with direction (`Scientia.direction`),
//! cotangent (`Scientia.cotangent`) and accumulator (`Scientia.accumulator`) symbols and compared
//! with `Σ_j ∂f/∂x_j·d_j` (JVP) or `acc + Σ_i ∂f_i/∂x_j·w_i` (VJP) built from Resolvent's
//! `Differentiate` of the source. Malleus naive, centered and affine interval enclosures are run
//! on boxes of the domain and must contain the exact source value at every exactly evaluable
//! corner and Halton point of each box; this checks containment of sampled exact values, not of
//! the exact range.

use crate::conservation::{
    ConservationLawProgram, ConservationLawSystem, FvDatumSource, FvKernel, FvKernelRole,
    FvOperandMeaning, NumericalFluxIntent,
};
use crate::id::{Digest, span_independent_digest};
use crate::point_expression::{PointExpressionKernels, symbol_definition};
use crate::projection::lift_standalone_expr;
use crate::property_kernel::PropertyKernel;
use crate::scientific::{BinaryOp as SourceBinary, Expr, PropertyDefinition, PropertyModel};
use crate::semantic::{
    ExactLiteral, ExprId, SemanticExprKind, SemanticModel, SemanticModule, SymbolId,
};
use crate::source::SourceSpan;
use malleus::interval::{
    IntervalProgram, ieee_environment_supported, lower_affine_intervals, lower_centered_intervals,
    lower_intervals,
};
use malleus::symbolic::{ExactConstant, SymbolicKernel, SymbolicNode, export_symbolic};
use malleus::{
    AccessMode, BinaryOp, BufferBinding, CompareOp, DerivativeMode, DerivativeOperand,
    DerivativeRequest, Executable, Interpreter, OperandId, StructuredKernel, UnaryOp,
};
use resolvent::{
    AlgebraBudget, Atom, BooleanOperator, OPERATOR_NAMESPACE, PiecewiseCase, Rational,
    RelationOperator, RingOps, SymbolName, TermAlgebraOperation, TermBudget, TermDigest, TermId,
    TermNode, TermStore,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

pub const SYMBOLIC_VERIFICATION_SCHEMA: &str = "scientia-symbolic-verification/1";
/// Namespace of Scientia source symbols (shared with [`crate::projection`]).
pub const SOURCE_NAMESPACE: &str = "Scientia";
/// Namespace of FV facet symbols: `u_minus`, `u_plus`, `n0`, `n1`, `n2`.
pub const FV_NAMESPACE: &str = "Scientia.fv";
pub const DIRECTION_NAMESPACE: &str = "Scientia.direction";
pub const COTANGENT_NAMESPACE: &str = "Scientia.cotangent";
pub const ACCUMULATOR_NAMESPACE: &str = "Scientia.accumulator";

/// Exported input element `(operand, element)` -> the Resolvent symbol it denotes.
pub type SymbolMap = BTreeMap<(OperandId, usize), SymbolName>;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{code}: {message}")]
pub struct SymbolicVerificationError {
    pub code: &'static str,
    pub message: String,
}

fn error(code: &'static str, message: impl ToString) -> SymbolicVerificationError {
    SymbolicVerificationError {
        code,
        message: message.to_string(),
    }
}

type Result<T> = std::result::Result<T, SymbolicVerificationError>;

fn term_error(error: impl ToString) -> SymbolicVerificationError {
    self::error("SYMBOLIC_TERM_FAILED", error)
}

// ---------------------------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------------------------

/// How two terms were shown equal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum EquivalenceMethod {
    /// Equal Resolvent canonical (`Simplify`) digests: proof modulo the canonicalizer.
    CanonicalEquality,
    /// Exact agreement at `agreeing` deterministic rational samples (`partial` samples were
    /// undefined on both sides; `canonical_fallbacks` were decided by canonical digests of the
    /// substituted terms). Evidence, not proof.
    ExactSamples {
        agreeing: usize,
        partial: usize,
        canonical_fallbacks: usize,
    },
}

impl EquivalenceMethod {
    /// The weaker of two methods, summing sample counts.
    fn combine(self, other: Self) -> Self {
        match (self, other) {
            (Self::CanonicalEquality, Self::CanonicalEquality) => Self::CanonicalEquality,
            (Self::CanonicalEquality, samples) | (samples, Self::CanonicalEquality) => samples,
            (
                Self::ExactSamples {
                    agreeing: a,
                    partial: b,
                    canonical_fallbacks: c,
                },
                Self::ExactSamples {
                    agreeing: x,
                    partial: y,
                    canonical_fallbacks: z,
                },
            ) => Self::ExactSamples {
                agreeing: a + x,
                partial: b + y,
                canonical_fallbacks: c + z,
            },
        }
    }
}

/// Counts of import encodings that a reader of the evidence should know about.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportStatistics {
    pub nodes: usize,
    /// `fma` nodes imported as the exact `a·b + c`.
    pub fused_multiply_adds: usize,
    /// `select`, `abs`, `min` and `max` nodes imported as Resolvent piecewise terms.
    pub piecewise: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum CheckOutcome {
    Verified(EquivalenceMethod),
    /// The check could not be run (for example a D6 nonsmooth-derivative refusal).
    Refused {
        code: String,
        message: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolicDerivativeEvidence {
    pub mode: DerivativeMode,
    pub independents: Vec<String>,
    pub outcome: CheckOutcome,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnclosureLowering {
    Naive,
    Centered,
    Affine,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum EnclosureOutcome {
    /// Every exactly evaluated point of every box lies inside the Malleus enclosure.
    Contained { boxes: usize, points: usize },
    /// The lowering refused the kernel (unsupported operation) or the host environment.
    Refused { reason: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnclosureEvidence {
    pub lowering: EnclosureLowering,
    pub outcome: EnclosureOutcome,
}

/// Which Scientia artifact a verified kernel came from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KernelProvenance {
    PropertyValue {
        property: String,
        identity: Digest,
    },
    PropertyTangent {
        property: String,
        input: String,
        identity: Digest,
    },
    PointExpression {
        model: String,
        expression: ExprId,
        artifact: Digest,
    },
    FiniteVolume {
        role: FvKernelRole,
        program: Digest,
    },
}

/// One kernel whose Malleus export was round-trip verified against its Scientia source.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolicVerificationEntry {
    pub kernel: String,
    pub kernel_digest: malleus::Digest,
    pub provenance: KernelProvenance,
    pub outputs: usize,
    pub rounded_literals: usize,
    pub import: ImportStatistics,
    pub round_trip: EquivalenceMethod,
    pub derivatives: Vec<SymbolicDerivativeEvidence>,
    pub enclosures: Vec<EnclosureEvidence>,
}

/// Report kind (`scientia-symbolic-verification/1`) recording which kernels are round-trip
/// verified and by which method. It is a standalone artifact: it adds no
/// `VerificationObligationKind` and changes no existing identity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolicVerificationReport {
    pub schema: String,
    pub entries: Vec<SymbolicVerificationEntry>,
    pub digest: Digest,
}

#[derive(Serialize)]
struct ReportPayload<'a> {
    schema: &'a str,
    entries: &'a [SymbolicVerificationEntry],
}

impl SymbolicVerificationReport {
    pub fn new(entries: Vec<SymbolicVerificationEntry>) -> Self {
        let digest = span_independent_digest(&ReportPayload {
            schema: SYMBOLIC_VERIFICATION_SCHEMA,
            entries: &entries,
        });
        Self {
            schema: SYMBOLIC_VERIFICATION_SCHEMA.into(),
            entries,
            digest,
        }
    }

    pub fn validate_identity(&self) -> Result<()> {
        let expected = span_independent_digest(&ReportPayload {
            schema: &self.schema,
            entries: &self.entries,
        });
        if self.schema != SYMBOLIC_VERIFICATION_SCHEMA || expected != self.digest {
            return Err(error(
                "SYMBOLIC_REPORT_IDENTITY",
                "report schema or digest does not match its entries",
            ));
        }
        Ok(())
    }

    /// Whether `kernel_digest` has a round-trip verified entry.
    pub fn verifies(&self, kernel_digest: &malleus::Digest) -> bool {
        self.entries
            .iter()
            .any(|entry| &entry.kernel_digest == kernel_digest)
    }
}

// ---------------------------------------------------------------------------------------------
// Term construction
// ---------------------------------------------------------------------------------------------

fn budget() -> TermBudget {
    TermBudget::default()
}

fn symbol(store: &mut TermStore, name: SymbolName) -> Result<TermId> {
    store.atom(Atom::Symbol(name), budget()).map_err(term_error)
}

fn rational(store: &mut TermStore, value: Rational) -> Result<TermId> {
    store.exact_rational(value, budget()).map_err(term_error)
}

fn integer(store: &mut TermStore, value: i128) -> Result<TermId> {
    store.exact_integer(value, budget()).map_err(term_error)
}

fn apply(store: &mut TermStore, head: &str, arguments: Vec<TermId>) -> Result<TermId> {
    let head = symbol(store, SymbolName::new(OPERATOR_NAMESPACE, head))?;
    store
        .intern(TermNode::Apply { head, arguments }, budget())
        .map_err(term_error)
}

fn relation(
    store: &mut TermStore,
    operator: RelationOperator,
    left: TermId,
    right: TermId,
) -> Result<TermId> {
    store
        .intern(
            TermNode::Relation {
                operator,
                left,
                right,
            },
            budget(),
        )
        .map_err(term_error)
}

fn piecewise(
    store: &mut TermStore,
    condition: TermId,
    value: TermId,
    otherwise: TermId,
) -> Result<TermId> {
    store
        .intern(
            TermNode::Piecewise {
                cases: vec![PiecewiseCase { value, condition }],
                otherwise: Some(otherwise),
            },
            budget(),
        )
        .map_err(term_error)
}

/// `|x|` as `x if x >= 0 else -x`.
fn abs(store: &mut TermStore, x: TermId) -> Result<TermId> {
    let zero = integer(store, 0)?;
    let condition = relation(store, RelationOperator::GreaterEqual, x, zero)?;
    let negated = apply(store, "Neg", vec![x])?;
    piecewise(store, condition, x, negated)
}

/// `min`/`max` as `l if l <= r else r` / `l if l >= r else r`.
fn extremum(store: &mut TermStore, max: bool, l: TermId, r: TermId) -> Result<TermId> {
    let operator = if max {
        RelationOperator::GreaterEqual
    } else {
        RelationOperator::LessEqual
    };
    let condition = relation(store, operator, l, r)?;
    piecewise(store, condition, l, r)
}

fn sum(store: &mut TermStore, terms: Vec<TermId>) -> Result<TermId> {
    match terms.len() {
        0 => integer(store, 0),
        1 => Ok(terms[0]),
        _ => apply(store, "Add", terms),
    }
}

fn substitute(
    store: &mut TermStore,
    root: TermId,
    pairs: &[(SymbolName, TermId)],
) -> Result<TermId> {
    let mut replacements = Vec::with_capacity(pairs.len());
    for (name, value) in pairs {
        replacements.push((symbol(store, name.clone())?, *value));
    }
    store
        .substitute_closed(root, &replacements, budget())
        .map_err(term_error)
}

/// Rewrite `f(.., piecewise(c ⇒ a; b), ..)` to `piecewise(c ⇒ f(.., a, ..); f(.., b, ..))`
/// through every operator application, so piecewise terms only occur at the top of values
/// (conditions are left as they are). This is exact in real semantics (branches stay lazy) and
/// is what Resolvent's branchwise `Differentiate` needs, since it treats piecewise terms only at
/// the root of a derivative request.
fn hoist_piecewise(
    store: &mut TermStore,
    root: TermId,
    memo: &mut HashMap<TermId, TermId>,
) -> Result<TermId> {
    if let Some(done) = memo.get(&root) {
        return Ok(*done);
    }
    let hoisted = match store.node(root).map_err(term_error)? {
        TermNode::Apply { head, arguments } => {
            let arguments = arguments
                .iter()
                .map(|argument| hoist_piecewise(store, *argument, memo))
                .collect::<Result<Vec<_>>>()?;
            let mut split = None;
            for (position, argument) in arguments.iter().enumerate() {
                if let TermNode::Piecewise { cases, otherwise } =
                    store.node(*argument).map_err(term_error)?
                {
                    split = Some((position, cases, otherwise));
                    break;
                }
            }
            match split {
                None => store
                    .intern(TermNode::Apply { head, arguments }, budget())
                    .map_err(term_error)?,
                Some((position, cases, otherwise)) => {
                    let mut branch = |store: &mut TermStore, value: TermId| {
                        let mut arguments = arguments.clone();
                        arguments[position] = value;
                        let applied = store
                            .intern(TermNode::Apply { head, arguments }, budget())
                            .map_err(term_error)?;
                        hoist_piecewise(store, applied, memo)
                    };
                    let mut hoisted_cases = Vec::with_capacity(cases.len());
                    for case in cases {
                        hoisted_cases.push(PiecewiseCase {
                            value: branch(store, case.value)?,
                            condition: case.condition,
                        });
                    }
                    let otherwise = match otherwise {
                        Some(value) => Some(branch(store, value)?),
                        None => None,
                    };
                    store
                        .intern(
                            TermNode::Piecewise {
                                cases: hoisted_cases,
                                otherwise,
                            },
                            budget(),
                        )
                        .map_err(term_error)?
                }
            }
        }
        TermNode::Piecewise { cases, otherwise } => {
            let mut hoisted_cases = Vec::with_capacity(cases.len());
            for case in cases {
                hoisted_cases.push(PiecewiseCase {
                    value: hoist_piecewise(store, case.value, memo)?,
                    condition: case.condition,
                });
            }
            let otherwise = match otherwise {
                Some(value) => Some(hoist_piecewise(store, value, memo)?),
                None => None,
            };
            store
                .intern(
                    TermNode::Piecewise {
                        cases: hoisted_cases,
                        otherwise,
                    },
                    budget(),
                )
                .map_err(term_error)?
        }
        _ => root,
    };
    memo.insert(root, hoisted);
    Ok(hoisted)
}

fn differentiate(store: &mut TermStore, root: TermId, variable: &SymbolName) -> Result<TermId> {
    let root = hoist_piecewise(store, root, &mut HashMap::new())?;
    store
        .apply_algebra_operation(
            root,
            &TermAlgebraOperation::Differentiate {
                variable: variable.clone(),
            },
            budget(),
            AlgebraBudget::default(),
        )
        .map(|outcome| outcome.root)
        .map_err(|error| self::error("SYMBOLIC_DERIVATIVE_FAILED", error))
}

fn canonical_digest(store: &mut TermStore, root: TermId) -> Option<TermDigest> {
    let simplified = store
        .apply_algebra_operation(
            root,
            &TermAlgebraOperation::Simplify,
            budget(),
            AlgebraBudget::default(),
        )
        .ok()?;
    store.digest(simplified.root, budget()).ok()
}

fn exact_rational_of(value: f64) -> Result<Rational> {
    Rational::from_f64(value).ok_or_else(|| {
        error(
            "SYMBOLIC_SOURCE_UNSUPPORTED",
            format!("literal {value} is not finite"),
        )
    })
}

// ---------------------------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedOutput {
    pub operand: OperandId,
    pub element: usize,
    pub term: TermId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedKernel {
    pub kernel: String,
    pub kernel_digest: malleus::Digest,
    pub outputs: Vec<ImportedOutput>,
    pub statistics: ImportStatistics,
}

fn constant_rational(constant: &ExactConstant) -> Result<Rational> {
    let value = f64::from_bits(constant.bits);
    if ExactConstant::new(value).as_ref() != Some(constant) {
        return Err(error(
            "SYMBOLIC_CONSTANT_INCONSISTENT",
            format!(
                "constant {}·2^{} disagrees with its bits {:#x}",
                constant.mantissa, constant.exponent, constant.bits
            ),
        ));
    }
    exact_rational_of(value)
}

fn integral_exponent(node: &SymbolicNode) -> Option<i128> {
    let SymbolicNode::Constant(constant) = node else {
        return None;
    };
    let value = f64::from_bits(constant.bits);
    (value.fract() == 0.0 && value.abs() <= f64::from(i32::MAX)).then_some(value as i128)
}

/// Import every exported output as a Resolvent term over `symbols`.
pub fn import_symbolic(
    export: &SymbolicKernel,
    symbols: &SymbolMap,
    store: &mut TermStore,
) -> Result<ImportedKernel> {
    export
        .check()
        .map_err(|error| self::error("SYMBOLIC_SCHEMA_INVALID", error))?;
    let mut statistics = ImportStatistics {
        nodes: export.nodes.len(),
        ..ImportStatistics::default()
    };
    let unrepresentable = |what: String| {
        error(
            "SYMBOLIC_OP_UNREPRESENTABLE",
            format!("{what} has no Resolvent representation"),
        )
    };
    let mut terms: Vec<TermId> = Vec::with_capacity(export.nodes.len());
    for node in &export.nodes {
        let term = match *node {
            SymbolicNode::Input { input } => {
                let item = &export.inputs[input];
                let name = symbols.get(&(item.operand, item.element)).ok_or_else(|| {
                    error(
                        "SYMBOLIC_INPUT_UNMAPPED",
                        format!(
                            "input `{}` (operand {}, element {}) has no Scientia provenance",
                            item.name,
                            item.operand.index(),
                            item.element
                        ),
                    )
                })?;
                symbol(store, name.clone())?
            }
            SymbolicNode::Constant(constant) => {
                let value = constant_rational(&constant)?;
                rational(store, value)?
            }
            SymbolicNode::Unary { op, arg } => {
                let x = terms[arg];
                match op {
                    UnaryOp::Neg => apply(store, "Neg", vec![x])?,
                    UnaryOp::Abs => {
                        statistics.piecewise += 1;
                        abs(store, x)?
                    }
                    UnaryOp::Sqrt => apply(store, "Sqrt", vec![x])?,
                    UnaryOp::Exp => apply(store, "Exp", vec![x])?,
                    UnaryOp::Ln => apply(store, "Ln", vec![x])?,
                    UnaryOp::Sin => apply(store, "Sin", vec![x])?,
                    UnaryOp::Cos => apply(store, "Cos", vec![x])?,
                    UnaryOp::Tan => {
                        let sin = apply(store, "Sin", vec![x])?;
                        let cos = apply(store, "Cos", vec![x])?;
                        apply(store, "Div", vec![sin, cos])?
                    }
                    UnaryOp::Floor | UnaryOp::Ceil => {
                        return Err(unrepresentable(format!("{op:?}")));
                    }
                }
            }
            SymbolicNode::Binary { op, lhs, rhs } => {
                let (l, r) = (terms[lhs], terms[rhs]);
                match op {
                    BinaryOp::Add => apply(store, "Add", vec![l, r])?,
                    BinaryOp::Sub => apply(store, "Sub", vec![l, r])?,
                    BinaryOp::Mul => apply(store, "Mul", vec![l, r])?,
                    BinaryOp::Div => apply(store, "Div", vec![l, r])?,
                    BinaryOp::Pow => {
                        let exponent = integral_exponent(&export.nodes[rhs]).ok_or_else(|| {
                            unrepresentable(
                                "pow with a non-integral or non-constant exponent".into(),
                            )
                        })?;
                        let exponent = integer(store, exponent)?;
                        apply(store, "Pow", vec![l, exponent])?
                    }
                    BinaryOp::Min | BinaryOp::Max => {
                        statistics.piecewise += 1;
                        extremum(store, op == BinaryOp::Max, l, r)?
                    }
                    BinaryOp::Atan2 => return Err(unrepresentable("atan2".into())),
                }
            }
            SymbolicNode::Fma {
                lhs, rhs, addend, ..
            } => {
                statistics.fused_multiply_adds += 1;
                let product = apply(store, "Mul", vec![terms[lhs], terms[rhs]])?;
                apply(store, "Add", vec![product, terms[addend]])?
            }
            SymbolicNode::Bool { value } => store
                .atom(Atom::Boolean(value), budget())
                .map_err(term_error)?,
            SymbolicNode::Compare { op, lhs, rhs } => {
                let operator = match op {
                    CompareOp::Eq => RelationOperator::Equal,
                    CompareOp::NotEq => RelationOperator::NotEqual,
                    CompareOp::Less => RelationOperator::Less,
                    CompareOp::LessEqual => RelationOperator::LessEqual,
                    CompareOp::Greater => RelationOperator::Greater,
                    CompareOp::GreaterEqual => RelationOperator::GreaterEqual,
                };
                relation(store, operator, terms[lhs], terms[rhs])?
            }
            SymbolicNode::Not { arg } => boolean(store, BooleanOperator::Not, vec![terms[arg]])?,
            SymbolicNode::And { lhs, rhs } => {
                boolean(store, BooleanOperator::And, vec![terms[lhs], terms[rhs]])?
            }
            SymbolicNode::Or { lhs, rhs } => {
                boolean(store, BooleanOperator::Or, vec![terms[lhs], terms[rhs]])?
            }
            SymbolicNode::Select {
                condition,
                if_true,
                if_false,
            } => {
                statistics.piecewise += 1;
                piecewise(store, terms[condition], terms[if_true], terms[if_false])?
            }
        };
        terms.push(term);
    }
    Ok(ImportedKernel {
        kernel: export.kernel.clone(),
        kernel_digest: export.kernel_digest.clone(),
        outputs: export
            .outputs
            .iter()
            .map(|output| ImportedOutput {
                operand: output.operand,
                element: output.element,
                term: terms[output.node],
            })
            .collect(),
        statistics,
    })
}

fn boolean(
    store: &mut TermStore,
    operator: BooleanOperator,
    arguments: Vec<TermId>,
) -> Result<TermId> {
    store
        .intern(
            TermNode::Boolean {
                operator,
                arguments,
            },
            budget(),
        )
        .map_err(term_error)
}

// ---------------------------------------------------------------------------------------------
// Exact evaluation
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Value {
    Number(Rational),
    Truth(bool),
}

#[derive(Clone, Debug, PartialEq)]
enum Stop {
    /// The real operation is undefined here.
    Partial,
    /// The value is real but not an exact rational (a transcendental of a nonspecial argument).
    NotExact,
    Unsupported(String),
}

struct Evaluator<'a> {
    store: &'a TermStore,
    environment: &'a BTreeMap<SymbolName, Rational>,
    memo: HashMap<TermId, std::result::Result<Value, Stop>>,
}

impl Evaluator<'_> {
    fn number(&mut self, id: TermId) -> std::result::Result<Rational, Stop> {
        match self.eval(id)? {
            Value::Number(value) => Ok(value),
            Value::Truth(_) => Err(Stop::Unsupported("a predicate used as a number".into())),
        }
    }

    fn truth(&mut self, id: TermId) -> std::result::Result<bool, Stop> {
        match self.eval(id)? {
            Value::Truth(value) => Ok(value),
            Value::Number(_) => Err(Stop::Unsupported("a number used as a predicate".into())),
        }
    }

    fn eval(&mut self, id: TermId) -> std::result::Result<Value, Stop> {
        if let Some(value) = self.memo.get(&id) {
            return value.clone();
        }
        let value = self.compute(id);
        self.memo.insert(id, value.clone());
        value
    }

    fn compute(&mut self, id: TermId) -> std::result::Result<Value, Stop> {
        let node = self
            .store
            .node(id)
            .map_err(|error| Stop::Unsupported(error.to_string()))?;
        match node {
            TermNode::Atom(Atom::Symbol(name)) => self
                .environment
                .get(&name)
                .cloned()
                .map(Value::Number)
                .ok_or_else(|| Stop::Unsupported(format!("unbound symbol {name:?}"))),
            TermNode::Atom(Atom::Boolean(value)) => Ok(Value::Truth(value)),
            TermNode::Atom(atom) => atom
                .exact_rational_value()
                .map(Value::Number)
                .ok_or_else(|| Stop::Unsupported(format!("atom {atom:?}"))),
            TermNode::Apply { head, arguments } => self.apply(head, &arguments).map(Value::Number),
            TermNode::Relation {
                operator,
                left,
                right,
            } => {
                let (l, r) = (self.number(left)?, self.number(right)?);
                Ok(Value::Truth(match operator {
                    RelationOperator::Equal => l == r,
                    RelationOperator::NotEqual => l != r,
                    RelationOperator::Less => l < r,
                    RelationOperator::LessEqual => l <= r,
                    RelationOperator::Greater => l > r,
                    RelationOperator::GreaterEqual => l >= r,
                }))
            }
            TermNode::Boolean {
                operator,
                arguments,
            } => {
                let values = arguments
                    .iter()
                    .map(|argument| self.truth(*argument))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(Value::Truth(match (operator, values.as_slice()) {
                    (BooleanOperator::Not, [value]) => !value,
                    (BooleanOperator::And, _) => values.iter().all(|value| *value),
                    (BooleanOperator::Or, _) => values.iter().any(|value| *value),
                    (operator, _) => {
                        return Err(Stop::Unsupported(format!("boolean {operator:?}")));
                    }
                }))
            }
            TermNode::Piecewise { cases, otherwise } => {
                for case in cases {
                    if self.truth(case.condition)? {
                        return self.eval(case.value);
                    }
                }
                match otherwise {
                    Some(otherwise) => self.eval(otherwise),
                    None => Err(Stop::Partial),
                }
            }
            other => Err(Stop::Unsupported(format!("{other:?}"))),
        }
    }

    fn apply(&mut self, head: TermId, arguments: &[TermId]) -> std::result::Result<Rational, Stop> {
        let name = match self.store.node(head) {
            Ok(TermNode::Atom(Atom::Symbol(name))) if name.namespace == OPERATOR_NAMESPACE => {
                name.name
            }
            _ => return Err(Stop::Unsupported("a non-operator head".into())),
        };
        let values = |this: &mut Self| {
            arguments
                .iter()
                .map(|argument| this.number(*argument))
                .collect::<std::result::Result<Vec<_>, _>>()
        };
        match (name.as_str(), arguments.len()) {
            ("Add", _) => Ok(values(self)?
                .iter()
                .fold(Rational::zero(), |sum, value| sum.add(value))),
            ("Mul", _) => Ok(values(self)?
                .iter()
                .fold(Rational::one(), |product, value| product.mul(value))),
            ("Sub", 2) => {
                let v = values(self)?;
                Ok(v[0].sub(&v[1]))
            }
            ("Div", 2) => {
                let v = values(self)?;
                let reciprocal = v[1].checked_recip().ok_or(Stop::Partial)?;
                Ok(v[0].mul(&reciprocal))
            }
            ("Neg", 1) => Ok(self.number(arguments[0])?.neg()),
            ("Pow", 2) => {
                let base = self.number(arguments[0])?;
                let exponent = self.number(arguments[1])?;
                let lossy = exponent.to_f64_lossy();
                let integral = lossy.fract() == 0.0
                    && lossy.abs() <= f64::from(i32::MAX)
                    && Rational::from_i64(lossy as i64) == exponent;
                if !integral {
                    return Err(Stop::Unsupported("a non-integral exponent".into()));
                }
                base.checked_pow(lossy as i32).ok_or(Stop::Partial)
            }
            ("Sqrt", 1) => {
                let x = self.number(arguments[0])?;
                if x < Rational::zero() {
                    return Err(Stop::Partial);
                }
                x.sqrt_exact().ok_or(Stop::NotExact)
            }
            ("Exp", 1) => special(
                self.number(arguments[0])?,
                Rational::zero(),
                Rational::one(),
            ),
            ("Ln" | "Log", 1) => {
                let x = self.number(arguments[0])?;
                if x <= Rational::zero() {
                    return Err(Stop::Partial);
                }
                special(x, Rational::one(), Rational::zero())
            }
            ("Sin", 1) => special(
                self.number(arguments[0])?,
                Rational::zero(),
                Rational::zero(),
            ),
            ("Cos", 1) => special(
                self.number(arguments[0])?,
                Rational::zero(),
                Rational::one(),
            ),
            (other, arity) => Err(Stop::Unsupported(format!("{other}/{arity}"))),
        }
    }
}

/// The exact value of a transcendental at its one rational special point.
fn special(x: Rational, at: Rational, value: Rational) -> std::result::Result<Rational, Stop> {
    if x == at {
        Ok(value)
    } else {
        Err(Stop::NotExact)
    }
}

fn evaluate(
    store: &TermStore,
    environment: &BTreeMap<SymbolName, Rational>,
    root: TermId,
) -> std::result::Result<Rational, Stop> {
    Evaluator {
        store,
        environment,
        memo: HashMap::new(),
    }
    .number(root)
}

// ---------------------------------------------------------------------------------------------
// Sampling and equivalence
// ---------------------------------------------------------------------------------------------

/// Deterministic sample domain: every free symbol ranges over its box (default `default`).
/// Box endpoints must be finite and ordered; they are taken as exact rationals.
#[derive(Clone, Debug, PartialEq)]
pub struct SampleDomain {
    pub default: (f64, f64),
    pub symbols: BTreeMap<SymbolName, (f64, f64)>,
    /// Halton points per equivalence check and per enclosure box.
    pub samples: usize,
}

impl Default for SampleDomain {
    fn default() -> Self {
        Self {
            default: (-2.0, 3.0),
            symbols: BTreeMap::new(),
            samples: 24,
        }
    }
}

impl SampleDomain {
    pub fn with(mut self, name: SymbolName, range: (f64, f64)) -> Self {
        self.symbols.insert(name, range);
        self
    }

    fn range(&self, name: &SymbolName) -> Result<(Rational, Rational)> {
        let (lo, hi) = self.symbols.get(name).copied().unwrap_or(self.default);
        if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
            return Err(error(
                "SYMBOLIC_SAMPLE_INVALID",
                format!("sample range [{lo}, {hi}] of {name:?} is not a finite ordered box"),
            ));
        }
        Ok((exact_rational_of(lo)?, exact_rational_of(hi)?))
    }
}

const PRIMES: [i64; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

/// The radical inverse of `index` in `base`, an exact rational in (0, 1).
fn halton(mut index: i64, base: i64) -> Rational {
    let mut result = Rational::zero();
    let mut scale = Rational::from_ratio(1, base);
    let step = Rational::from_ratio(1, base);
    while index > 0 {
        result = result.add(&scale.mul(&Rational::from_i64(index % base)));
        index /= base;
        scale = scale.mul(&step);
    }
    result
}

fn free_symbols(store: &TermStore, roots: &[TermId]) -> Result<Vec<SymbolName>> {
    let mut names = std::collections::BTreeSet::new();
    for root in roots {
        for name in store.free_symbols(*root, budget()).map_err(term_error)? {
            if name.namespace != OPERATOR_NAMESPACE {
                names.insert(name);
            }
        }
    }
    Ok(names.into_iter().collect())
}

fn point(
    names: &[SymbolName],
    boxes: &[(Rational, Rational)],
    index: i64,
) -> Result<BTreeMap<SymbolName, Rational>> {
    if names.len() > PRIMES.len() {
        return Err(error(
            "SYMBOLIC_SAMPLE_INVALID",
            format!(
                "{} free symbols exceed the {} sampled dimensions",
                names.len(),
                PRIMES.len()
            ),
        ));
    }
    Ok(names
        .iter()
        .zip(boxes)
        .zip(PRIMES)
        .map(|((name, (lo, hi)), base)| {
            let t = halton(index, base);
            (name.clone(), lo.add(&hi.sub(lo).mul(&t)))
        })
        .collect())
}

fn describe(environment: &BTreeMap<SymbolName, Rational>) -> String {
    environment
        .iter()
        .map(|(name, value)| format!("{}={}", name.name, value.to_f64_lossy()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Show `expected` (the Scientia source meaning) and `actual` (the imported export) equal.
pub fn check_equivalence(
    store: &mut TermStore,
    expected: TermId,
    actual: TermId,
    domain: &SampleDomain,
) -> Result<EquivalenceMethod> {
    if let (Some(a), Some(b)) = (
        canonical_digest(store, expected),
        canonical_digest(store, actual),
    ) && a == b
    {
        return Ok(EquivalenceMethod::CanonicalEquality);
    }
    let names = free_symbols(store, &[expected, actual])?;
    let boxes = names
        .iter()
        .map(|name| domain.range(name))
        .collect::<Result<Vec<_>>>()?;
    let (mut agreeing, mut partial, mut canonical_fallbacks) = (0, 0, 0);
    for index in 1..=domain.samples as i64 {
        let environment = point(&names, &boxes, index)?;
        let left = evaluate(store, &environment, expected);
        let right = evaluate(store, &environment, actual);
        match (left, right) {
            (Ok(a), Ok(b)) if a == b => agreeing += 1,
            (Ok(a), Ok(b)) => {
                return Err(error(
                    "SYMBOLIC_EQUIVALENCE_MISMATCH",
                    format!(
                        "source {} != export {} at {}",
                        a.to_f64_lossy(),
                        b.to_f64_lossy(),
                        describe(&environment)
                    ),
                ));
            }
            (Err(Stop::Partial), Err(Stop::Partial)) => partial += 1,
            (Err(Stop::Unsupported(what)), _) | (_, Err(Stop::Unsupported(what))) => {
                return Err(error(
                    "SYMBOLIC_EQUIVALENCE_UNDECIDED",
                    format!("exact evaluation does not support {what}"),
                ));
            }
            (Err(Stop::Partial), _) | (_, Err(Stop::Partial)) => {
                return Err(error(
                    "SYMBOLIC_EQUIVALENCE_MISMATCH",
                    format!("only one side is undefined at {}", describe(&environment)),
                ));
            }
            (Err(Stop::NotExact), _) | (_, Err(Stop::NotExact)) => {
                let pairs = environment
                    .iter()
                    .map(|(name, value)| Ok((name.clone(), rational(store, value.clone())?)))
                    .collect::<Result<Vec<_>>>()?;
                let a = substitute(store, expected, &pairs)?;
                let b = substitute(store, actual, &pairs)?;
                match (canonical_digest(store, a), canonical_digest(store, b)) {
                    (Some(a), Some(b)) if a == b => canonical_fallbacks += 1,
                    _ => {
                        return Err(error(
                            "SYMBOLIC_EQUIVALENCE_UNDECIDED",
                            format!(
                                "transcendental values at {} are not decided exactly",
                                describe(&environment)
                            ),
                        ));
                    }
                }
            }
        }
    }
    if agreeing + canonical_fallbacks == 0 {
        return Err(error(
            "SYMBOLIC_EQUIVALENCE_UNDECIDED",
            "no sample point is inside both terms' domains",
        ));
    }
    Ok(EquivalenceMethod::ExactSamples {
        agreeing,
        partial,
        canonical_fallbacks,
    })
}

// ---------------------------------------------------------------------------------------------
// Source meaning
// ---------------------------------------------------------------------------------------------

/// The independent Scientia meaning of a kernel: input provenance and one source term per
/// scalar output element.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelSources {
    pub symbols: SymbolMap,
    pub outputs: BTreeMap<(OperandId, usize), TermId>,
    /// Source literals whose binary64 value (the one the kernel uses) is not their decimal.
    pub rounded_literals: usize,
}

struct SourceProjector<'a> {
    model: &'a SemanticModel,
    substitutions: BTreeMap<SymbolId, TermId>,
    inlining: Vec<SymbolId>,
    rounded_literals: usize,
}

fn unsupported(what: impl ToString) -> SymbolicVerificationError {
    error("SYMBOLIC_SOURCE_UNSUPPORTED", what)
}

fn literal_rational(exact: &ExactLiteral) -> Result<Rational> {
    match exact {
        ExactLiteral::Integer(value) => Ok(Rational::from_i128(*value)),
        ExactLiteral::Decimal { mantissa, exponent } => {
            resolvent::ExactDecimal::from_parts(*mantissa, *exponent)
                .map(|decimal| decimal.to_rational())
                .map_err(term_error)
        }
    }
}

impl SourceProjector<'_> {
    fn scalar(&mut self, store: &mut TermStore, id: ExprId) -> Result<TermId> {
        let components = self.project(store, id)?;
        match components.as_slice() {
            [value] => Ok(*value),
            _ => Err(unsupported(format!("expression {id} is not scalar"))),
        }
    }

    fn project(&mut self, store: &mut TermStore, id: ExprId) -> Result<Vec<TermId>> {
        let model = self.model;
        let expression = model
            .expressions
            .get(id.index())
            .ok_or_else(|| unsupported(format!("expression {id} is outside the arena")))?;
        match &expression.kind {
            SemanticExprKind::Number { value, exact, .. } => {
                let kernel_value = exact_rational_of(*value)?;
                if literal_rational(exact)? != kernel_value {
                    self.rounded_literals += 1;
                }
                Ok(vec![rational(store, kernel_value)?])
            }
            SemanticExprKind::Symbol { symbol: id } => {
                if let Some(term) = self.substitutions.get(id) {
                    return Ok(vec![*term]);
                }
                if let Some(definition) = symbol_definition(model, *id) {
                    if self.inlining.contains(id) {
                        return Err(unsupported("a cyclic definition"));
                    }
                    self.inlining.push(*id);
                    let value = self.project(store, definition);
                    self.inlining.pop();
                    return value;
                }
                let name = &model.symbols[id.index()].name;
                Ok(vec![symbol(
                    store,
                    SymbolName::new(SOURCE_NAMESPACE, name.clone()),
                )?])
            }
            SemanticExprKind::Unary { arg, .. } => self
                .project(store, *arg)?
                .into_iter()
                .map(|value| apply(store, "Neg", vec![value]))
                .collect(),
            SemanticExprKind::Binary { op, lhs, rhs } => {
                if *op == SourceBinary::Pow {
                    let base = self.scalar(store, *lhs)?;
                    let exponent = match &model.expressions[rhs.index()].kind {
                        SemanticExprKind::Number {
                            exact: ExactLiteral::Integer(value),
                            ..
                        } => *value,
                        _ => return Err(unsupported("a non-integer-literal exponent")),
                    };
                    let exponent = integer(store, exponent)?;
                    return Ok(vec![apply(store, "Pow", vec![base, exponent])?]);
                }
                let head = match op {
                    SourceBinary::Add => "Add",
                    SourceBinary::Sub => "Sub",
                    SourceBinary::Mul => "Mul",
                    SourceBinary::Div => "Div",
                    _ => return Err(unsupported("a comparison")),
                };
                let l = self.project(store, *lhs)?;
                let r = self.project(store, *rhs)?;
                let pairs: Vec<(TermId, TermId)> = match (l.len(), r.len()) {
                    (1, _) => r.iter().map(|value| (l[0], *value)).collect(),
                    (_, 1) => l.iter().map(|value| (*value, r[0])).collect(),
                    (a, b) if a == b => l.into_iter().zip(r).collect(),
                    _ => return Err(unsupported("operand shapes that do not combine")),
                };
                pairs
                    .into_iter()
                    .map(|(a, b)| apply(store, head, vec![a, b]))
                    .collect()
            }
            SemanticExprKind::Call { function, args } => {
                let args = args
                    .iter()
                    .map(|arg| self.scalar(store, *arg))
                    .collect::<Result<Vec<_>>>()?;
                let value = match (function.as_str(), args.as_slice()) {
                    ("sqrt", [x]) => apply(store, "Sqrt", vec![*x])?,
                    ("exp", [x]) => apply(store, "Exp", vec![*x])?,
                    ("ln" | "log", [x]) => apply(store, "Ln", vec![*x])?,
                    ("sin", [x]) => apply(store, "Sin", vec![*x])?,
                    ("cos", [x]) => apply(store, "Cos", vec![*x])?,
                    ("abs", [x]) => abs(store, *x)?,
                    ("min", [l, r]) => extremum(store, false, *l, *r)?,
                    ("max", [l, r]) => extremum(store, true, *l, *r)?,
                    _ => return Err(unsupported(format!("call `{function}`"))),
                };
                Ok(vec![value])
            }
            SemanticExprKind::Vector { elements } => elements
                .iter()
                .map(|element| self.scalar(store, *element))
                .collect(),
            other => Err(unsupported(format!("{other:?}"))),
        }
    }
}

fn scalar_operand(kernel: &StructuredKernel, operand: OperandId) -> bool {
    kernel
        .operands
        .get(operand.index())
        .is_some_and(|item| item.shape.iter().product::<usize>() <= 1)
}

fn output_operand(kernel: &StructuredKernel) -> Result<OperandId> {
    let writes = kernel
        .operands
        .iter()
        .enumerate()
        .filter(|(_, operand)| operand.access == AccessMode::Write)
        .map(|(index, _)| OperandId::new(index))
        .collect::<Vec<_>>();
    match writes.as_slice() {
        [single] if scalar_operand(kernel, *single) => Ok(*single),
        _ => Err(unsupported(format!(
            "kernel `{}` does not have exactly one scalar output",
            kernel.name
        ))),
    }
}

fn property_model(definition: &PropertyDefinition) -> Result<(SemanticModel, ExprId)> {
    let expr: &Expr = match &definition.model {
        PropertyModel::Expression(expr) => expr,
        PropertyModel::Constant(expr @ Expr::Number { unit: None, .. }) => expr,
        _ => {
            return Err(unsupported(
                "only expression and unitless constant property models have a symbolic source",
            ));
        }
    };
    let mut symbols = Vec::new();
    let mut expressions = Vec::new();
    let inputs = match &definition.model {
        PropertyModel::Expression(_) => definition.signature.inputs.as_slice(),
        _ => &[],
    };
    let value =
        lift_standalone_expr(expr, inputs, &mut symbols, &mut expressions).map_err(unsupported)?;
    let model = SemanticModel {
        ports: vec![],
        name: definition.signature.id.clone(),
        domains: vec![],
        regions: vec![],
        providers: vec![],
        symbols,
        expressions: expressions.into(),
        declarations: vec![],
        span: SourceSpan::default(),
    };
    Ok((model, value))
}

fn source_symbol(name: &str) -> SymbolName {
    SymbolName::new(SOURCE_NAMESPACE, name)
}

/// The source of a property's value kernel (operand `i` is signature input `i`).
pub fn property_value_sources(
    definition: &PropertyDefinition,
    kernel: &PropertyKernel,
    store: &mut TermStore,
) -> Result<KernelSources> {
    if span_independent_digest(definition) != kernel.definition_digest {
        return Err(error(
            "SYMBOLIC_PROVENANCE_MISMATCH",
            "property kernel was not lowered from this definition",
        ));
    }
    let (model, value) = property_model(definition)?;
    let mut projector = SourceProjector {
        model: &model,
        substitutions: BTreeMap::new(),
        inlining: Vec::new(),
        rounded_literals: 0,
    };
    let term = projector.scalar(store, value)?;
    let value_kernel = &kernel.module.kernels[kernel.value_kernel];
    let symbols = kernel
        .inputs
        .iter()
        .enumerate()
        .map(|(index, input)| ((OperandId::new(index), 0), source_symbol(&input.name)))
        .collect();
    Ok(KernelSources {
        symbols,
        outputs: BTreeMap::from([((output_operand(value_kernel)?, 0), term)]),
        rounded_literals: projector.rounded_literals,
    })
}

/// The source of a point expression's primal kernel (its first and only bundle).
pub fn point_expression_sources(
    module: &SemanticModule,
    kernels: &PointExpressionKernels,
    store: &mut TermStore,
) -> Result<KernelSources> {
    let model = module
        .models
        .iter()
        .find(|model| model.name == kernels.model)
        .ok_or_else(|| {
            error(
                "SYMBOLIC_PROVENANCE_MISMATCH",
                "point expression model is absent",
            )
        })?;
    let node = &kernels.root;
    let [bundle] = node.kernels.bundles.as_slice() else {
        return Err(unsupported("point expressions with several kernel bundles"));
    };
    let program = &node
        .factorization
        .integrals
        .get(bundle.integral_index)
        .ok_or_else(|| error("SYMBOLIC_PROVENANCE_MISMATCH", "bundle integral is absent"))?
        .primal;
    let primal = &bundle.module.kernels[bundle.primal_kernel_index];
    let mut symbols = SymbolMap::new();
    for binding in &bundle.primal_inputs {
        let Some(input) = program
            .inputs
            .iter()
            .find(|input| input.id == binding.input)
        else {
            continue;
        };
        if scalar_operand(primal, binding.operand) {
            let name = &model.symbols[input.binding.symbol.index()].name;
            symbols.insert((binding.operand, 0), source_symbol(name));
        }
    }
    let mut projector = SourceProjector {
        model,
        substitutions: BTreeMap::new(),
        inlining: Vec::new(),
        rounded_literals: 0,
    };
    let term = projector.scalar(store, kernels.expression)?;
    if !scalar_operand(primal, bundle.primal_output) {
        return Err(unsupported("a non-scalar point expression"));
    }
    Ok(KernelSources {
        symbols,
        outputs: BTreeMap::from([((bundle.primal_output, 0), term)]),
        rounded_literals: projector.rounded_literals,
    })
}

fn fv_symbol(name: &str) -> SymbolName {
    SymbolName::new(FV_NAMESPACE, name)
}

/// The source of a lowered FV kernel: physical flux `F·n`, characteristic speed `|∂(F·n)/∂u|`
/// (Resolvent derivative) or a generic face flux built from them. Authored fluxes, boundary,
/// source and limiter kernels refuse with `SYMBOLIC_SOURCE_UNSUPPORTED`.
pub fn fv_kernel_sources(
    module: &SemanticModule,
    system: &ConservationLawSystem,
    program: &ConservationLawProgram,
    kernel: &FvKernel,
    store: &mut TermStore,
) -> Result<KernelSources> {
    if program.system_digest != system.artifact_digest {
        return Err(error(
            "SYMBOLIC_PROVENANCE_MISMATCH",
            "program was not lowered from this conservation-law system",
        ));
    }
    let model = module
        .models
        .iter()
        .find(|model| model.name == system.model)
        .ok_or_else(|| error("SYMBOLIC_PROVENANCE_MISMATCH", "system model is absent"))?;
    let conserved = source_symbol(&model.symbols[system.method.conserved.index()].name);
    let mut symbols = SymbolMap::new();
    for (index, meaning) in kernel.operands.iter().enumerate() {
        let name = match meaning {
            FvOperandMeaning::State { side: None } => conserved.clone(),
            FvOperandMeaning::State {
                side: Some(malleus::FacetSide::Minus),
            } => fv_symbol("u_minus"),
            FvOperandMeaning::State {
                side: Some(malleus::FacetSide::Plus),
            } => fv_symbol("u_plus"),
            FvOperandMeaning::Normal { axis } => fv_symbol(&format!("n{axis}")),
            FvOperandMeaning::Datum { datum } => match &program.data[*datum].source {
                FvDatumSource::Symbol {
                    symbol,
                    component: 0,
                } => source_symbol(&model.symbols[symbol.index()].name),
                _ => continue,
            },
            FvOperandMeaning::SlopeRatio | FvOperandMeaning::Output => continue,
        };
        symbols.insert((OperandId::new(index), 0), name);
    }
    let mut projector = SourceProjector {
        model,
        substitutions: BTreeMap::new(),
        inlining: Vec::new(),
        rounded_literals: 0,
    };
    let components = projector.project(store, system.method.flux)?;
    if components.len() != usize::from(system.spatial_dimension) {
        return Err(unsupported(
            "a flux whose components do not match the dimension",
        ));
    }
    let mut products = Vec::new();
    for (axis, component) in components.into_iter().enumerate() {
        let normal = symbol(store, fv_symbol(&format!("n{axis}")))?;
        products.push(apply(store, "Mul", vec![component, normal])?);
    }
    let mut flux_normal = sum(store, products)?;
    if system.flux_sign < 0 {
        flux_normal = apply(store, "Neg", vec![flux_normal])?;
    }
    let speed = differentiate(store, flux_normal, &conserved)?;
    let at = |store: &mut TermStore, root: TermId, side: &str| {
        let value = symbol(store, fv_symbol(side))?;
        substitute(store, root, &[(conserved.clone(), value)])
    };
    let term = match kernel.role {
        FvKernelRole::PhysicalFluxNormal => flux_normal,
        FvKernelRole::CharacteristicSpeed => abs(store, speed)?,
        FvKernelRole::FaceFlux => {
            let fm = at(store, flux_normal, "u_minus")?;
            let fp = at(store, flux_normal, "u_plus")?;
            let um = symbol(store, fv_symbol("u_minus"))?;
            let up = symbol(store, fv_symbol("u_plus"))?;
            let half = rational(store, Rational::from_ratio(1, 2))?;
            let average = {
                let total = apply(store, "Add", vec![fm, fp])?;
                apply(store, "Mul", vec![half, total])?
            };
            match &program.intent.numerical_flux {
                NumericalFluxIntent::Central => average,
                NumericalFluxIntent::LocalLaxFriedrichs => {
                    let sm = at(store, speed, "u_minus")?;
                    let sp = at(store, speed, "u_plus")?;
                    let (am, ap) = (abs(store, sm)?, abs(store, sp)?);
                    let lambda = extremum(store, true, am, ap)?;
                    let jump = apply(store, "Sub", vec![up, um])?;
                    let dissipation = apply(store, "Mul", vec![half, lambda, jump])?;
                    apply(store, "Sub", vec![average, dissipation])?
                }
                NumericalFluxIntent::Hll => {
                    let sm = at(store, speed, "u_minus")?;
                    let sp = at(store, speed, "u_plus")?;
                    let smin = extremum(store, false, sm, sp)?;
                    let smax = extremum(store, true, sm, sp)?;
                    let a = apply(store, "Mul", vec![smax, fm])?;
                    let b = apply(store, "Mul", vec![smin, fp])?;
                    let jump = apply(store, "Sub", vec![up, um])?;
                    let c = apply(store, "Mul", vec![smin, smax, jump])?;
                    let difference = apply(store, "Sub", vec![a, b])?;
                    let numerator = apply(store, "Add", vec![difference, c])?;
                    let width = apply(store, "Sub", vec![smax, smin])?;
                    let middle = apply(store, "Div", vec![numerator, width])?;
                    let zero = integer(store, 0)?;
                    let left = relation(store, RelationOperator::GreaterEqual, smin, zero)?;
                    let right = relation(store, RelationOperator::LessEqual, smax, zero)?;
                    let inner = piecewise(store, right, fp, middle)?;
                    piecewise(store, left, fm, inner)?
                }
                NumericalFluxIntent::Authored { declaration, .. } => {
                    return Err(unsupported(format!(
                        "authored flux `{declaration}` (its claims are checked by L1 samples)"
                    )));
                }
            }
        }
        role => return Err(unsupported(format!("FV kernel role {role:?}"))),
    };
    Ok(KernelSources {
        symbols,
        outputs: BTreeMap::from([((output_operand(&kernel.kernel)?, 0), term)]),
        rounded_literals: projector.rounded_literals,
    })
}

// ---------------------------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------------------------

fn executable(kernel: &StructuredKernel) -> Result<Executable> {
    malleus::validate(kernel.clone())
        .map(Executable::reference)
        .map_err(|error| self::error("SYMBOLIC_EXPORT_FAILED", error))
}

fn export(kernel: &StructuredKernel) -> Result<SymbolicKernel> {
    export_symbolic(&executable(kernel)?)
        .map_err(|error| self::error("SYMBOLIC_EXPORT_FAILED", error))
}

/// Import `export` and show every source output equal to the exported one; the exported
/// outputs must be exactly the source's.
pub fn verify_export(
    store: &mut TermStore,
    export: &SymbolicKernel,
    sources: &KernelSources,
    domain: &SampleDomain,
) -> Result<(ImportStatistics, EquivalenceMethod)> {
    let imported = import_symbolic(export, &sources.symbols, store)?;
    verify_outputs(store, &imported, &sources.outputs, domain)
        .map(|method| (imported.statistics, method))
}

fn verify_outputs(
    store: &mut TermStore,
    imported: &ImportedKernel,
    expected: &BTreeMap<(OperandId, usize), TermId>,
    domain: &SampleDomain,
) -> Result<EquivalenceMethod> {
    let actual = imported
        .outputs
        .iter()
        .map(|output| ((output.operand, output.element), output.term))
        .collect::<BTreeMap<_, _>>();
    if actual.keys().ne(expected.keys()) {
        return Err(error(
            "SYMBOLIC_OUTPUT_MISMATCH",
            format!(
                "kernel `{}` stores {:?}, the source defines {:?}",
                imported.kernel,
                actual.keys().collect::<Vec<_>>(),
                expected.keys().collect::<Vec<_>>()
            ),
        ));
    }
    let mut method = EquivalenceMethod::CanonicalEquality;
    for (key, source) in expected {
        method = method.combine(check_equivalence(store, *source, actual[key], domain)?);
    }
    Ok(method)
}

/// A derivative kernel and its operand correspondence to the primal kernel.
struct DerivativeView<'a> {
    mode: DerivativeMode,
    kernel: &'a StructuredKernel,
    primal: Vec<DerivativeOperand>,
    independent: &'a [DerivativeOperand],
    dependent: &'a [DerivativeOperand],
}

fn check_derivative(
    store: &mut TermStore,
    view: &DerivativeView<'_>,
    sources: &KernelSources,
    domain: &SampleDomain,
) -> Result<EquivalenceMethod> {
    let name_of = |operand: OperandId| {
        sources.symbols.get(&(operand, 0)).cloned().ok_or_else(|| {
            error(
                "SYMBOLIC_INPUT_UNMAPPED",
                format!("operand {} has no symbol", operand.index()),
            )
        })
    };
    let mut symbols = SymbolMap::new();
    for operand in &view.primal {
        if let Some(name) = sources.symbols.get(&(operand.primal, 0)) {
            symbols.insert((operand.derivative, 0), name.clone());
        }
    }
    let mut expected = BTreeMap::new();
    match view.mode {
        DerivativeMode::Jvp => {
            let mut directions = Vec::new();
            for independent in view.independent {
                let variable = name_of(independent.primal)?;
                let direction = SymbolName::new(DIRECTION_NAMESPACE, variable.name.clone());
                symbols.insert((independent.derivative, 0), direction.clone());
                directions.push((variable, direction));
            }
            for dependent in view.dependent {
                let source = sources.outputs.get(&(dependent.primal, 0)).ok_or_else(|| {
                    error(
                        "SYMBOLIC_OUTPUT_MISMATCH",
                        "a dependent operand has no source",
                    )
                })?;
                let mut terms = Vec::new();
                for (variable, direction) in &directions {
                    let partial = differentiate(store, *source, variable)?;
                    let direction = symbol(store, direction.clone())?;
                    terms.push(apply(store, "Mul", vec![partial, direction])?);
                }
                expected.insert((dependent.derivative, 0), sum(store, terms)?);
            }
        }
        DerivativeMode::Vjp => {
            let mut cotangents = Vec::new();
            for dependent in view.dependent {
                let source = *sources.outputs.get(&(dependent.primal, 0)).ok_or_else(|| {
                    error(
                        "SYMBOLIC_OUTPUT_MISMATCH",
                        "a dependent operand has no source",
                    )
                })?;
                let cotangent = SymbolName::new(
                    COTANGENT_NAMESPACE,
                    format!("out{}", dependent.primal.index()),
                );
                symbols.insert((dependent.derivative, 0), cotangent.clone());
                cotangents.push((source, cotangent));
            }
            for independent in view.independent {
                let variable = name_of(independent.primal)?;
                let mut terms = Vec::new();
                if matches!(
                    view.kernel.operands[independent.derivative.index()].access,
                    AccessMode::ReadWrite | AccessMode::Reduce(_)
                ) {
                    let accumulator = SymbolName::new(ACCUMULATOR_NAMESPACE, variable.name.clone());
                    symbols.insert((independent.derivative, 0), accumulator.clone());
                    terms.push(symbol(store, accumulator)?);
                }
                for (source, cotangent) in &cotangents {
                    let partial = differentiate(store, *source, &variable)?;
                    let cotangent = symbol(store, cotangent.clone())?;
                    terms.push(apply(store, "Mul", vec![partial, cotangent])?);
                }
                expected.insert((independent.derivative, 0), sum(store, terms)?);
            }
        }
        mode => {
            return Err(unsupported(format!("derivative mode {mode:?}")));
        }
    }
    let imported = import_symbolic(&export(view.kernel)?, &symbols, store)?;
    verify_outputs(store, &imported, &expected, domain)
}

fn product_view(product: &malleus::DerivativeProduct) -> DerivativeView<'_> {
    DerivativeView {
        mode: product.mode,
        kernel: &product.kernel,
        primal: product.primal_operands.clone(),
        independent: &product.independent_operands,
        dependent: &product.dependent_operands,
    }
}

fn derivative_evidence(
    store: &mut TermStore,
    view: &DerivativeView<'_>,
    sources: &KernelSources,
    domain: &SampleDomain,
) -> Result<SymbolicDerivativeEvidence> {
    let independents = view
        .independent
        .iter()
        .map(|operand| {
            sources
                .symbols
                .get(&(operand.primal, 0))
                .map(|name| name.name.clone())
                .unwrap_or_else(|| format!("operand{}", operand.primal.index()))
        })
        .collect();
    Ok(SymbolicDerivativeEvidence {
        mode: view.mode,
        independents,
        outcome: CheckOutcome::Verified(check_derivative(store, view, sources, domain)?),
    })
}

/// JVP and VJP of `kernel` with respect to `independents`, through `malleus::differentiate`.
fn malleus_derivatives(
    store: &mut TermStore,
    kernel: &StructuredKernel,
    independents: &[OperandId],
    sources: &KernelSources,
    domain: &SampleDomain,
) -> Result<Vec<SymbolicDerivativeEvidence>> {
    let dependents = sources
        .outputs
        .keys()
        .map(|(operand, _)| *operand)
        .collect::<Vec<_>>();
    let mut evidence = Vec::new();
    for mode in [DerivativeMode::Jvp, DerivativeMode::Vjp] {
        let product = malleus::differentiate(
            kernel,
            &DerivativeRequest {
                mode,
                independent_operands: independents.to_vec(),
                dependent_operands: dependents.clone(),
            },
        )
        .map_err(|error| self::error("SYMBOLIC_DERIVATIVE_FAILED", format!("{error:?}")))?;
        evidence.push(derivative_evidence(
            store,
            &product_view(&product),
            sources,
            domain,
        )?);
    }
    Ok(evidence)
}

fn enclosure_boxes(names: &[SymbolName], domain: &SampleDomain) -> Result<Vec<Vec<(f64, f64)>>> {
    let whole = names
        .iter()
        .map(|name| {
            domain.range(name)?;
            Ok(domain.symbols.get(name).copied().unwrap_or(domain.default))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut boxes = vec![whole.clone()];
    let halves = |upper: bool| -> Option<Vec<(f64, f64)>> {
        whole
            .iter()
            .map(|(lo, hi)| {
                let middle = lo * 0.5 + hi * 0.5;
                let exact = Rational::from_f64(*lo)?
                    .add(&Rational::from_f64(*hi)?)
                    .mul(&Rational::from_ratio(1, 2));
                (Rational::from_f64(middle)? == exact).then_some(if upper {
                    (middle, *hi)
                } else {
                    (*lo, middle)
                })
            })
            .collect()
    };
    boxes.extend(halves(false));
    boxes.extend(halves(true));
    Ok(boxes)
}

fn run_enclosure(
    program: &IntervalProgram,
    kernel: &StructuredKernel,
    symbols: &SymbolMap,
    ranges: &BTreeMap<SymbolName, (f64, f64)>,
) -> Result<Vec<Vec<f64>>> {
    let mut buffers = Vec::with_capacity(2 * kernel.operands.len());
    for (index, operand) in kernel.operands.iter().enumerate() {
        let size = operand.shape.iter().product::<usize>().max(1);
        let (mut lo, mut hi) = (vec![0.0; size], vec![0.0; size]);
        for element in 0..size {
            if let Some(range) = symbols
                .get(&(OperandId::new(index), element))
                .and_then(|name| ranges.get(name))
            {
                lo[element] = range.0;
                hi[element] = range.1;
            }
        }
        buffers.push(lo);
        buffers.push(hi);
    }
    let mut bindings = buffers
        .iter_mut()
        .enumerate()
        .map(|(index, values)| BufferBinding::new(OperandId::new(index), values))
        .collect::<Vec<_>>();
    Interpreter::run(&program.executable, &mut bindings)
        .map_err(|error| self::error("SYMBOLIC_ENCLOSURE_FAILED", format!("{error:?}")))?;
    Ok(buffers)
}

fn contains(lo: f64, hi: f64, value: &Rational) -> bool {
    let above = lo == f64::NEG_INFINITY || Rational::from_f64(lo).is_some_and(|lo| &lo <= value);
    let below = hi == f64::INFINITY || Rational::from_f64(hi).is_some_and(|hi| value <= &hi);
    above && below
}

/// Run Malleus' naive, centered and affine enclosures of `kernel` on boxes of `domain` and
/// require every exactly evaluable sampled source value of each box inside them.
fn check_enclosures(
    store: &mut TermStore,
    kernel: &StructuredKernel,
    sources: &KernelSources,
    domain: &SampleDomain,
) -> Result<Vec<EnclosureEvidence>> {
    let roots = sources.outputs.values().copied().collect::<Vec<_>>();
    let names = free_symbols(store, &roots)?;
    let boxes = enclosure_boxes(&names, domain)?;
    let validated = malleus::validate(kernel.clone())
        .map_err(|error| self::error("SYMBOLIC_EXPORT_FAILED", error))?;
    let mut evidence = Vec::new();
    for lowering in [
        EnclosureLowering::Naive,
        EnclosureLowering::Centered,
        EnclosureLowering::Affine,
    ] {
        let program = match lowering {
            EnclosureLowering::Naive => lower_intervals(&validated),
            EnclosureLowering::Centered => lower_centered_intervals(&validated),
            EnclosureLowering::Affine => lower_affine_intervals(&validated),
        };
        let program = match program {
            Ok(program) if ieee_environment_supported() => program,
            Ok(_) => {
                evidence.push(EnclosureEvidence {
                    lowering,
                    outcome: EnclosureOutcome::Refused {
                        reason: "the host floating-point environment is not supported".into(),
                    },
                });
                continue;
            }
            Err(error) => {
                evidence.push(EnclosureEvidence {
                    lowering,
                    outcome: EnclosureOutcome::Refused {
                        reason: error.to_string(),
                    },
                });
                continue;
            }
        };
        let mut points = 0;
        for bounds in &boxes {
            let ranges = names.iter().cloned().zip(bounds.iter().copied()).collect();
            let buffers = run_enclosure(&program, kernel, &sources.symbols, &ranges)?;
            let exact_bounds = bounds
                .iter()
                .map(|(lo, hi)| Ok((exact_rational_of(*lo)?, exact_rational_of(*hi)?)))
                .collect::<Result<Vec<_>>>()?;
            let mut environments = Vec::new();
            if names.len() <= 4 {
                for corner in 0..(1usize << names.len()) {
                    environments.push(
                        names
                            .iter()
                            .zip(&exact_bounds)
                            .enumerate()
                            .map(|(axis, (name, (lo, hi)))| {
                                let value = if corner >> axis & 1 == 1 { hi } else { lo };
                                (name.clone(), value.clone())
                            })
                            .collect::<BTreeMap<_, _>>(),
                    );
                }
            }
            for index in 1..=domain.samples as i64 {
                environments.push(point(&names, &exact_bounds, index)?);
            }
            for environment in &environments {
                for ((operand, element), source) in &sources.outputs {
                    let Ok(value) = evaluate(store, environment, *source) else {
                        continue;
                    };
                    let [lo_id, hi_id] = IntervalProgram::endpoints(*operand);
                    let (lo, hi) = (
                        buffers[lo_id.index()][*element],
                        buffers[hi_id.index()][*element],
                    );
                    if !contains(lo, hi, &value) {
                        return Err(error(
                            "SYMBOLIC_ENCLOSURE_VIOLATED",
                            format!(
                                "{lowering:?} enclosure [{lo}, {hi}] of `{}` misses {} at {}",
                                kernel.name,
                                value.to_f64_lossy(),
                                describe(environment)
                            ),
                        ));
                    }
                    points += 1;
                }
            }
        }
        evidence.push(EnclosureEvidence {
            lowering,
            outcome: EnclosureOutcome::Contained {
                boxes: boxes.len(),
                points,
            },
        });
    }
    Ok(evidence)
}

fn entry(
    store: &mut TermStore,
    kernel: &StructuredKernel,
    provenance: KernelProvenance,
    sources: &KernelSources,
    domain: &SampleDomain,
    derivatives: Vec<SymbolicDerivativeEvidence>,
) -> Result<SymbolicVerificationEntry> {
    let exported = export(kernel)?;
    let (import, round_trip) = verify_export(store, &exported, sources, domain)?;
    Ok(SymbolicVerificationEntry {
        kernel: exported.kernel.clone(),
        kernel_digest: exported.kernel_digest.clone(),
        provenance,
        outputs: sources.outputs.len(),
        rounded_literals: sources.rounded_literals,
        import,
        round_trip,
        derivatives,
        enclosures: check_enclosures(store, kernel, sources, domain)?,
    })
}

/// Verify a property's value kernel (round trip, Malleus JVP/VJP against Resolvent's
/// derivative, enclosures) and every tangent kernel against Resolvent's derivative.
pub fn verify_property_kernel(
    definition: &PropertyDefinition,
    kernel: &PropertyKernel,
    domain: &SampleDomain,
) -> Result<Vec<SymbolicVerificationEntry>> {
    let mut store = TermStore::new().map_err(term_error)?;
    let sources = property_value_sources(definition, kernel, &mut store)?;
    let value_kernel = &kernel.module.kernels[kernel.value_kernel];
    let independents = (0..kernel.inputs.len())
        .map(OperandId::new)
        .collect::<Vec<_>>();
    let derivatives = if independents.is_empty() {
        Vec::new()
    } else {
        malleus_derivatives(&mut store, value_kernel, &independents, &sources, domain)?
    };
    let property = definition.signature.id.clone();
    let mut entries = vec![entry(
        &mut store,
        value_kernel,
        KernelProvenance::PropertyValue {
            property: property.clone(),
            identity: kernel.identity.clone(),
        },
        &sources,
        domain,
        derivatives,
    )?];
    let (&key, &value) = sources
        .outputs
        .iter()
        .next()
        .expect("a property has one value output");
    for tangent in &kernel.tangents {
        let tangent_kernel = &kernel.module.kernels[tangent.kernel];
        let derivative = differentiate(&mut store, value, &source_symbol(&tangent.input))?;
        let mut tangent_sources = sources.clone();
        tangent_sources.outputs =
            BTreeMap::from([((output_operand(tangent_kernel)?, key.1), derivative)]);
        entries.push(entry(
            &mut store,
            tangent_kernel,
            KernelProvenance::PropertyTangent {
                property: property.clone(),
                input: tangent.input.clone(),
                identity: kernel.identity.clone(),
            },
            &tangent_sources,
            domain,
            Vec::new(),
        )?);
    }
    Ok(entries)
}

/// Verify a point expression's primal kernel and its bundle's own JVP, VJP and parameter
/// kernels (primal operands keep their identifiers in a bundle's derivative kernels).
pub fn verify_point_expression(
    module: &SemanticModule,
    kernels: &PointExpressionKernels,
    domain: &SampleDomain,
) -> Result<SymbolicVerificationEntry> {
    let mut store = TermStore::new().map_err(term_error)?;
    let sources = point_expression_sources(module, kernels, &mut store)?;
    let bundle = &kernels.root.kernels.bundles[0];
    let primal = &bundle.module.kernels[bundle.primal_kernel_index];
    let identity = (0..primal.operands.len())
        .map(|index| DerivativeOperand {
            primal: OperandId::new(index),
            derivative: OperandId::new(index),
        })
        .collect::<Vec<_>>();
    let mut derivatives = Vec::new();
    for contract in [&bundle.jvp, &bundle.vjp, &bundle.parameter] {
        if contract.independent_operands.is_empty() {
            continue;
        }
        let view = DerivativeView {
            mode: contract.mode,
            kernel: &bundle.module.kernels[contract.kernel_index],
            primal: identity.clone(),
            independent: &contract.independent_operands,
            dependent: &contract.dependent_operands,
        };
        derivatives.push(derivative_evidence(&mut store, &view, &sources, domain)?);
    }
    entry(
        &mut store,
        primal,
        KernelProvenance::PointExpression {
            model: kernels.model.clone(),
            expression: kernels.expression,
            artifact: kernels.artifact_digest.clone(),
        },
        &sources,
        domain,
        derivatives,
    )
}

/// Verify the physical flux, characteristic speed and generic face flux of an FV program.
/// Face-flux JVPs go through [`FvKernel::differentiate`]; its D6 refusals are recorded.
pub fn verify_conservation_program(
    module: &SemanticModule,
    system: &ConservationLawSystem,
    program: &ConservationLawProgram,
    domain: &SampleDomain,
) -> Result<Vec<SymbolicVerificationEntry>> {
    let mut store = TermStore::new().map_err(term_error)?;
    let mut entries = Vec::new();
    for kernel in [
        &program.physical_flux_normal,
        &program.characteristic_speed,
        &program.face_flux,
    ] {
        let sources = fv_kernel_sources(module, system, program, kernel, &mut store)?;
        let output = output_operand(&kernel.kernel)?;
        let mut derivatives = Vec::new();
        if matches!(
            kernel.role,
            FvKernelRole::PhysicalFluxNormal | FvKernelRole::FaceFlux
        ) {
            let states = kernel
                .operands
                .iter()
                .enumerate()
                .filter(|(_, meaning)| matches!(meaning, FvOperandMeaning::State { .. }))
                .map(|(index, _)| OperandId::new(index))
                .collect::<Vec<_>>();
            let request = DerivativeRequest {
                mode: DerivativeMode::Jvp,
                independent_operands: states.clone(),
                dependent_operands: vec![output],
            };
            match kernel.differentiate(&request) {
                Ok(product) => derivatives.push(derivative_evidence(
                    &mut store,
                    &product_view(&product),
                    &sources,
                    domain,
                )?),
                Err(refusal) => derivatives.push(SymbolicDerivativeEvidence {
                    mode: DerivativeMode::Jvp,
                    independents: states
                        .iter()
                        .filter_map(|operand| sources.symbols.get(&(*operand, 0)))
                        .map(|name| name.name.clone())
                        .collect(),
                    outcome: CheckOutcome::Refused {
                        code: refusal.code.into(),
                        message: refusal.message,
                    },
                }),
            }
        }
        entries.push(entry(
            &mut store,
            &kernel.kernel,
            KernelProvenance::FiniteVolume {
                role: kernel.role,
                program: program.artifact_digest.clone(),
            },
            &sources,
            domain,
            derivatives,
        )?);
    }
    Ok(entries)
}
