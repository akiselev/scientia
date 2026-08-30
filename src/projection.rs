//! GX-A2 projection (contract C8): Scientia's own bridge from the typed semantic arena into
//! Resolvent's RV1-C2/C3 consumer-neutral term algebra, and back.
//!
//! This replaces the old `algebra.rs` bridge, which projected the *parser-owned*
//! `scientific::Expr` tree directly, silently zeroed comparisons instead of refusing them
//! (STATUS.md's recorded limit), and rounded every literal through `f64` before handing it to
//! Resolvent. This module instead:
//!
//! - projects the *typed* semantic arena (`SemanticModel` + `ExprId`), so refusals can name a
//!   scientific construct rather than a bare parser node;
//! - ingests numeric literals via [`crate::semantic::ExactLiteral`] (contract C8), never through
//!   `f64` rounding;
//! - re-embeds a projection's result under fresh Scientia expression ids in an *owned arena
//!   extension* returned to the caller (contract C8's `AlgebraOutcome::result` doc comment),
//!   since `SemanticModel::expressions` is an immutable `Arc<[SemanticExpr]>` and cannot be
//!   mutated in place. Concretely: [`AlgebraOutcome::arena`] is `model.expressions` followed by
//!   every freshly synthesized node, and `AlgebraOutcome::result` indexes into that combined
//!   arena, not into `model.expressions` alone.
//!
//! ## Scope concretizations (recorded here, not silently)
//!
//! - `AlgebraOperation::Differentiate` differentiates with respect to a [`SymbolId`], never a
//!   bare name. A manufactured-solution expression's `x`/`y`/`z`/`t` coordinate names are given
//!   real (synthesized) symbols by [`lift_standalone_expr`] specifically so every differentiation
//!   target in this module is a `SymbolId`.
//! - `AlgebraOperation::ManufacturedForcing`'s `exact: BTreeMap<SymbolId, ExprId>` ExprIds are
//!   assumed to live in the *same* arena as `equation` (i.e. `model.expressions`, or an
//!   extension of it the caller built with [`lift_standalone_exprs`] before calling `project`).
//!   `project` does not accept a second arena parameter, so this is the only coherent reading.
//! - Non-integer rational coefficients that Resolvent's own `Differentiate`/`Simplify` produce
//!   (e.g. the `1/2` from a `sqrt` derivative) come back with `spelling not preserved` per
//!   Resolvent's own documented RV1-C2 limitation; since [`crate::semantic::ExactLiteral`] has
//!   no general-rational variant, this module reconstructs such a coefficient as a decimal
//!   literal from its nearest `f64` rather than as an exact fraction. Literal *ingestion* (every
//!   literal actually written in `.res` source, and every exact-decimal-lexeme constant this
//!   module itself emits) stays fully exact; only a rational *produced* by differentiation can
//!   degrade this way.
//! - `ManufacturedForcing` expands `Differential::{TimeDerivative, Gradient, Divergence}` of a
//!   fully-substituted closed-form expression in `x`/`y`/`z`/`t`. `Curl`, `RotatedGradient`, and
//!   `SymmetricGradient` are refused (`PROPERTY_KERNEL_UNSUPPORTED`) rather than approximated;
//!   none of the frozen contract's worked examples (03/04/06/17/25) need them.

use crate::scientific::{BinaryOp, Expr, PropertyInput, ScientificError, UnaryOp};
use crate::semantic::{
    DeclarationId, DifferentialOperator, ExactLiteral, ExprId, Frame, SemanticDeclarationKind,
    SemanticExpr, SemanticExprKind, SemanticModel, SemanticRole, SemanticShape, SemanticSymbol,
    SemanticType, SymbolId,
};
use crate::source::SourceSpan;
use quantitas::Dimension;
use resolvent::{
    AlgebraBudget, Atom, OPERATOR_NAMESPACE, SymbolName, TermAlgebraBudgetReport, TermAlgebraError,
    TermAlgebraOperation, TermAlgebraReceipt, TermAlgebraRefusal as ResolventRefusal, TermBudget,
    TermId, TermNode, TermStore,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use thiserror::Error;

/// A requested Scientia-level algebra operation (GX-A2, contract C8).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AlgebraOperation {
    Differentiate {
        wrt: SymbolId,
    },
    Simplify,
    Substitute {
        bindings: BTreeMap<SymbolId, ExprId>,
    },
    /// Derive the forcing term for a manufactured-solution case binding (contract C3.2): every
    /// field symbol in `exact` is replaced by its exact-solution expression throughout
    /// `equation`'s strong form (`lhs - rhs`), and every field-operator (`grad`/`div`/`dt`)
    /// applied to a now-substituted closed-form expression is expanded via partial
    /// differentiation through this same projection.
    ManufacturedForcing {
        equation: DeclarationId,
        exact: BTreeMap<SymbolId, ExprId>,
    },
}

/// Result of one [`AlgebraOperation`] (GX-A2, contract C8).
#[derive(Clone, Debug, PartialEq)]
pub struct AlgebraOutcome {
    /// `model.expressions` followed by every freshly re-embedded node; `result` indexes into
    /// this combined arena, not into the caller's original arena alone.
    pub arena: Arc<[SemanticExpr]>,
    pub result: ExprId,
    pub assumptions: Vec<String>,
    pub budget: TermAlgebraBudgetReport,
    pub receipt: TermAlgebraReceipt,
}

/// A projection request outside its operation's contract, or a Scientia-side refusal for a
/// construct the projection never hands to Resolvent (units-unresolved, non-scalar, or a field
/// operator this module does not expand).
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum AlgebraRefusal {
    #[error("ALGEBRA_OUTSIDE_OPERATION: {construct}")]
    OutsideOperation { construct: String },
    #[error("PROPERTY_KERNEL_UNSUPPORTED: {0}")]
    Unsupported(String),
}

impl AlgebraRefusal {
    pub const fn code(&self) -> &'static str {
        match self {
            Self::OutsideOperation { .. } => "ALGEBRA_OUTSIDE_OPERATION",
            Self::Unsupported(_) => "PROPERTY_KERNEL_UNSUPPORTED",
        }
    }
}

fn term_budget() -> TermBudget {
    TermBudget::default()
}
fn algebra_budget() -> AlgebraBudget {
    AlgebraBudget::default()
}

/// Apply one [`AlgebraOperation`] to `expr` within `model`'s arena.
pub fn project(
    model: &SemanticModel,
    expr: ExprId,
    operation: &AlgebraOperation,
) -> Result<AlgebraOutcome, AlgebraRefusal> {
    match operation {
        AlgebraOperation::Differentiate { wrt } => project_root_operation(
            model,
            expr,
            TermAlgebraOperation::Differentiate {
                variable: symbol_name(model, *wrt),
            },
        ),
        AlgebraOperation::Simplify => {
            project_root_operation(model, expr, TermAlgebraOperation::Simplify)
        }
        AlgebraOperation::Substitute { bindings } => project_substitute(model, expr, bindings),
        AlgebraOperation::ManufacturedForcing { equation, exact } => {
            project_manufactured_forcing(model, *equation, exact)
        }
    }
}

fn project_root_operation(
    model: &SemanticModel,
    expr: ExprId,
    operation: TermAlgebraOperation,
) -> Result<AlgebraOutcome, AlgebraRefusal> {
    let mut store = TermStore::new().map_err(store_error)?;
    let root = to_term(model, expr, &mut store)?;
    let outcome = store
        .apply_algebra_operation(root, &operation, term_budget(), algebra_budget())
        .map_err(operation_error)?;
    reembed(model, &store, outcome, span_of(model, expr))
}

fn project_substitute(
    model: &SemanticModel,
    expr: ExprId,
    bindings: &BTreeMap<SymbolId, ExprId>,
) -> Result<AlgebraOutcome, AlgebraRefusal> {
    let mut store = TermStore::new().map_err(store_error)?;
    let root = to_term(model, expr, &mut store)?;
    let mut resolved = Vec::with_capacity(bindings.len());
    for (symbol, replacement) in bindings {
        let replacement_term = to_term(model, *replacement, &mut store)?;
        resolved.push((symbol_name(model, *symbol), replacement_term));
    }
    let outcome = store
        .apply_algebra_operation(
            root,
            &TermAlgebraOperation::Substitute { bindings: resolved },
            term_budget(),
            algebra_budget(),
        )
        .map_err(operation_error)?;
    reembed(model, &store, outcome, span_of(model, expr))
}

fn project_manufactured_forcing(
    model: &SemanticModel,
    equation: DeclarationId,
    exact: &BTreeMap<SymbolId, ExprId>,
) -> Result<AlgebraOutcome, AlgebraRefusal> {
    let declaration = model
        .declarations
        .iter()
        .find(|declaration| declaration.id == equation)
        .ok_or_else(|| {
            AlgebraRefusal::Unsupported(format!("declaration {equation:?} is not in the arena"))
        })?;
    let SemanticDeclarationKind::Equation { lhs, rhs } = &declaration.kind else {
        return Err(AlgebraRefusal::Unsupported(format!(
            "declaration `{}` is not an equation",
            declaration.name
        )));
    };
    let span = declaration.span;
    let mut extension = Vec::new();
    let base = model.expressions.len();
    let expanded_lhs = expand_manufactured(model, *lhs, exact, &mut extension, base)?;
    let expanded_rhs = expand_manufactured(model, *rhs, exact, &mut extension, base)?;
    let result = push(
        &mut extension,
        base,
        SemanticExprKind::Binary {
            op: BinaryOp::Sub,
            lhs: expanded_lhs,
            rhs: expanded_rhs,
        },
        scalar_dimensionless(SemanticRole::Intrinsic),
        span,
    )?;
    let mut arena = model.expressions.to_vec();
    arena.extend(extension);
    Ok(AlgebraOutcome {
        arena: arena.into(),
        result,
        assumptions: vec![
            "manufactured forcing substitutes the exact solution for every field symbol in the \
             equation's strong form and expands dt/grad/div of the substituted closed form \
             symbolically; every other symbol (parameters, sources, properties, providers) is \
             left as a symbolic reference"
                .into(),
        ],
        budget: TermAlgebraBudgetReport {
            term_budget: term_budget(),
            algebra_budget: algebra_budget(),
            expression_nodes_used: 0,
            output: resolvent::TermStats {
                unique_nodes: 0,
                edge_references: 0,
                max_depth: 0,
                shared_nodes: 0,
            },
        },
        receipt: manufactured_forcing_receipt(&declaration.name, &result, model),
    })
}

/// Recursively substitute every field symbol in `exact` with its exact-solution expression, and
/// expand every `dt`/`grad`/`div` differential operator whose (substituted) argument is a
/// closed-form expression in `x`/`y`/`z`/`t` via [`AlgebraOperation::Differentiate`]. Every id
/// this returns lives in the *combined* arena `model.expressions ++ extension`.
fn expand_manufactured(
    model: &SemanticModel,
    id: ExprId,
    exact: &BTreeMap<SymbolId, ExprId>,
    extension: &mut Vec<SemanticExpr>,
    base: usize,
) -> Result<ExprId, AlgebraRefusal> {
    let combined_len = base + extension.len();
    let expression = if id.index() < model.expressions.len() {
        model.expressions[id.index()].clone()
    } else if id.index() < combined_len {
        extension[id.index() - base].clone()
    } else {
        return Err(AlgebraRefusal::Unsupported(format!(
            "expression {id} is outside the combined arena"
        )));
    };
    let span = expression.span;
    match &expression.kind {
        SemanticExprKind::Symbol { symbol } => {
            if let Some(exact_id) = exact.get(symbol) {
                Ok(*exact_id)
            } else {
                Ok(id)
            }
        }
        SemanticExprKind::Number { .. } | SemanticExprKind::String { .. } => Ok(id),
        SemanticExprKind::Unary { op, arg } => {
            let arg = expand_manufactured(model, *arg, exact, extension, base)?;
            push(
                extension,
                base,
                SemanticExprKind::Unary { op: *op, arg },
                expression.ty.clone(),
                span,
            )
        }
        SemanticExprKind::Binary { op, lhs, rhs } => {
            let lhs = expand_manufactured(model, *lhs, exact, extension, base)?;
            let rhs = expand_manufactured(model, *rhs, exact, extension, base)?;
            push(
                extension,
                base,
                SemanticExprKind::Binary { op: *op, lhs, rhs },
                expression.ty.clone(),
                span,
            )
        }
        SemanticExprKind::Call { function, args } => {
            let args = args
                .iter()
                .map(|arg| expand_manufactured(model, *arg, exact, extension, base))
                .collect::<Result<Vec<_>, _>>()?;
            push(
                extension,
                base,
                SemanticExprKind::Call {
                    function: function.clone(),
                    args,
                },
                expression.ty.clone(),
                span,
            )
        }
        SemanticExprKind::ProviderCall { provider, args } => {
            let args = args
                .iter()
                .map(|arg| expand_manufactured(model, *arg, exact, extension, base))
                .collect::<Result<Vec<_>, _>>()?;
            push(
                extension,
                base,
                SemanticExprKind::ProviderCall {
                    provider: *provider,
                    args,
                },
                expression.ty.clone(),
                span,
            )
        }
        SemanticExprKind::Differential { operator, arg } => {
            let expanded_arg = expand_manufactured(model, *arg, exact, extension, base)?;
            expand_differential(model, *operator, expanded_arg, extension, base, span)
        }
        other => Err(AlgebraRefusal::Unsupported(format!(
            "manufactured forcing cannot expand a `{}` construct",
            construct_label(other)
        ))),
    }
}

fn expand_differential(
    model: &SemanticModel,
    operator: DifferentialOperator,
    arg: ExprId,
    extension: &mut Vec<SemanticExpr>,
    base: usize,
    span: SourceSpan,
) -> Result<ExprId, AlgebraRefusal> {
    match operator {
        DifferentialOperator::TimeDerivative => {
            differentiate_combined(model, arg, "t", extension, base)
        }
        DifferentialOperator::Gradient => {
            let dimension = spatial_dimension(model);
            let mut components = Vec::with_capacity(dimension);
            for coordinate in ["x", "y", "z"].into_iter().take(dimension) {
                components.push(differentiate_combined(
                    model, arg, coordinate, extension, base,
                )?);
            }
            push(
                extension,
                base,
                SemanticExprKind::Vector {
                    elements: components,
                },
                scalar_dimensionless(SemanticRole::Intrinsic),
                span,
            )
        }
        DifferentialOperator::Divergence => {
            let combined_len = base + extension.len();
            let elements = if arg.index() < model.expressions.len() {
                match &model.expressions[arg.index()].kind {
                    SemanticExprKind::Vector { elements } => Some(elements.clone()),
                    _ => None,
                }
            } else if arg.index() < combined_len {
                match &extension[arg.index() - base].kind {
                    SemanticExprKind::Vector { elements } => Some(elements.clone()),
                    _ => None,
                }
            } else {
                None
            };
            let Some(elements) = elements else {
                return Err(AlgebraRefusal::Unsupported(
                    "manufactured forcing expands `div` only of an explicit vector-valued exact \
                     expression (e.g. a substituted `grad(...)`), which this reduced projection \
                     does not otherwise decompose"
                        .into(),
                ));
            };
            let mut sum = None;
            for (coordinate, component) in ["x", "y", "z"].into_iter().zip(elements) {
                let term = differentiate_combined(model, component, coordinate, extension, base)?;
                sum = Some(match sum {
                    None => term,
                    Some(accumulated) => push(
                        extension,
                        base,
                        SemanticExprKind::Binary {
                            op: BinaryOp::Add,
                            lhs: accumulated,
                            rhs: term,
                        },
                        scalar_dimensionless(SemanticRole::Intrinsic),
                        span,
                    )?,
                });
            }
            sum.ok_or_else(|| {
                AlgebraRefusal::Unsupported("divergence of a zero-dimensional vector".into())
            })
        }
        DifferentialOperator::Curl
        | DifferentialOperator::RotatedGradient
        | DifferentialOperator::SymmetricGradient => Err(AlgebraRefusal::Unsupported(format!(
            "manufactured forcing does not expand `{operator:?}`"
        ))),
    }
}

/// `ManufacturedForcing` is a composite of one substitution pass and several field-operator
/// differentiations, not one Resolvent operation, so it has no single `TermStore` digest pair;
/// this receipt instead digests the equation name and the resulting expression identity so it
/// still satisfies `TermAlgebraReceipt`'s wire shape (64 lowercase hex digits per digest).
fn manufactured_forcing_receipt(
    name: &str,
    result: &ExprId,
    model: &SemanticModel,
) -> TermAlgebraReceipt {
    #[derive(serde::Serialize)]
    struct Input<'a> {
        model: &'a str,
        equation: &'a str,
    }
    #[derive(serde::Serialize)]
    struct Output<'a> {
        model: &'a str,
        equation: &'a str,
        result: u32,
    }
    TermAlgebraReceipt {
        schema: "resolvent-term-algebra-receipt/1".into(),
        operation: resolvent::TermAlgebraReceiptOperation::Differentiate {
            namespace: "Scientia::ManufacturedForcing".into(),
            name: name.into(),
        },
        input_digest: crate::id::span_independent_digest(&Input {
            model: &model.name,
            equation: name,
        })
        .hex,
        output_digest: crate::id::span_independent_digest(&Output {
            model: &model.name,
            equation: name,
            result: result.0,
        })
        .hex,
    }
}

fn spatial_dimension(model: &SemanticModel) -> usize {
    model
        .domains
        .first()
        .map(|domain| domain.spatial_dimension as usize)
        .unwrap_or(3)
        .min(3)
}

/// Differentiate the expression at `arg` (in the combined `model.expressions ++ extension`
/// arena) with respect to the coordinate symbol named `coordinate_name`, appending the result to
/// `extension`. Refuses if `arg`'s free symbols are not all coordinate/declared symbols already
/// present in `model.symbols` (i.e. `arg` still contains an un-substituted field reference).
fn differentiate_combined(
    model: &SemanticModel,
    arg: ExprId,
    coordinate_name: &str,
    extension: &mut Vec<SemanticExpr>,
    _base: usize,
) -> Result<ExprId, AlgebraRefusal> {
    let Some(coordinate) = model
        .symbols
        .iter()
        .find(|symbol| symbol.name == coordinate_name)
        .map(|symbol| symbol.id)
    else {
        return Err(AlgebraRefusal::Unsupported(format!(
            "manufactured forcing needs a `{coordinate_name}` coordinate symbol; lift the exact \
             solution with `lift_standalone_exprs` before calling `project`"
        )));
    };
    let combined: Vec<SemanticExpr> = model
        .expressions
        .iter()
        .cloned()
        .chain(extension.iter().cloned())
        .collect();
    let combined_model = SemanticModel {
        expressions: combined.into(),
        ..model.clone()
    };
    let outcome = project(
        &combined_model,
        arg,
        &AlgebraOperation::Differentiate { wrt: coordinate },
    )?;
    // `outcome.arena` is `combined_model.expressions` (== `model.expressions ++ extension` as
    // they stood on entry) followed by whatever this differentiation freshly synthesized;
    // re-append only that new suffix.
    let grown = outcome.arena.len() - combined_model.expressions.len();
    let new_nodes = &outcome.arena[outcome.arena.len() - grown..];
    extension.extend_from_slice(new_nodes);
    Ok(outcome.result)
}

fn reembed(
    model: &SemanticModel,
    store: &TermStore,
    outcome: resolvent::TermAlgebraOutcome,
    span: SourceSpan,
) -> Result<AlgebraOutcome, AlgebraRefusal> {
    let mut extension = Vec::new();
    let base = model.expressions.len();
    let result = from_term(model, store, outcome.root, span, &mut extension, base)?;
    let mut arena = model.expressions.to_vec();
    arena.extend(extension);
    Ok(AlgebraOutcome {
        arena: arena.into(),
        result,
        assumptions: outcome.assumptions,
        budget: outcome.budget,
        receipt: outcome.receipt,
    })
}

fn store_error(error: resolvent::TermError) -> AlgebraRefusal {
    AlgebraRefusal::Unsupported(error.to_string())
}

fn operation_error(error: TermAlgebraError) -> AlgebraRefusal {
    match error {
        TermAlgebraError::Refused(ResolventRefusal::OutsideOperation { construct }) => {
            AlgebraRefusal::OutsideOperation { construct }
        }
        other => AlgebraRefusal::Unsupported(other.to_string()),
    }
}

fn symbol_name(model: &SemanticModel, symbol: SymbolId) -> SymbolName {
    SymbolName::new(
        "Scientia",
        model
            .symbols
            .get(symbol.index())
            .map(|s| s.name.clone())
            .unwrap_or_else(|| format!("symbol_{}", symbol.0)),
    )
}

fn span_of(model: &SemanticModel, id: ExprId) -> SourceSpan {
    model
        .expressions
        .get(id.index())
        .map(|e| e.span)
        .unwrap_or_default()
}

// ---------------- Scientia expression -> Resolvent term ----------------

fn to_term(
    model: &SemanticModel,
    id: ExprId,
    store: &mut TermStore,
) -> Result<TermId, AlgebraRefusal> {
    let expression = model.expressions.get(id.index()).ok_or_else(|| {
        AlgebraRefusal::Unsupported(format!("expression {id} is outside the arena"))
    })?;
    match &expression.kind {
        SemanticExprKind::Number { exact, .. } => exact_term(store, exact),
        SemanticExprKind::Symbol { symbol } => store
            .atom(Atom::Symbol(symbol_name(model, *symbol)), term_budget())
            .map_err(store_error),
        SemanticExprKind::Unary {
            op: UnaryOp::Neg,
            arg,
        } => {
            let arg_term = to_term(model, *arg, store)?;
            apply(store, "Neg", vec![arg_term])
        }
        SemanticExprKind::Binary { op, lhs, rhs } => {
            let lhs_term = to_term(model, *lhs, store)?;
            let rhs_term = to_term(model, *rhs, store)?;
            let name = match op {
                BinaryOp::Add => "Add",
                BinaryOp::Sub => "Sub",
                BinaryOp::Mul => "Mul",
                BinaryOp::Div => "Div",
                BinaryOp::Pow => "Pow",
                BinaryOp::Eq | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                    return Err(AlgebraRefusal::OutsideOperation {
                        construct: "comparison".into(),
                    });
                }
            };
            apply(store, name, vec![lhs_term, rhs_term])
        }
        SemanticExprKind::Call { function, args } => {
            let Some(name) = intrinsic_operator_name(function) else {
                return Err(AlgebraRefusal::OutsideOperation {
                    construct: format!("Call[{function}]"),
                });
            };
            let terms = args
                .iter()
                .map(|arg| to_term(model, *arg, store))
                .collect::<Result<Vec<_>, _>>()?;
            apply(store, name, terms)
        }
        SemanticExprKind::ProviderCall { .. } => Err(AlgebraRefusal::OutsideOperation {
            construct: "ProviderCall".into(),
        }),
        other => Err(AlgebraRefusal::OutsideOperation {
            construct: construct_label(other),
        }),
    }
}

fn intrinsic_operator_name(function: &str) -> Option<&'static str> {
    Some(match function {
        "sin" => "Sin",
        "cos" => "Cos",
        "exp" => "Exp",
        "log" | "ln" => "Ln",
        "sqrt" => "Sqrt",
        _ => return None,
    })
}

fn construct_label(kind: &SemanticExprKind) -> String {
    match kind {
        SemanticExprKind::Number { .. } => "Number",
        SemanticExprKind::String { .. } => "String",
        SemanticExprKind::Symbol { .. } => "Symbol",
        SemanticExprKind::Unary { .. } => "Unary",
        SemanticExprKind::Binary { .. } => "Binary",
        SemanticExprKind::Call { .. } => "Call",
        SemanticExprKind::Differential { .. } => "Differential",
        SemanticExprKind::Contraction { .. } => "Contraction",
        SemanticExprKind::TensorTrace { .. } => "TensorTrace",
        SemanticExprKind::FacetTrace { .. } => "FacetTrace",
        SemanticExprKind::Jump { .. } => "Jump",
        SemanticExprKind::Average { .. } => "Average",
        SemanticExprKind::Conjugate { .. } => "Conjugate",
        SemanticExprKind::NormalComponent { .. } => "NormalComponent",
        SemanticExprKind::Index { .. } => "Index",
        SemanticExprKind::Vector { .. } => "Vector",
        SemanticExprKind::ProviderCall { .. } => "ProviderCall",
    }
    .into()
}

fn exact_term(store: &mut TermStore, exact: &ExactLiteral) -> Result<TermId, AlgebraRefusal> {
    match exact {
        ExactLiteral::Integer(value) => store
            .exact_integer(*value, term_budget())
            .map_err(store_error),
        ExactLiteral::Decimal { mantissa, exponent } => store
            .exact_decimal(*mantissa, *exponent, term_budget())
            .map_err(store_error),
    }
}

fn apply(
    store: &mut TermStore,
    head_name: &str,
    arguments: Vec<TermId>,
) -> Result<TermId, AlgebraRefusal> {
    let head = store
        .atom(
            Atom::Symbol(SymbolName::new(OPERATOR_NAMESPACE, head_name)),
            term_budget(),
        )
        .map_err(store_error)?;
    store
        .intern(TermNode::Apply { head, arguments }, term_budget())
        .map_err(store_error)
}

// ---------------- Resolvent term -> Scientia expression (re-embedding) ----------------

fn from_term(
    model: &SemanticModel,
    store: &TermStore,
    term: TermId,
    span: SourceSpan,
    extension: &mut Vec<SemanticExpr>,
    base: usize,
) -> Result<ExprId, AlgebraRefusal> {
    let node = store.node(term).map_err(store_error)?;
    match &node {
        TermNode::Atom(Atom::Symbol(name)) => {
            let symbol = model
                .symbols
                .iter()
                .find(|s| s.name == name.name)
                .map(|s| s.id)
                .ok_or_else(|| {
                    AlgebraRefusal::Unsupported(format!(
                        "projection result references unknown symbol `{}`",
                        name.name
                    ))
                })?;
            let ty = model.symbols[symbol.index()].ty.clone();
            push(
                extension,
                base,
                SemanticExprKind::Symbol { symbol },
                ty,
                span,
            )
        }
        TermNode::Atom(atom) => {
            let value = atom.exact_rational_value().ok_or_else(|| {
                AlgebraRefusal::Unsupported("projection result atom has no exact value".into())
            })?;
            let exact = ExactLiteral::from_value(value.to_f64_lossy());
            push(
                extension,
                base,
                SemanticExprKind::Number {
                    value: value.to_f64_lossy(),
                    unit: None,
                    exact,
                },
                scalar_dimensionless(SemanticRole::Literal),
                span,
            )
        }
        TermNode::Apply { head, arguments } => {
            let name = operator_name(store, *head).ok_or_else(|| {
                AlgebraRefusal::Unsupported("projection result applies an unrecognized head".into())
            })?;
            let children = arguments
                .iter()
                .map(|argument| from_term(model, store, *argument, span, extension, base))
                .collect::<Result<Vec<_>, _>>()?;
            build_apply(&name, children, span, extension, base)
        }
        other => Err(AlgebraRefusal::Unsupported(format!(
            "projection result is a {other:?} node outside the scalar contract"
        ))),
    }
}

fn operator_name(store: &TermStore, head: TermId) -> Option<String> {
    let node = store.node(head).ok()?;
    match node {
        TermNode::Atom(Atom::Symbol(name)) if name.namespace == OPERATOR_NAMESPACE => {
            Some(name.name)
        }
        _ => None,
    }
}

fn build_apply(
    name: &str,
    mut children: Vec<ExprId>,
    span: SourceSpan,
    extension: &mut Vec<SemanticExpr>,
    base: usize,
) -> Result<ExprId, AlgebraRefusal> {
    let ty = scalar_dimensionless(SemanticRole::Intrinsic);
    match name {
        "Add" => fold_binary(children, BinaryOp::Add, 0.0, span, extension, base),
        "Mul" => fold_binary(children, BinaryOp::Mul, 1.0, span, extension, base),
        "Sub" if children.len() == 2 => {
            let rhs = children.pop().unwrap();
            let lhs = children.pop().unwrap();
            push(
                extension,
                base,
                SemanticExprKind::Binary {
                    op: BinaryOp::Sub,
                    lhs,
                    rhs,
                },
                ty,
                span,
            )
        }
        "Div" if children.len() == 2 => {
            let rhs = children.pop().unwrap();
            let lhs = children.pop().unwrap();
            push(
                extension,
                base,
                SemanticExprKind::Binary {
                    op: BinaryOp::Div,
                    lhs,
                    rhs,
                },
                ty,
                span,
            )
        }
        "Pow" if children.len() == 2 => {
            let rhs = children.pop().unwrap();
            let lhs = children.pop().unwrap();
            push(
                extension,
                base,
                SemanticExprKind::Binary {
                    op: BinaryOp::Pow,
                    lhs,
                    rhs,
                },
                ty,
                span,
            )
        }
        "Neg" if children.len() == 1 => push(
            extension,
            base,
            SemanticExprKind::Unary {
                op: UnaryOp::Neg,
                arg: children[0],
            },
            ty,
            span,
        ),
        "Sin" | "Cos" | "Exp" | "Sqrt" if children.len() == 1 => push(
            extension,
            base,
            SemanticExprKind::Call {
                function: name.to_lowercase(),
                args: children,
            },
            ty,
            span,
        ),
        "Ln" | "Log" if children.len() == 1 => push(
            extension,
            base,
            SemanticExprKind::Call {
                function: "ln".into(),
                args: children,
            },
            ty,
            span,
        ),
        other => Err(AlgebraRefusal::Unsupported(format!(
            "projection result applies unsupported operator `{other}` with {} argument(s)",
            children.len()
        ))),
    }
}

fn fold_binary(
    children: Vec<ExprId>,
    op: BinaryOp,
    identity: f64,
    span: SourceSpan,
    extension: &mut Vec<SemanticExpr>,
    base: usize,
) -> Result<ExprId, AlgebraRefusal> {
    let mut iter = children.into_iter();
    let Some(first) = iter.next() else {
        return push(
            extension,
            base,
            SemanticExprKind::Number {
                value: identity,
                unit: None,
                exact: ExactLiteral::from_value(identity),
            },
            scalar_dimensionless(SemanticRole::Literal),
            span,
        );
    };
    iter.try_fold(first, |lhs, rhs| {
        push(
            extension,
            base,
            SemanticExprKind::Binary { op, lhs, rhs },
            scalar_dimensionless(SemanticRole::Intrinsic),
            span,
        )
    })
}

fn push(
    extension: &mut Vec<SemanticExpr>,
    base: usize,
    kind: SemanticExprKind,
    ty: SemanticType,
    span: SourceSpan,
) -> Result<ExprId, AlgebraRefusal> {
    let id = ExprId((base + extension.len()) as u32);
    extension.push(SemanticExpr { id, kind, ty, span });
    Ok(id)
}

/// A scalar, dimensionless [`SemanticType`]. Every re-embedded projection node uses this: exact
/// dimension propagation through arbitrary differentiation/simplification is out of scope here
/// (see the module documentation), and `kernel::lower_expr`'s scalar path only checks
/// `SemanticShape::Numeric`, never `dimension`, so this does not weaken any downstream check.
fn scalar_dimensionless(role: SemanticRole) -> SemanticType {
    SemanticType {
        shape: SemanticShape::Numeric(crate::scientific::ValueShape::Scalar),
        axes: vec![],
        dimension: None,
        quantity_kind: None,
        frame: Frame::Neutral,
        role,
    }
}

// ==================== standalone (arena-free) expression lifting ====================
//
// `PropertyDefinition`/`PropertyModel` expressions (GX-A2, contract C7) and case-file
// `manufactured`/`expression` bindings (contract C3.2) are parsed with
// [`crate::scientific::parse_expression`] against a *declared input signature*, not resolved
// against a whole `.res` model. This section elaborates such a standalone [`Expr`] into
// [`SemanticExpr`] nodes suitable for [`project`], synthesizing symbols for its declared inputs
// plus the always-available `x`/`y`/`z`/`t` coordinate names (contract C3.2).

/// One symbol synthesized for standalone lifting: a declared property/provider input, or an
/// implicit `x`/`y`/`z`/`t` coordinate.
#[derive(Clone, Debug)]
pub struct LiftedSymbol {
    pub name: String,
    pub id: SymbolId,
}

/// Elaborate `expr` against `inputs` (contract C7's `PropertySignature::inputs`, or an empty
/// slice for a pure coordinate expression) into fresh [`SemanticExpr`] nodes appended to
/// `symbols`/`expressions`, reusing any symbol `symbols` already contains by name (so repeated
/// calls sharing one `symbols`/`expressions` pair -- e.g. lifting several exact-solution
/// bindings for one [`AlgebraOperation::ManufacturedForcing`] call -- share one `x`/`y`/`z`/`t`
/// symbol set).
pub fn lift_standalone_expr(
    expr: &Expr,
    inputs: &[PropertyInput],
    symbols: &mut Vec<SemanticSymbol>,
    expressions: &mut Vec<SemanticExpr>,
) -> Result<ExprId, ScientificError> {
    let mut names: BTreeMap<String, SymbolId> = symbols
        .iter()
        .map(|symbol| (symbol.name.clone(), symbol.id))
        .collect();
    for input in inputs {
        names
            .entry(input.name.clone())
            .or_insert_with(|| declare_symbol(symbols, &input.name, Some(input.dimension)));
    }
    for (name, dimension) in [
        ("x", Dimension::LENGTH),
        ("y", Dimension::LENGTH),
        ("z", Dimension::LENGTH),
        ("t", Dimension::TIME),
    ] {
        names
            .entry(name.into())
            .or_insert_with(|| declare_symbol(symbols, name, Some(dimension)));
    }
    lift_expr(expr, &names, expressions)
}

/// [`lift_standalone_expr`] plural form for a batch of named expressions (e.g. every field's
/// exact-solution binding for one manufactured-forcing derivation) sharing one coordinate
/// symbol set. Returns each input expression's [`ExprId`] in the same order.
pub fn lift_standalone_exprs(
    exprs: &[(&Expr, &[PropertyInput])],
    symbols: &mut Vec<SemanticSymbol>,
    expressions: &mut Vec<SemanticExpr>,
) -> Result<Vec<ExprId>, ScientificError> {
    exprs
        .iter()
        .map(|(expr, inputs)| lift_standalone_expr(expr, inputs, symbols, expressions))
        .collect()
}

fn declare_symbol(
    symbols: &mut Vec<SemanticSymbol>,
    name: &str,
    dimension: Option<Dimension>,
) -> SymbolId {
    let id = SymbolId(symbols.len() as u32);
    symbols.push(SemanticSymbol {
        id,
        name: name.into(),
        ty: SemanticType::numeric(
            crate::scientific::ValueShape::Scalar,
            dimension,
            Frame::Neutral,
            SemanticRole::Parameter,
        ),
        domain: None,
        space: None,
        span: SourceSpan::default(),
    });
    id
}

fn lift_expr(
    expr: &Expr,
    names: &BTreeMap<String, SymbolId>,
    expressions: &mut Vec<SemanticExpr>,
) -> Result<ExprId, ScientificError> {
    let (kind, ty) = match expr {
        Expr::Number {
            value,
            lexeme,
            unit,
            ..
        } => {
            if unit.is_some() {
                return Err(ScientificError::Property(
                    "unit-bearing literals are not supported in a standalone property/exact-solution expression"
                        .into(),
                ));
            }
            (
                SemanticExprKind::Number {
                    value: *value,
                    unit: None,
                    exact: ExactLiteral::from_lexeme(lexeme),
                },
                scalar_dimensionless(SemanticRole::Literal),
            )
        }
        Expr::Name { name, .. } if name == "pi" || name == "π" => (
            SemanticExprKind::Number {
                value: std::f64::consts::PI,
                unit: None,
                exact: ExactLiteral::from_value(std::f64::consts::PI),
            },
            scalar_dimensionless(SemanticRole::Literal),
        ),
        Expr::Name { name, .. } => {
            let symbol = *names
                .get(name)
                .ok_or_else(|| ScientificError::UnknownName(name.clone()))?;
            (
                SemanticExprKind::Symbol { symbol },
                SemanticType::numeric(
                    crate::scientific::ValueShape::Scalar,
                    None,
                    Frame::Neutral,
                    SemanticRole::Parameter,
                ),
            )
        }
        Expr::Unary { op, arg, .. } => {
            let arg_id = lift_expr(arg, names, expressions)?;
            (
                SemanticExprKind::Unary {
                    op: *op,
                    arg: arg_id,
                },
                scalar_dimensionless(SemanticRole::Intrinsic),
            )
        }
        Expr::Binary { op, lhs, rhs, .. } => {
            let lhs_id = lift_expr(lhs, names, expressions)?;
            let rhs_id = lift_expr(rhs, names, expressions)?;
            (
                SemanticExprKind::Binary {
                    op: *op,
                    lhs: lhs_id,
                    rhs: rhs_id,
                },
                scalar_dimensionless(SemanticRole::Intrinsic),
            )
        }
        Expr::Call { function, args, .. } => {
            let args_ids = args
                .iter()
                .map(|arg| lift_expr(arg, names, expressions))
                .collect::<Result<Vec<_>, _>>()?;
            (
                SemanticExprKind::Call {
                    function: function.clone(),
                    args: args_ids,
                },
                scalar_dimensionless(SemanticRole::Intrinsic),
            )
        }
        Expr::String { .. } | Expr::Index { .. } | Expr::Vector { .. } => {
            return Err(ScientificError::Property(
                "expression is outside the scalar property/exact-solution projection".into(),
            ));
        }
    };
    let id = ExprId(expressions.len() as u32);
    expressions.push(SemanticExpr {
        id,
        kind,
        ty,
        span: expr.span(),
    });
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scientific::{ValueShape, parse_expression};
    use quantitas::QuantityKindId;

    fn model_for(expr_text: &str) -> (SemanticModel, ExprId) {
        let expr = parse_expression(expr_text).unwrap();
        let inputs = vec![PropertyInput {
            name: "T".into(),
            quantity_kind: QuantityKindId::new("scientia:Unspecified"),
            dimension: Dimension::DIMENSIONLESS,
            shape: ValueShape::Scalar,
            physical_min: None,
            physical_max: None,
            nominal: None,
        }];
        let mut symbols = Vec::new();
        let mut expressions = Vec::new();
        let id = lift_standalone_expr(&expr, &inputs, &mut symbols, &mut expressions).unwrap();
        let model = SemanticModel {
            name: "test".into(),
            domains: vec![],
            regions: vec![],
            providers: vec![],
            symbols,
            expressions: expressions.into(),
            declarations: vec![],
            span: SourceSpan::default(),
        };
        (model, id)
    }

    fn symbol(model: &SemanticModel, name: &str) -> SymbolId {
        model
            .symbols
            .iter()
            .find(|symbol| symbol.name == name)
            .unwrap()
            .id
    }

    fn eval_numeric(arena: &[SemanticExpr], id: ExprId) -> f64 {
        match &arena[id.index()].kind {
            SemanticExprKind::Number { value, .. } => *value,
            SemanticExprKind::Unary { arg, .. } => -eval_numeric(arena, *arg),
            SemanticExprKind::Binary { op, lhs, rhs } => {
                let lhs = eval_numeric(arena, *lhs);
                let rhs = eval_numeric(arena, *rhs);
                match op {
                    BinaryOp::Add => lhs + rhs,
                    BinaryOp::Sub => lhs - rhs,
                    BinaryOp::Mul => lhs * rhs,
                    BinaryOp::Div => lhs / rhs,
                    BinaryOp::Pow => lhs.powf(rhs),
                    _ => panic!("unexpected comparison in a constant-folded derivative"),
                }
            }
            other => panic!("unexpected node {other:?} in a constant-folded derivative"),
        }
    }

    // Ported from the old `algebra.rs`-era `scientific.rs` tests: symbolic differentiation of a
    // property expression now projects through Resolvent's RV1-C2 surface (contract C8) instead
    // of the deleted `Expr`-only bridge.
    #[test]
    fn differentiate_matches_finite_difference() {
        let (model, id) = model_for("10 + 0.5 * T");
        let wrt = symbol(&model, "T");
        let outcome = project(&model, id, &AlgebraOperation::Differentiate { wrt }).unwrap();
        let value = eval_numeric(&outcome.arena, outcome.result);
        assert!((value - 0.5).abs() < 1e-12);
    }

    #[test]
    fn comparisons_are_refused_not_zeroed() {
        let (model, id) = model_for("T < 300");
        let wrt = symbol(&model, "T");
        let err = project(&model, id, &AlgebraOperation::Differentiate { wrt }).unwrap_err();
        assert_eq!(err.code(), "ALGEBRA_OUTSIDE_OPERATION");
    }

    #[test]
    fn simplify_folds_constant_arithmetic() {
        let (model, id) = model_for("2 + 3");
        let outcome = project(&model, id, &AlgebraOperation::Simplify).unwrap();
        let value = eval_numeric(&outcome.arena, outcome.result);
        assert!((value - 5.0).abs() < 1e-12);
    }

    #[test]
    fn substitute_replaces_every_occurrence() {
        let (model, id) = model_for("T * T");
        let wrt = symbol(&model, "T");
        let symbols = model.symbols.clone();
        let mut expressions = model.expressions.to_vec();
        let replacement = crate::projection::lift_expr(
            &parse_expression("5").unwrap(),
            &symbols.iter().map(|s| (s.name.clone(), s.id)).collect(),
            &mut expressions,
        )
        .unwrap();
        let extended = SemanticModel {
            expressions: expressions.into(),
            ..model.clone()
        };
        let mut bindings = BTreeMap::new();
        bindings.insert(wrt, replacement);
        let outcome = project(&extended, id, &AlgebraOperation::Substitute { bindings }).unwrap();
        let value = eval_numeric(&outcome.arena, outcome.result);
        assert!((value - 25.0).abs() < 1e-12);
    }

    fn eval_with_coordinates(
        arena: &[SemanticExpr],
        id: ExprId,
        symbols: &[crate::semantic::SemanticSymbol],
        x: f64,
        y: f64,
        t: f64,
    ) -> f64 {
        let name = |symbol: SymbolId| symbols[symbol.index()].name.as_str();
        match &arena[id.index()].kind {
            SemanticExprKind::Number { value, .. } => *value,
            SemanticExprKind::Symbol { symbol } => match name(*symbol) {
                "x" => x,
                "y" => y,
                "t" => t,
                other => panic!("unbound coordinate symbol `{other}` in forcing expression"),
            },
            SemanticExprKind::Unary { arg, .. } => {
                -eval_with_coordinates(arena, *arg, symbols, x, y, t)
            }
            SemanticExprKind::Binary { op, lhs, rhs } => {
                let lhs = eval_with_coordinates(arena, *lhs, symbols, x, y, t);
                let rhs = eval_with_coordinates(arena, *rhs, symbols, x, y, t);
                match op {
                    BinaryOp::Add => lhs + rhs,
                    BinaryOp::Sub => lhs - rhs,
                    BinaryOp::Mul => lhs * rhs,
                    BinaryOp::Div => lhs / rhs,
                    BinaryOp::Pow => lhs.powf(rhs),
                    other => panic!("unexpected comparison {other:?} in forcing expression"),
                }
            }
            SemanticExprKind::Call { function, args } if args.len() == 1 => {
                let arg = eval_with_coordinates(arena, args[0], symbols, x, y, t);
                match function.as_str() {
                    "sin" => arg.sin(),
                    "cos" => arg.cos(),
                    "exp" => arg.exp(),
                    other => panic!("unexpected call `{other}` in forcing expression"),
                }
            }
            other => panic!("unexpected node {other:?} in forcing expression"),
        }
    }

    // GX-A3 (contract C3.2/C8): `AlgebraOperation::ManufacturedForcing` substitutes the exact
    // solution into the strong form and expands `dt`/`grad`/`div` symbolically. For
    // `u_exact(x, t) = sin(x) + t` on the homogeneous heat equation `dt(u) - div(grad(u)) = 0`,
    // the forcing is `dt(u_exact) - div(grad(u_exact)) = 1 - (-sin(x)) = 1 + sin(x)`, computed
    // here without any corpus fixture.
    #[test]
    fn manufactured_forcing_expands_dt_and_div_grad_of_the_exact_solution() {
        let source = r#"
module gx_a3.manufactured;
model Diffusion {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: state scalar H1(order=1) on Omega { time_role = differential; };
  equation heat on Omega { dt(u) - div(grad(u)) = 0; }
}
"#;
        let compilation =
            crate::semantic::compile_semantics(source, &quantitas::UnitRegistry::si_bootstrap())
                .unwrap();
        let model = &compilation.semantic.models[0];
        let equation = model
            .declarations
            .iter()
            .find(|declaration| declaration.name == "heat")
            .unwrap()
            .id;
        let u_symbol = symbol(model, "u");

        let exact = parse_expression("sin(x) + t").unwrap();
        let mut symbols = model.symbols.clone();
        let mut expressions = model.expressions.to_vec();
        let exact_id = lift_standalone_expr(&exact, &[], &mut symbols, &mut expressions).unwrap();
        let combined = SemanticModel {
            symbols: symbols.clone(),
            expressions: expressions.into(),
            ..model.clone()
        };

        let mut exact_bindings = BTreeMap::new();
        exact_bindings.insert(u_symbol, exact_id);
        let outcome = project(
            &combined,
            ExprId(0),
            &AlgebraOperation::ManufacturedForcing {
                equation,
                exact: exact_bindings,
            },
        )
        .unwrap();

        for x in [0.0_f64, 0.7, -1.3] {
            let forcing =
                eval_with_coordinates(&outcome.arena, outcome.result, &symbols, x, 0.0, 0.0);
            let expected = 1.0 + x.sin();
            assert!(
                (forcing - expected).abs() < 1e-9,
                "x={x}: forcing={forcing}, expected={expected}"
            );
        }
    }
}
