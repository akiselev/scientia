//! Factoring of typed forms and lowering of point-local scalar work into Malleus IR.

use crate::formulation::{FormArgumentRole, FormCaptureRole, VariationalForm};
use crate::id::{Digest, span_independent_digest};
use crate::scientific::{BinaryOp as ResBinaryOp, FieldRole, SpaceSpec, UnaryOp as ResUnaryOp};
use crate::semantic::{
    DomainId, ExprId, SemanticExpr, SemanticExprKind, SemanticMeasure, SemanticShape, SemanticType,
    SymbolId,
};
use crate::source::SourceSpan;
use malleus::{
    AccessMode, BinaryOp, IndexingMap, IterationDomain, KernelOperand, KernelRegion, NumericPolicy,
    OperandId, ScalarExpr, Statement, StructuredKernel, UnaryOp,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const LOCAL_FORM_PROGRAM_SCHEMA: &str = "scientia-local-form-program/3";
pub const KERNEL_LOWERING_SCHEMA: &str = "scientia-kernel-lowering/1";

/// Mathematical role of an externally bound value in point-local form evaluation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalInputRole {
    TestBasis,
    TrialBasis,
    PhysicalField(FieldRole),
    Parameter,
    Constant,
    Source,
    Property,
    ConstitutiveLaw,
}

/// Point-local quantity required from an input binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputEvaluation {
    Value,
    Gradient,
    TimeDerivative,
    Trace,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalInput {
    pub symbol: SymbolId,
    pub role: LocalInputRole,
    pub ty: SemanticType,
    pub domain: Option<DomainId>,
    pub space: Option<SpaceSpec>,
    pub evaluations: Vec<InputEvaluation>,
    pub source_span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalOutputRole {
    ResidualContribution,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalOutput {
    pub role: LocalOutputRole,
    pub ty: SemanticType,
}

/// Scientia kernels are QFunctions evaluated at one already-selected quadrature point.
/// Quadrature selection and traversal belong to Finitum's realization plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalIterationContract {
    QuadraturePoint,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalFactorizationReceipt {
    pub source_form_digest: Digest,
    pub integral_index: usize,
    pub source_span: SourceSpan,
    pub transformation: LocalTransformation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalTransformation {
    SelectAuthoredIntegral,
}

/// Realization-neutral point work selected from one typed form integral.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LocalFormProgram {
    pub schema: String,
    pub name: String,
    pub source_form_digest: Digest,
    pub artifact_digest: Digest,
    pub measure: SemanticMeasure,
    pub inputs: Vec<LocalInput>,
    pub output: LocalOutput,
    pub expressions: Vec<SemanticExpr>,
    pub expression: ExprId,
    pub iteration: LocalIterationContract,
    pub receipt: LocalFactorizationReceipt,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelLoweringReceipt {
    pub schema: String,
    pub source_program_digest: Digest,
    pub artifact_digest: Digest,
    pub lowering: KernelLoweringMethod,
    pub iteration: LocalIterationContract,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelLoweringMethod {
    SemanticScalarPointKernel,
}

/// Malleus IR plus the evidence link back to the local form program that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct LoweredKernel {
    pub kernel: StructuredKernel,
    pub receipt: KernelLoweringReceipt,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum KernelLoweringError {
    #[error("form `{form}` has no integral at index {index}")]
    MissingIntegral { form: String, index: usize },
    #[error("semantic expression id {0} is outside the form arena")]
    InvalidExpression(ExprId),
    #[error("form expression references unclassified symbol {0}")]
    UnclassifiedSymbol(SymbolId),
    #[error("kernel expression references unbound symbol {0}")]
    UnboundSymbol(SymbolId),
    #[error("expression `{0}` requires tensor/form lowering that is not implemented yet")]
    UnsupportedExpression(String),
}

/// Select typed, realization-neutral point work from a form integral.
pub fn factor_local_integral(
    form: &VariationalForm,
    integral_index: usize,
) -> Result<LocalFormProgram, KernelLoweringError> {
    let integral =
        form.integrals
            .get(integral_index)
            .ok_or_else(|| KernelLoweringError::MissingIntegral {
                form: form.name.clone(),
                index: integral_index,
            })?;
    let mut evaluations = BTreeMap::<SymbolId, BTreeSet<InputEvaluation>>::new();
    let default_evaluation = match integral.measure {
        SemanticMeasure::ExteriorFacet { .. }
        | SemanticMeasure::InteriorFacet { .. }
        | SemanticMeasure::Interface { .. } => InputEvaluation::Trace,
        SemanticMeasure::Cell { .. } | SemanticMeasure::Point { .. } => InputEvaluation::Value,
    };
    collect_input_evaluations(
        &form.expressions,
        integral.integrand,
        default_evaluation,
        &mut evaluations,
    )?;

    let mut inputs = Vec::with_capacity(evaluations.len());
    for (symbol, evaluations) in evaluations {
        if let Some(argument) = form.arguments.iter().find(|item| item.symbol == symbol) {
            inputs.push(LocalInput {
                symbol,
                role: match argument.role {
                    FormArgumentRole::Test => LocalInputRole::TestBasis,
                    FormArgumentRole::Trial => LocalInputRole::TrialBasis,
                },
                ty: argument.ty.clone(),
                domain: Some(argument.domain),
                space: Some(argument.space.clone()),
                evaluations: evaluations.into_iter().collect(),
                source_span: argument.source_span,
            });
            continue;
        }
        let capture = form
            .captures
            .iter()
            .find(|item| item.symbol == symbol)
            .ok_or(KernelLoweringError::UnclassifiedSymbol(symbol))?;
        inputs.push(LocalInput {
            symbol,
            role: capture_role(&capture.role),
            ty: capture.ty.clone(),
            domain: capture.domain,
            space: capture.space.clone(),
            evaluations: evaluations.into_iter().collect(),
            source_span: capture.source_span,
        });
    }

    let expression = form
        .expressions
        .get(integral.integrand.index())
        .ok_or(KernelLoweringError::InvalidExpression(integral.integrand))?;
    let name = format!("{}::{}::integral_{integral_index}", form.model, form.name);
    let receipt = LocalFactorizationReceipt {
        source_form_digest: form.artifact_digest.clone(),
        integral_index,
        source_span: integral.source_span,
        transformation: LocalTransformation::SelectAuthoredIntegral,
    };
    let artifact_digest = span_independent_digest(&LocalProgramDigestPayload {
        schema: LOCAL_FORM_PROGRAM_SCHEMA,
        name: &name,
        source_form_digest: &form.artifact_digest,
        measure: &integral.measure,
        inputs: &inputs,
        output_type: &expression.ty,
        expressions: &form.expressions,
        expression: integral.integrand,
        iteration: LocalIterationContract::QuadraturePoint,
        receipt: &receipt,
    });

    Ok(LocalFormProgram {
        schema: LOCAL_FORM_PROGRAM_SCHEMA.into(),
        name,
        source_form_digest: form.artifact_digest.clone(),
        artifact_digest,
        measure: integral.measure.clone(),
        inputs,
        output: LocalOutput {
            role: LocalOutputRole::ResidualContribution,
            ty: expression.ty.clone(),
        },
        expressions: form.expressions.clone(),
        expression: integral.integrand,
        iteration: LocalIterationContract::QuadraturePoint,
        receipt,
    })
}

#[derive(Serialize)]
struct LocalProgramDigestPayload<'a> {
    schema: &'static str,
    name: &'a str,
    source_form_digest: &'a Digest,
    measure: &'a SemanticMeasure,
    inputs: &'a [LocalInput],
    output_type: &'a SemanticType,
    expressions: &'a [SemanticExpr],
    expression: ExprId,
    iteration: LocalIterationContract,
    receipt: &'a LocalFactorizationReceipt,
}

/// Lower scalar point work into Malleus. The empty Malleus iteration domain means exactly one
/// invocation; it does not stand in for an omitted quadrature loop.
pub fn lower_local_program(
    program: &LocalFormProgram,
) -> Result<LoweredKernel, KernelLoweringError> {
    let mut operands = Vec::with_capacity(program.inputs.len() + 1);
    let mut bindings = BTreeMap::new();
    for input in &program.inputs {
        let id = OperandId::new(operands.len());
        operands.push(KernelOperand::scalar(
            format!("symbol_{}", input.symbol.0),
            AccessMode::Read,
        ));
        bindings.insert(input.symbol, id);
    }
    let output = OperandId::new(operands.len());
    operands.push(KernelOperand::scalar("residual", AccessMode::Write));
    let indexing_maps = (0..operands.len())
        .map(|index| IndexingMap::scalar(OperandId::new(index)))
        .collect();
    let value = lower_expr(&program.expressions, program.expression, &bindings)?;

    let kernel = StructuredKernel {
        name: program.name.clone(),
        iteration_domain: IterationDomain::default(),
        iterators: Vec::new(),
        operands,
        indexing_maps,
        body: KernelRegion {
            statements: vec![Statement::Store {
                operand: output,
                value,
            }],
        },
        numeric_policy: NumericPolicy::default(),
    };
    let lowering = KernelLoweringMethod::SemanticScalarPointKernel;
    let artifact_digest = span_independent_digest(&KernelDigestPayload {
        schema: KERNEL_LOWERING_SCHEMA,
        source_program_digest: &program.artifact_digest,
        lowering,
        iteration: program.iteration,
    });
    Ok(LoweredKernel {
        kernel,
        receipt: KernelLoweringReceipt {
            schema: KERNEL_LOWERING_SCHEMA.into(),
            source_program_digest: program.artifact_digest.clone(),
            artifact_digest,
            lowering,
            iteration: program.iteration,
        },
    })
}

#[derive(Serialize)]
struct KernelDigestPayload<'a> {
    schema: &'static str,
    source_program_digest: &'a Digest,
    lowering: KernelLoweringMethod,
    iteration: LocalIterationContract,
}

fn expression(
    expressions: &[SemanticExpr],
    id: ExprId,
) -> Result<&SemanticExpr, KernelLoweringError> {
    expressions
        .get(id.index())
        .ok_or(KernelLoweringError::InvalidExpression(id))
}

fn collect_input_evaluations(
    expressions: &[SemanticExpr],
    id: ExprId,
    evaluation: InputEvaluation,
    inputs: &mut BTreeMap<SymbolId, BTreeSet<InputEvaluation>>,
) -> Result<(), KernelLoweringError> {
    match &expression(expressions, id)?.kind {
        SemanticExprKind::Symbol { symbol } => {
            inputs.entry(*symbol).or_default().insert(evaluation);
        }
        SemanticExprKind::Unary { arg, .. } => {
            collect_input_evaluations(expressions, *arg, evaluation, inputs)?;
        }
        SemanticExprKind::Differential { operator, arg } => {
            let nested = match operator {
                crate::semantic::DifferentialOperator::Gradient
                | crate::semantic::DifferentialOperator::RotatedGradient
                | crate::semantic::DifferentialOperator::SymmetricGradient
                | crate::semantic::DifferentialOperator::Divergence
                | crate::semantic::DifferentialOperator::Curl => InputEvaluation::Gradient,
                crate::semantic::DifferentialOperator::TimeDerivative => {
                    InputEvaluation::TimeDerivative
                }
            };
            collect_input_evaluations(expressions, *arg, nested, inputs)?;
        }
        SemanticExprKind::FacetTrace { value, .. }
        | SemanticExprKind::Jump { value }
        | SemanticExprKind::Average { value } => {
            collect_input_evaluations(expressions, *value, InputEvaluation::Trace, inputs)?;
        }
        SemanticExprKind::TensorTrace { value, .. }
        | SemanticExprKind::Conjugate { value }
        | SemanticExprKind::NormalComponent { value, .. } => {
            collect_input_evaluations(expressions, *value, evaluation, inputs)?;
        }
        SemanticExprKind::Binary { lhs, rhs, .. } => {
            collect_input_evaluations(expressions, *lhs, evaluation, inputs)?;
            collect_input_evaluations(expressions, *rhs, evaluation, inputs)?;
        }
        SemanticExprKind::Contraction { lhs, rhs, .. } => {
            collect_input_evaluations(expressions, *lhs, evaluation, inputs)?;
            collect_input_evaluations(expressions, *rhs, evaluation, inputs)?;
        }
        SemanticExprKind::Call { function, args } => {
            let nested = match function.as_str() {
                "grad" => InputEvaluation::Gradient,
                "dt" => InputEvaluation::TimeDerivative,
                "trace" => InputEvaluation::Trace,
                _ => evaluation,
            };
            for argument in args {
                collect_input_evaluations(expressions, *argument, nested, inputs)?;
            }
        }
        SemanticExprKind::Index { value, indices } => {
            collect_input_evaluations(expressions, *value, evaluation, inputs)?;
            for index in indices {
                collect_input_evaluations(expressions, *index, InputEvaluation::Value, inputs)?;
            }
        }
        SemanticExprKind::Vector { elements } => {
            for element in elements {
                collect_input_evaluations(expressions, *element, evaluation, inputs)?;
            }
        }
        SemanticExprKind::Number { .. } | SemanticExprKind::String { .. } => {}
    }
    Ok(())
}

fn capture_role(role: &FormCaptureRole) -> LocalInputRole {
    match role {
        FormCaptureRole::PhysicalField(role) => LocalInputRole::PhysicalField(role.clone()),
        FormCaptureRole::Parameter => LocalInputRole::Parameter,
        FormCaptureRole::Constant => LocalInputRole::Constant,
        FormCaptureRole::Source => LocalInputRole::Source,
        FormCaptureRole::Property => LocalInputRole::Property,
        FormCaptureRole::ConstitutiveLaw => LocalInputRole::ConstitutiveLaw,
    }
}

fn lower_expr(
    expressions: &[SemanticExpr],
    id: ExprId,
    bindings: &BTreeMap<SymbolId, OperandId>,
) -> Result<ScalarExpr, KernelLoweringError> {
    let expression = expression(expressions, id)?;
    if !matches!(expression.ty.shape, SemanticShape::Numeric(_)) {
        return Err(KernelLoweringError::UnsupportedExpression(format!(
            "non-numeric expression {id}"
        )));
    }
    Ok(match &expression.kind {
        SemanticExprKind::Number { value, unit: None } => ScalarExpr::Constant(*value),
        SemanticExprKind::Number { unit: Some(_), .. } => {
            return Err(KernelLoweringError::UnsupportedExpression(
                "unit-bearing literal before numeric canonicalization".into(),
            ));
        }
        SemanticExprKind::Symbol { symbol } => ScalarExpr::Load(
            *bindings
                .get(symbol)
                .ok_or(KernelLoweringError::UnboundSymbol(*symbol))?,
        ),
        SemanticExprKind::Unary {
            op: ResUnaryOp::Neg,
            arg,
        } => ScalarExpr::unary(UnaryOp::Neg, lower_expr(expressions, *arg, bindings)?),
        SemanticExprKind::Binary { op, lhs, rhs } => {
            let op = match op {
                ResBinaryOp::Add => BinaryOp::Add,
                ResBinaryOp::Sub => BinaryOp::Sub,
                ResBinaryOp::Mul => BinaryOp::Mul,
                ResBinaryOp::Div => BinaryOp::Div,
                ResBinaryOp::Pow => BinaryOp::Pow,
                ResBinaryOp::Eq
                | ResBinaryOp::Lt
                | ResBinaryOp::Le
                | ResBinaryOp::Gt
                | ResBinaryOp::Ge => {
                    return Err(KernelLoweringError::UnsupportedExpression(
                        "comparison in a scalar integrand".into(),
                    ));
                }
            };
            ScalarExpr::binary(
                op,
                lower_expr(expressions, *lhs, bindings)?,
                lower_expr(expressions, *rhs, bindings)?,
            )
        }
        SemanticExprKind::Call { function, args } => {
            lower_call(expressions, function, args, bindings)?
        }
        SemanticExprKind::String { .. } => {
            return Err(KernelLoweringError::UnsupportedExpression(
                "string literal".into(),
            ));
        }
        SemanticExprKind::Index { .. } => {
            return Err(KernelLoweringError::UnsupportedExpression(
                "indexed tensor expression".into(),
            ));
        }
        SemanticExprKind::Vector { .. } => {
            return Err(KernelLoweringError::UnsupportedExpression(
                "vector expression".into(),
            ));
        }
        SemanticExprKind::Differential { .. }
        | SemanticExprKind::Contraction { .. }
        | SemanticExprKind::TensorTrace { .. }
        | SemanticExprKind::FacetTrace { .. }
        | SemanticExprKind::Jump { .. }
        | SemanticExprKind::Average { .. }
        | SemanticExprKind::Conjugate { .. }
        | SemanticExprKind::NormalComponent { .. } => {
            return Err(KernelLoweringError::UnsupportedExpression(
                "typed form operation before tensor lowering".into(),
            ));
        }
    })
}

fn lower_call(
    expressions: &[SemanticExpr],
    function: &str,
    args: &[ExprId],
    bindings: &BTreeMap<SymbolId, OperandId>,
) -> Result<ScalarExpr, KernelLoweringError> {
    let unary = match function {
        "abs" => Some(UnaryOp::Abs),
        "sqrt" => Some(UnaryOp::Sqrt),
        "exp" => Some(UnaryOp::Exp),
        "log" | "ln" => Some(UnaryOp::Ln),
        "sin" => Some(UnaryOp::Sin),
        "cos" => Some(UnaryOp::Cos),
        "tan" => Some(UnaryOp::Tan),
        "floor" => Some(UnaryOp::Floor),
        "ceil" => Some(UnaryOp::Ceil),
        _ => None,
    };
    if let (Some(op), [arg]) = (unary, args) {
        return Ok(ScalarExpr::unary(
            op,
            lower_expr(expressions, *arg, bindings)?,
        ));
    }
    let binary = match function {
        "min" => Some(BinaryOp::Min),
        "max" => Some(BinaryOp::Max),
        "atan2" => Some(BinaryOp::Atan2),
        _ => None,
    };
    if let (Some(op), [left, right]) = (binary, args) {
        return Ok(ScalarExpr::binary(
            op,
            lower_expr(expressions, *left, bindings)?,
            lower_expr(expressions, *right, bindings)?,
        ));
    }
    Err(KernelLoweringError::UnsupportedExpression(format!(
        "call to `{function}`"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compile_semantics, compile_variational_form};
    use quantitas::UnitRegistry;

    fn compile_form(source: &str, model: &str) -> VariationalForm {
        let compilation = compile_semantics(source, &UnitRegistry::si_bootstrap()).unwrap();
        compile_variational_form(&compilation.semantic, model, "residual").unwrap()
    }

    #[test]
    fn scalar_form_integrand_preserves_roles_and_lowers_to_valid_malleus_ir() {
        let form = compile_form(
            r#"
module kernel.test;
model Reaction {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: trial scalar H1(order=1) on Omega;
  field v: test scalar H1(order=1) on Omega;
  parameter alpha: Rate;
  form residual { cell(Omega): alpha * u * v; }
}
"#,
            "Reaction",
        );
        let program = factor_local_integral(&form, 0).unwrap();
        assert_eq!(program.inputs[0].role, LocalInputRole::TrialBasis);
        assert_eq!(program.inputs[1].role, LocalInputRole::TestBasis);
        assert_eq!(program.inputs[2].role, LocalInputRole::Parameter);
        assert_eq!(program.iteration, LocalIterationContract::QuadraturePoint);
        assert_eq!(program.receipt.source_form_digest, form.artifact_digest);

        let lowered = lower_local_program(&program).unwrap();
        assert_eq!(lowered.kernel.operands.len(), 4);
        assert!(lowered.kernel.iteration_domain.extents.is_empty());
        assert_eq!(
            lowered.receipt.source_program_digest,
            program.artifact_digest
        );
        malleus::validate(lowered.kernel).unwrap();
    }

    #[test]
    fn differential_requirements_are_typed_but_not_smuggled_into_scalar_ir() {
        let form = compile_form(
            r#"
module kernel.test;
model Diffusion {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: trial scalar H1(order=1) on Omega;
  field v: test scalar H1(order=1) on Omega;
  form residual { cell(Omega): dot(grad(u), grad(v)); }
}
"#,
            "Diffusion",
        );
        let program = factor_local_integral(&form, 0).unwrap();
        assert!(
            program
                .inputs
                .iter()
                .all(|input| input.evaluations == [InputEvaluation::Gradient])
        );
        assert!(matches!(
            lower_local_program(&program),
            Err(KernelLoweringError::UnsupportedExpression(_))
        ));
    }
}
