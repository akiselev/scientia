//! Deterministic reference interpretation of typed variational-form expressions.

use crate::formulation::VariationalForm;
use crate::scientific::{BinaryOp, UnaryOp};
use crate::semantic::{
    DifferentialOperator, ExprId, SemanticExpr, SemanticExprKind, SymbolId, TraceSide,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FormValue {
    Real { value: f64 },
    Complex { re: f64, im: f64 },
    Tensor { shape: Vec<usize>, data: Vec<f64> },
    Boolean { value: bool },
}

impl FormValue {
    pub const fn real(value: f64) -> Self {
        Self::Real { value }
    }

    pub fn vector(data: Vec<f64>) -> Self {
        Self::Tensor {
            shape: vec![data.len()],
            data,
        }
    }

    pub fn tensor(shape: Vec<usize>, data: Vec<f64>) -> Result<Self, FormInterpretError> {
        let expected = shape.iter().try_fold(1usize, |size, extent| {
            size.checked_mul(*extent)
                .ok_or(FormInterpretError::InvalidTensorShape)
        })?;
        if expected != data.len() {
            return Err(FormInterpretError::InvalidTensorShape);
        }
        Ok(Self::Tensor { shape, data })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormEvaluation {
    Value,
    Gradient,
    Divergence,
    Curl,
    RotatedGradient,
    SymmetricGradient,
    TimeDerivative,
    Trace(TraceSide),
    NormalComponent(TraceSide),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FormEvaluationKey {
    pub expression: ExprId,
    pub evaluation: FormEvaluation,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FormEvaluationContext {
    pub symbols: BTreeMap<SymbolId, FormValue>,
    pub evaluations: BTreeMap<FormEvaluationKey, FormValue>,
    pub normals: BTreeMap<TraceSide, Vec<f64>>,
}

impl FormEvaluationContext {
    pub fn bind_symbol(&mut self, symbol: SymbolId, value: FormValue) {
        self.symbols.insert(symbol, value);
    }

    pub fn bind_evaluation(
        &mut self,
        expression: ExprId,
        evaluation: FormEvaluation,
        value: FormValue,
    ) {
        self.evaluations.insert(
            FormEvaluationKey {
                expression,
                evaluation,
            },
            value,
        );
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FormSample {
    pub integral_index: usize,
    pub weight: f64,
    pub context: FormEvaluationContext,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum FormInterpretError {
    #[error("form has no integral at index {0}")]
    MissingIntegral(usize),
    #[error("semantic expression id {0} is outside the form arena")]
    InvalidExpression(ExprId),
    #[error("no value was bound for symbol {0}")]
    MissingSymbol(SymbolId),
    #[error("no {evaluation:?} evaluation was bound for expression {expression}")]
    MissingEvaluation {
        expression: ExprId,
        evaluation: FormEvaluation,
    },
    #[error("no normal was bound for side {0:?}")]
    MissingNormal(TraceSide),
    #[error("operation `{operation}` does not support operands {operands}")]
    TypeMismatch {
        operation: &'static str,
        operands: String,
    },
    #[error("tensor shape does not match its data")]
    InvalidTensorShape,
    #[error("tensor index is outside its declared shape")]
    IndexOutOfBounds,
    #[error("unit-bearing literal requires prior numeric canonicalization")]
    UnitBearingLiteral,
    #[error("string expressions cannot be numerically interpreted")]
    StringExpression,
    #[error("call `{0}` has no deterministic form-interpreter implementation")]
    UnsupportedCall(String),
}

pub fn interpret_integral(
    form: &VariationalForm,
    integral_index: usize,
    context: &FormEvaluationContext,
) -> Result<FormValue, FormInterpretError> {
    let integral = form
        .integrals
        .get(integral_index)
        .ok_or(FormInterpretError::MissingIntegral(integral_index))?;
    Interpreter {
        expressions: &form.expressions,
        context,
    }
    .expression(integral.integrand)
}

/// Return the deterministic set of expression/evaluation bindings accepted by the reference
/// interpreter for one integral. Scalar literals and implemented algebra need no binding.
pub fn required_evaluations(
    form: &VariationalForm,
    integral_index: usize,
) -> Result<Vec<FormEvaluationKey>, FormInterpretError> {
    let integral = form
        .integrals
        .get(integral_index)
        .ok_or(FormInterpretError::MissingIntegral(integral_index))?;
    let mut required = std::collections::BTreeSet::new();
    collect_required_evaluations(&form.expressions, integral.integrand, &mut required)?;
    Ok(required.into_iter().collect())
}

/// Interpret and accumulate weighted integral samples in caller-provided order.
/// Quadrature construction and traversal remain outside Scientia.
pub fn interpret_form(
    form: &VariationalForm,
    samples: &[FormSample],
) -> Result<FormValue, FormInterpretError> {
    let mut total = FormValue::real(0.0);
    for sample in samples {
        let value = interpret_integral(form, sample.integral_index, &sample.context)?;
        total = add(total, scale(value, sample.weight)?)?;
    }
    Ok(total)
}

struct Interpreter<'a> {
    expressions: &'a [SemanticExpr],
    context: &'a FormEvaluationContext,
}

fn collect_required_evaluations(
    expressions: &[SemanticExpr],
    id: ExprId,
    required: &mut std::collections::BTreeSet<FormEvaluationKey>,
) -> Result<(), FormInterpretError> {
    let expression = expressions
        .get(id.index())
        .ok_or(FormInterpretError::InvalidExpression(id))?;
    let insert =
        |required: &mut std::collections::BTreeSet<FormEvaluationKey>, expression, evaluation| {
            required.insert(FormEvaluationKey {
                expression,
                evaluation,
            });
        };
    match &expression.kind {
        SemanticExprKind::Symbol { .. } => insert(required, id, FormEvaluation::Value),
        SemanticExprKind::Differential { operator, arg } => insert(
            required,
            *arg,
            match operator {
                DifferentialOperator::Gradient => FormEvaluation::Gradient,
                DifferentialOperator::Divergence => FormEvaluation::Divergence,
                DifferentialOperator::Curl => FormEvaluation::Curl,
                DifferentialOperator::RotatedGradient => FormEvaluation::RotatedGradient,
                DifferentialOperator::SymmetricGradient => FormEvaluation::SymmetricGradient,
                DifferentialOperator::TimeDerivative => FormEvaluation::TimeDerivative,
            },
        ),
        SemanticExprKind::FacetTrace { value, side } => {
            insert(required, *value, FormEvaluation::Trace(*side));
        }
        SemanticExprKind::Jump { value } | SemanticExprKind::Average { value } => {
            insert(required, *value, FormEvaluation::Trace(TraceSide::Minus));
            insert(required, *value, FormEvaluation::Trace(TraceSide::Plus));
        }
        SemanticExprKind::NormalComponent { value, side } => {
            insert(required, *value, FormEvaluation::NormalComponent(*side));
        }
        SemanticExprKind::Call { function, args } => {
            if matches!(
                function.as_str(),
                "abs"
                    | "sqrt"
                    | "exp"
                    | "log"
                    | "ln"
                    | "sin"
                    | "cos"
                    | "tan"
                    | "floor"
                    | "ceil"
                    | "min"
                    | "max"
            ) {
                for arg in args {
                    collect_required_evaluations(expressions, *arg, required)?;
                }
            } else {
                insert(required, id, FormEvaluation::Value);
            }
        }
        SemanticExprKind::ProviderCall { .. } => {
            // A typed provider call is opaque to the deterministic form interpreter for the
            // same reason an untyped `Call` is: it needs a caller-supplied bound value at
            // this expression id (`InputSourceRequirement::ModelDefinedProperty`), not a
            // recursive evaluation of its own arguments.
            insert(required, id, FormEvaluation::Value);
        }
        SemanticExprKind::Unary { arg, .. }
        | SemanticExprKind::TensorTrace { value: arg, .. }
        | SemanticExprKind::Conjugate { value: arg } => {
            collect_required_evaluations(expressions, *arg, required)?;
        }
        SemanticExprKind::Binary { lhs, rhs, .. }
        | SemanticExprKind::Contraction { lhs, rhs, .. } => {
            collect_required_evaluations(expressions, *lhs, required)?;
            collect_required_evaluations(expressions, *rhs, required)?;
        }
        SemanticExprKind::Index { value, indices } => {
            collect_required_evaluations(expressions, *value, required)?;
            for index in indices {
                collect_required_evaluations(expressions, *index, required)?;
            }
        }
        SemanticExprKind::Vector { elements } => {
            for element in elements {
                collect_required_evaluations(expressions, *element, required)?;
            }
        }
        SemanticExprKind::Number { .. } | SemanticExprKind::String { .. } => {}
    }
    Ok(())
}

impl Interpreter<'_> {
    fn node(&self, id: ExprId) -> Result<&SemanticExpr, FormInterpretError> {
        self.expressions
            .get(id.index())
            .ok_or(FormInterpretError::InvalidExpression(id))
    }

    fn expression(&self, id: ExprId) -> Result<FormValue, FormInterpretError> {
        let expression = self.node(id)?;
        Ok(match &expression.kind {
            SemanticExprKind::Number {
                value, unit: None, ..
            } => FormValue::real(*value),
            SemanticExprKind::Number { unit: Some(_), .. } => {
                return Err(FormInterpretError::UnitBearingLiteral);
            }
            SemanticExprKind::String { .. } => return Err(FormInterpretError::StringExpression),
            SemanticExprKind::Symbol { symbol } => self
                .context
                .symbols
                .get(symbol)
                .cloned()
                .or_else(|| self.bound(id, FormEvaluation::Value))
                .ok_or(FormInterpretError::MissingSymbol(*symbol))?,
            SemanticExprKind::Unary {
                op: UnaryOp::Neg,
                arg,
            } => scale(self.expression(*arg)?, -1.0)?,
            SemanticExprKind::Binary { op, lhs, rhs } => {
                binary(*op, self.expression(*lhs)?, self.expression(*rhs)?)?
            }
            SemanticExprKind::Call { function, args } => self.call(id, function, args)?,
            SemanticExprKind::ProviderCall { .. } => self.required(id, FormEvaluation::Value)?,
            SemanticExprKind::Differential { operator, arg } => {
                let evaluation = match operator {
                    DifferentialOperator::Gradient => FormEvaluation::Gradient,
                    DifferentialOperator::Divergence => FormEvaluation::Divergence,
                    DifferentialOperator::Curl => FormEvaluation::Curl,
                    DifferentialOperator::RotatedGradient => FormEvaluation::RotatedGradient,
                    DifferentialOperator::SymmetricGradient => FormEvaluation::SymmetricGradient,
                    DifferentialOperator::TimeDerivative => FormEvaluation::TimeDerivative,
                };
                self.required(*arg, evaluation)?
            }
            SemanticExprKind::Contraction {
                lhs,
                rhs,
                axes,
                conjugate_lhs,
            } => contract(
                self.expression(*lhs)?,
                self.expression(*rhs)?,
                axes,
                *conjugate_lhs,
            )?,
            SemanticExprKind::TensorTrace { value, axes } => {
                tensor_trace(self.expression(*value)?, axes.lhs, axes.rhs)?
            }
            SemanticExprKind::FacetTrace { value, side } => {
                self.required(*value, FormEvaluation::Trace(*side))?
            }
            SemanticExprKind::Jump { value } => sub(
                self.required(*value, FormEvaluation::Trace(TraceSide::Minus))?,
                self.required(*value, FormEvaluation::Trace(TraceSide::Plus))?,
            )?,
            SemanticExprKind::Average { value } => scale(
                add(
                    self.required(*value, FormEvaluation::Trace(TraceSide::Minus))?,
                    self.required(*value, FormEvaluation::Trace(TraceSide::Plus))?,
                )?,
                0.5,
            )?,
            SemanticExprKind::Conjugate { value } => conjugate(self.expression(*value)?),
            SemanticExprKind::NormalComponent { value, side } => self
                .bound(*value, FormEvaluation::NormalComponent(*side))
                .map(Ok)
                .unwrap_or_else(|| {
                    let traced = self
                        .bound(*value, FormEvaluation::Trace(*side))
                        .map(Ok)
                        .unwrap_or_else(|| self.expression(*value))?;
                    let normal = self
                        .context
                        .normals
                        .get(side)
                        .ok_or(FormInterpretError::MissingNormal(*side))?;
                    normal_component(traced, normal)
                })?,
            SemanticExprKind::Index { value, indices } => {
                let value = self.expression(*value)?;
                let indices = indices
                    .iter()
                    .map(|index| as_index(self.expression(*index)?))
                    .collect::<Result<Vec<_>, _>>()?;
                index(value, &indices)?
            }
            SemanticExprKind::Vector { elements } => {
                let data = elements
                    .iter()
                    .map(|element| as_real(self.expression(*element)?))
                    .collect::<Result<Vec<_>, _>>()?;
                FormValue::vector(data)
            }
        })
    }

    fn bound(&self, expression: ExprId, evaluation: FormEvaluation) -> Option<FormValue> {
        self.context
            .evaluations
            .get(&FormEvaluationKey {
                expression,
                evaluation,
            })
            .cloned()
    }

    fn required(
        &self,
        expression: ExprId,
        evaluation: FormEvaluation,
    ) -> Result<FormValue, FormInterpretError> {
        self.bound(expression, evaluation)
            .ok_or(FormInterpretError::MissingEvaluation {
                expression,
                evaluation,
            })
    }

    fn call(
        &self,
        id: ExprId,
        function: &str,
        args: &[ExprId],
    ) -> Result<FormValue, FormInterpretError> {
        if let Some(value) = self.bound(id, FormEvaluation::Value) {
            return Ok(value);
        }
        let values = args
            .iter()
            .map(|arg| self.expression(*arg))
            .collect::<Result<Vec<_>, _>>()?;
        let unary = |f: fn(f64) -> f64| -> Result<FormValue, FormInterpretError> {
            match values.as_slice() {
                [value] => Ok(FormValue::real(f(as_real(value.clone())?))),
                _ => Err(type_mismatch("unary call", &values)),
            }
        };
        match function {
            "abs" => unary(f64::abs),
            "sqrt" => unary(f64::sqrt),
            "exp" => unary(f64::exp),
            "log" | "ln" => unary(f64::ln),
            "sin" => unary(f64::sin),
            "cos" => unary(f64::cos),
            "tan" => unary(f64::tan),
            "floor" => unary(f64::floor),
            "ceil" => unary(f64::ceil),
            "min" if !values.is_empty() => {
                values
                    .into_iter()
                    .try_fold(FormValue::real(f64::INFINITY), |left, right| {
                        binary(BinaryOp::Lt, left.clone(), right.clone()).map(|less| {
                            if matches!(less, FormValue::Boolean { value: true }) {
                                left
                            } else {
                                right
                            }
                        })
                    })
            }
            "max" if !values.is_empty() => {
                values
                    .into_iter()
                    .try_fold(FormValue::real(f64::NEG_INFINITY), |left, right| {
                        binary(BinaryOp::Gt, left.clone(), right.clone()).map(|more| {
                            if matches!(more, FormValue::Boolean { value: true }) {
                                left
                            } else {
                                right
                            }
                        })
                    })
            }
            _ => Err(FormInterpretError::UnsupportedCall(function.to_owned())),
        }
    }
}

fn binary(
    op: BinaryOp,
    left: FormValue,
    right: FormValue,
) -> Result<FormValue, FormInterpretError> {
    match op {
        BinaryOp::Add => add(left, right),
        BinaryOp::Sub => sub(left, right),
        BinaryOp::Mul => multiply(left, right),
        BinaryOp::Div => divide(left, right),
        BinaryOp::Pow => Ok(FormValue::real(as_real(left)?.powf(as_real(right)?))),
        BinaryOp::Eq => Ok(FormValue::Boolean {
            value: left == right,
        }),
        BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
            let left = as_real(left)?;
            let right = as_real(right)?;
            Ok(FormValue::Boolean {
                value: match op {
                    BinaryOp::Lt => left < right,
                    BinaryOp::Le => left <= right,
                    BinaryOp::Gt => left > right,
                    BinaryOp::Ge => left >= right,
                    _ => unreachable!(),
                },
            })
        }
    }
}

fn add(left: FormValue, right: FormValue) -> Result<FormValue, FormInterpretError> {
    elementwise("addition", left, right, |left, right| left + right)
}

fn sub(left: FormValue, right: FormValue) -> Result<FormValue, FormInterpretError> {
    elementwise("subtraction", left, right, |left, right| left - right)
}

fn multiply(left: FormValue, right: FormValue) -> Result<FormValue, FormInterpretError> {
    match (left, right) {
        (FormValue::Real { value }, tensor @ FormValue::Tensor { .. })
        | (tensor @ FormValue::Tensor { .. }, FormValue::Real { value }) => scale(tensor, value),
        (FormValue::Complex { re, im }, FormValue::Complex { re: cr, im: ci }) => {
            Ok(FormValue::Complex {
                re: re * cr - im * ci,
                im: re * ci + im * cr,
            })
        }
        (FormValue::Complex { re, im }, FormValue::Real { value })
        | (FormValue::Real { value }, FormValue::Complex { re, im }) => Ok(FormValue::Complex {
            re: re * value,
            im: im * value,
        }),
        (left, right) => elementwise("multiplication", left, right, |left, right| left * right),
    }
}

fn divide(left: FormValue, right: FormValue) -> Result<FormValue, FormInterpretError> {
    match right {
        FormValue::Real { value } => scale(left, 1.0 / value),
        FormValue::Complex { re, im } => {
            let (lr, li) = as_complex(left)?;
            let denominator = re * re + im * im;
            Ok(FormValue::Complex {
                re: (lr * re + li * im) / denominator,
                im: (li * re - lr * im) / denominator,
            })
        }
        right => Err(type_mismatch("division", &[left, right])),
    }
}

fn scale(value: FormValue, factor: f64) -> Result<FormValue, FormInterpretError> {
    Ok(match value {
        FormValue::Real { value } => FormValue::real(value * factor),
        FormValue::Complex { re, im } => FormValue::Complex {
            re: re * factor,
            im: im * factor,
        },
        FormValue::Tensor { shape, data } => FormValue::Tensor {
            shape,
            data: data.into_iter().map(|value| value * factor).collect(),
        },
        value @ FormValue::Boolean { .. } => return Err(type_mismatch("scale", &[value])),
    })
}

fn elementwise(
    operation: &'static str,
    left: FormValue,
    right: FormValue,
    op: impl Fn(f64, f64) -> f64,
) -> Result<FormValue, FormInterpretError> {
    match (left, right) {
        (FormValue::Real { value: left }, FormValue::Real { value: right }) => {
            Ok(FormValue::real(op(left, right)))
        }
        (
            FormValue::Tensor {
                shape: left_shape,
                data: left,
            },
            FormValue::Tensor {
                shape: right_shape,
                data: right,
            },
        ) if left_shape == right_shape => Ok(FormValue::Tensor {
            shape: left_shape,
            data: left
                .into_iter()
                .zip(right)
                .map(|(left, right)| op(left, right))
                .collect(),
        }),
        (left, right) => Err(type_mismatch(operation, &[left, right])),
    }
}

fn contract(
    left: FormValue,
    right: FormValue,
    axes: &[crate::semantic::AxisContraction],
    conjugate_left: bool,
) -> Result<FormValue, FormInterpretError> {
    match (left, right) {
        (
            FormValue::Tensor {
                shape: left_shape,
                data: left,
            },
            FormValue::Tensor {
                shape: right_shape,
                data: right,
            },
        ) if axes.len() == left_shape.len()
            && axes.len() == right_shape.len()
            && left_shape == right_shape =>
        {
            Ok(FormValue::real(
                left.into_iter()
                    .zip(right)
                    .map(|(left, right)| left * right)
                    .sum(),
            ))
        }
        (FormValue::Real { value: left }, FormValue::Real { value: right }) if axes.is_empty() => {
            Ok(FormValue::real(left * right))
        }
        (FormValue::Complex { re, im }, FormValue::Complex { re: rr, im: ri })
            if axes.is_empty() =>
        {
            let left = if conjugate_left { (re, -im) } else { (re, im) };
            Ok(FormValue::Complex {
                re: left.0 * rr - left.1 * ri,
                im: left.0 * ri + left.1 * rr,
            })
        }
        (left, right) => Err(type_mismatch("contraction", &[left, right])),
    }
}

fn tensor_trace(value: FormValue, first: u8, second: u8) -> Result<FormValue, FormInterpretError> {
    let FormValue::Tensor { shape, data } = value else {
        return Err(type_mismatch("tensor trace", &[value]));
    };
    if shape.len() != 2 || first == second || first > 1 || second > 1 || shape[0] != shape[1] {
        return Err(type_mismatch(
            "tensor trace",
            &[FormValue::Tensor { shape, data }],
        ));
    }
    let extent = shape[0];
    Ok(FormValue::real(
        (0..extent).map(|index| data[index * extent + index]).sum(),
    ))
}

fn normal_component(value: FormValue, normal: &[f64]) -> Result<FormValue, FormInterpretError> {
    match value {
        FormValue::Tensor { shape, data } if shape == [normal.len()] => Ok(FormValue::real(
            data.into_iter()
                .zip(normal.iter().copied())
                .map(|(value, normal)| value * normal)
                .sum(),
        )),
        FormValue::Tensor { shape, data } if shape.len() == 2 && shape[1] == normal.len() => {
            let rows = shape[0];
            let columns = shape[1];
            Ok(FormValue::vector(
                (0..rows)
                    .map(|row| {
                        (0..columns)
                            .map(|column| data[row * columns + column] * normal[column])
                            .sum()
                    })
                    .collect(),
            ))
        }
        value => Err(type_mismatch("normal component", &[value])),
    }
}

fn index(value: FormValue, indices: &[usize]) -> Result<FormValue, FormInterpretError> {
    let FormValue::Tensor { shape, data } = value else {
        return Err(type_mismatch("index", &[value]));
    };
    if indices.len() > shape.len()
        || indices
            .iter()
            .zip(&shape)
            .any(|(index, extent)| index >= extent)
    {
        return Err(FormInterpretError::IndexOutOfBounds);
    }
    let mut offset = 0;
    for (axis, index) in indices.iter().enumerate() {
        let stride = shape[axis + 1..].iter().product::<usize>();
        offset += index * stride;
    }
    let remaining = &shape[indices.len()..];
    if remaining.is_empty() {
        Ok(FormValue::real(data[offset]))
    } else {
        let size = remaining.iter().product::<usize>();
        Ok(FormValue::Tensor {
            shape: remaining.to_vec(),
            data: data[offset..offset + size].to_vec(),
        })
    }
}

fn conjugate(value: FormValue) -> FormValue {
    match value {
        FormValue::Complex { re, im } => FormValue::Complex { re, im: -im },
        value => value,
    }
}

fn as_real(value: FormValue) -> Result<f64, FormInterpretError> {
    match value {
        FormValue::Real { value } => Ok(value),
        value => Err(type_mismatch("real scalar", &[value])),
    }
}

fn as_complex(value: FormValue) -> Result<(f64, f64), FormInterpretError> {
    match value {
        FormValue::Real { value } => Ok((value, 0.0)),
        FormValue::Complex { re, im } => Ok((re, im)),
        value => Err(type_mismatch("complex scalar", &[value])),
    }
}

fn as_index(value: FormValue) -> Result<usize, FormInterpretError> {
    let value = as_real(value)?;
    if value >= 0.0 && value.fract() == 0.0 {
        Ok(value as usize)
    } else {
        Err(FormInterpretError::IndexOutOfBounds)
    }
}

fn type_mismatch(operation: &'static str, operands: &[FormValue]) -> FormInterpretError {
    FormInterpretError::TypeMismatch {
        operation,
        operands: format!("{operands:?}"),
    }
}
