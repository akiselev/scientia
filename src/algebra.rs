use crate::scientific::{BinaryOp, Expr, ScientificError};
use crate::source::SourceSpan;

pub(crate) fn differentiate_expr(expr: &Expr, variable: &str) -> Result<Expr, ScientificError> {
    let span = expr.span();
    let algebra = to_algebra(expr)?;
    let derivative = algebra
        .differentiate(variable, resolvent::AlgebraBudget::default())
        .map_err(error)?;
    from_algebra(&derivative, span)
}

fn to_algebra(expr: &Expr) -> Result<resolvent::Expr, ScientificError> {
    Ok(match expr {
        Expr::Number {
            value, unit: None, ..
        } => resolvent::Expr::Rational(
            resolvent::rational_from_f64(*value)
                .ok_or_else(|| property("non-finite numeric literal in exact algebra"))?,
        ),
        Expr::Name { name, .. } => resolvent::Expr::symbol(name),
        Expr::Unary { arg, .. } => {
            resolvent::Expr::mul([resolvent::Expr::integer(-1), to_algebra(arg)?])
        }
        Expr::Binary { op, lhs, rhs, .. } => match op {
            BinaryOp::Add => resolvent::Expr::add([to_algebra(lhs)?, to_algebra(rhs)?]),
            BinaryOp::Sub => resolvent::Expr::add([
                to_algebra(lhs)?,
                resolvent::Expr::mul([resolvent::Expr::integer(-1), to_algebra(rhs)?]),
            ]),
            BinaryOp::Mul => resolvent::Expr::mul([to_algebra(lhs)?, to_algebra(rhs)?]),
            BinaryOp::Div => resolvent::Expr::mul([to_algebra(lhs)?, to_algebra(rhs)?.pow(-1)]),
            BinaryOp::Pow => {
                let Expr::Number { value, .. } = &**rhs else {
                    return Err(property("exact differentiation requires an integer power"));
                };
                if value.fract() != 0.0
                    || *value < f64::from(i32::MIN)
                    || *value > f64::from(i32::MAX)
                {
                    return Err(property("exact differentiation requires an integer power"));
                }
                to_algebra(lhs)?.pow(*value as i32)
            }
            BinaryOp::Eq | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                resolvent::Expr::integer(0)
            }
        },
        Expr::Call { function, args, .. } => resolvent::Expr::function(
            function,
            args.iter().map(to_algebra).collect::<Result<Vec<_>, _>>()?,
        ),
        Expr::Number { unit: Some(_), .. }
        | Expr::String { .. }
        | Expr::Index { .. }
        | Expr::Vector { .. } => {
            return Err(property(
                "expression is outside consumer-neutral exact algebra",
            ));
        }
    })
}

fn from_algebra(expr: &resolvent::Expr, span: SourceSpan) -> Result<Expr, ScientificError> {
    Ok(match expr {
        resolvent::Expr::Rational(value) => Expr::Number {
            value: resolvent::rational_to_f64(value)
                .ok_or_else(|| property("exact derivative does not fit finite f64"))?,
            unit: None,
            span,
        },
        resolvent::Expr::Symbol(name) => Expr::Name {
            name: name.clone(),
            span,
        },
        resolvent::Expr::Add(terms) => fold(terms, BinaryOp::Add, 0.0, span)?,
        resolvent::Expr::Mul(factors) => fold(factors, BinaryOp::Mul, 1.0, span)?,
        resolvent::Expr::Pow { base, exponent } => Expr::Binary {
            op: BinaryOp::Pow,
            lhs: Box::new(from_algebra(base, span)?),
            rhs: Box::new(Expr::Number {
                value: f64::from(*exponent),
                unit: None,
                span,
            }),
            span,
        },
        resolvent::Expr::Function { name, args } => Expr::Call {
            function: name.clone(),
            args: args
                .iter()
                .map(|arg| from_algebra(arg, span))
                .collect::<Result<_, _>>()?,
            span,
        },
    })
}

fn fold(
    expressions: &[resolvent::Expr],
    op: BinaryOp,
    identity: f64,
    span: SourceSpan,
) -> Result<Expr, ScientificError> {
    let mut values = expressions.iter().map(|expr| from_algebra(expr, span));
    let Some(first) = values.next() else {
        return Ok(Expr::Number {
            value: identity,
            unit: None,
            span,
        });
    };
    values.try_fold(first?, |lhs, rhs| {
        Ok(Expr::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs?),
            span,
        })
    })
}

fn property(message: impl Into<String>) -> ScientificError {
    ScientificError::Property(message.into())
}

fn error(error: resolvent::AlgebraError) -> ScientificError {
    property(error.to_string())
}
