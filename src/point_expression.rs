//! Compiler-owned point expressions and cell functional integrands.
//!
//! These artifacts use the existing form, factorization and Malleus kernel representations.
//! The synthetic L2(0) test argument is removed by test differentiation: kernel outputs are
//! **point values**, without a test weight, geometry determinant or quadrature weight.
//!
//! Bundle JVP/VJP products are partial derivatives with captured inputs fixed. A consumer
//! obtains total products by executing the compiled argument DAG, applying the provider's
//! fallible JVP/VJP, and adding the parameter JVP/VJP contribution. Captures are never
//! implicitly frozen. External data may be explicitly held fixed by the consumer's request.
use crate::formulation::{derive_point_form, expression_children};
use crate::id::span_independent_digest;
use crate::scientific::{BinaryOp, CoordinateSystem, DerivativeContract};
use crate::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const POINT_EXPRESSION_KERNELS_SCHEMA: &str = "scientia-point-expression-kernels/1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PointCaptureDisposition {
    /// A case input: use its parameter product for active data, or explicitly hold it fixed.
    ExternalInput,
    /// Value and argument JVP/VJP must be provided by the bound provider. No zero fallback.
    ProviderValueAndProductsRequired,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PointCapture {
    pub input: TensorInputId,
    pub symbol: SymbolId,
    /// Original semantic expression; never an expression invented by the product runtime.
    pub definition: Option<ExprId>,
    pub provider: Option<ProviderId>,
    /// Keys into `PointExpressionKernels::arguments`, in declared provider argument order.
    pub arguments: Vec<ExprId>,
    /// Transitive source symbols, including symbols hidden behind authored definitions.
    pub dependencies: Vec<SymbolId>,
    pub disposition: PointCaptureDisposition,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointExpressionNode {
    pub expression: ExprId,
    pub form: VariationalForm,
    pub requirements: FormRequirements,
    pub factorization: OperatorFactorization,
    pub kernels: StructuredOperatorKernels,
    /// One capture-covector product per bundle, in bundle order. Use its `primal_operands`
    /// remapping, seed `dependent_operands`, and read `independent_operands` covectors.
    pub parameter_vjp: Vec<malleus::DerivativeProduct>,
    pub captures: Vec<PointCapture>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointExpressionKernels {
    pub schema: String,
    pub model: String,
    pub declaration: DeclarationId,
    /// Root point integrand (for a functional, the argument of `integrate`).
    pub expression: ExprId,
    pub domain: DomainId,
    pub root: PointExpressionNode,
    /// Shared expression DAG: each provider argument is compiled once, including nested calls.
    pub arguments: BTreeMap<ExprId, PointExpressionNode>,
    pub artifact_digest: Digest,
}

impl PointExpressionKernels {
    /// Check integrity after transport. This authenticates all node payloads and capture
    /// contracts against this artifact's digest; it is not a substitute for source trust.
    pub fn validate_identity(&self) -> Result<(), PointExpressionError> {
        let mut payload = self.clone();
        payload.artifact_digest = Digest {
            algorithm: "blake3".into(),
            hex: String::new(),
        };
        if self.schema != POINT_EXPRESSION_KERNELS_SCHEMA
            || span_independent_digest(&payload) != self.artifact_digest
        {
            return Err(error(
                "POINT_IDENTITY_MISMATCH",
                "point expression artifact changed",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{code}: {message}")]
pub struct PointExpressionError {
    pub code: &'static str,
    pub message: String,
}
fn error(code: &'static str, message: impl ToString) -> PointExpressionError {
    PointExpressionError {
        code,
        message: message.to_string(),
    }
}

pub(crate) fn symbol_definition(model: &SemanticModel, symbol: SymbolId) -> Option<ExprId> {
    model.declarations.iter().find_map(|declaration| {
        if declaration.symbol != Some(symbol) {
            return None;
        }
        match declaration.kind {
            SemanticDeclarationKind::Value { value } => value,
            SemanticDeclarationKind::Property { value }
            | SemanticDeclarationKind::ConstitutiveLaw { value }
            | SemanticDeclarationKind::Output { value, .. } => Some(value),
            _ => None,
        }
    })
}
fn declaration_value(declaration: &SemanticDeclaration) -> Option<ExprId> {
    match declaration.kind {
        SemanticDeclarationKind::Value { value } => value,
        SemanticDeclarationKind::Property { value }
        | SemanticDeclarationKind::ConstitutiveLaw { value }
        | SemanticDeclarationKind::Output { value, .. }
        | SemanticDeclarationKind::Observable { value }
        | SemanticDeclarationKind::Objective { value, .. } => Some(value),
        _ => None,
    }
}
fn closure(
    model: &SemanticModel,
    expression: ExprId,
) -> Result<(BTreeSet<ExprId>, BTreeSet<SymbolId>), PointExpressionError> {
    fn visit(
        model: &SemanticModel,
        id: ExprId,
        stack: &mut BTreeSet<ExprId>,
        expressions: &mut BTreeSet<ExprId>,
        symbols: &mut BTreeSet<SymbolId>,
    ) -> Result<(), PointExpressionError> {
        if stack.contains(&id) {
            return Err(error("POINT_EXPRESSION_CYCLE", format!("expression {id}")));
        }
        if !expressions.insert(id) {
            return Ok(());
        }
        stack.insert(id);
        let expr = model
            .expressions
            .get(id.index())
            .ok_or_else(|| error("POINT_EXPRESSION_INVALID", format!("expression {id}")))?;
        if let SemanticExprKind::Symbol { symbol } = expr.kind {
            if model.symbols.get(symbol.index()).is_none() {
                return Err(error(
                    "POINT_EXPRESSION_INVALID",
                    format!("symbol {symbol}"),
                ));
            }
            symbols.insert(symbol);
            if let Some(value) = symbol_definition(model, symbol) {
                visit(model, value, stack, expressions, symbols)?;
            }
        }
        for child in expression_children(&expr.kind) {
            visit(model, child, stack, expressions, symbols)?;
        }
        stack.remove(&id);
        Ok(())
    }
    let (mut expressions, mut symbols) = (BTreeSet::new(), BTreeSet::new());
    visit(
        model,
        expression,
        &mut BTreeSet::new(),
        &mut expressions,
        &mut symbols,
    )?;
    Ok((expressions, symbols))
}

/// Compile an authored expression or its transitive subexpression. Attribution to an unrelated
/// declaration is rejected. Domain support is explicit and checked, not inferred from a name.
pub fn compile_point_expression(
    module: &SemanticModule,
    model_name: &str,
    declaration: DeclarationId,
    expression: ExprId,
    domain: DomainId,
) -> Result<PointExpressionKernels, PointExpressionError> {
    let model = module
        .models
        .iter()
        .find(|model| model.name == model_name)
        .ok_or_else(|| error("POINT_MODEL_MISSING", model_name))?;
    let declaration = model
        .declarations
        .iter()
        .find(|item| item.id == declaration)
        .ok_or_else(|| error("POINT_DECLARATION_MISSING", format!("{declaration:?}")))?;
    let authored = declaration_value(declaration)
        .ok_or_else(|| error("POINT_DECLARATION_UNSUPPORTED", &declaration.name))?;
    if !closure(model, authored)?.0.contains(&expression) {
        return Err(error(
            "POINT_EXPRESSION_ORIGIN",
            "expression does not belong to declaration",
        ));
    }
    let support = model
        .domains
        .iter()
        .find(|item| item.id == domain)
        .ok_or_else(|| error("POINT_DOMAIN_INVALID", format!("{domain:?}")))?;
    if support.coordinates != CoordinateSystem::Cartesian {
        return Err(error(
            "POINT_COORDINATES_UNSUPPORTED",
            "point lowering currently requires Cartesian coordinates",
        ));
    }
    if declaration
        .domain
        .is_some_and(|declared| declared != domain)
    {
        return Err(error(
            "POINT_DOMAIN_MISMATCH",
            "declaration and requested support differ",
        ));
    }
    let mut arguments = BTreeMap::new();
    let root = compile_node(
        module,
        model,
        declaration,
        expression,
        domain,
        &mut arguments,
    )?;
    let mut artifact = PointExpressionKernels {
        schema: POINT_EXPRESSION_KERNELS_SCHEMA.into(),
        model: model.name.clone(),
        declaration: declaration.id,
        expression,
        domain,
        root,
        arguments,
        artifact_digest: Digest {
            algorithm: "blake3".into(),
            hex: String::new(),
        },
    };
    // Covers every executable payload, support, original identity and capture disposition.
    artifact.artifact_digest = span_independent_digest(&artifact);
    Ok(artifact)
}

/// Compile a declared `observable`/`objective` with exactly `integrate(point_expression)`.
/// Boundary/nested integrals and ambiguous multi-domain support are typed refusals.
pub fn compile_cell_functional(
    module: &SemanticModule,
    model_name: &str,
    declaration_name: &str,
) -> Result<PointExpressionKernels, PointExpressionError> {
    let model = module
        .models
        .iter()
        .find(|model| model.name == model_name)
        .ok_or_else(|| error("POINT_MODEL_MISSING", model_name))?;
    let declaration = model
        .declarations
        .iter()
        .find(|item| item.name == declaration_name)
        .ok_or_else(|| error("POINT_DECLARATION_MISSING", declaration_name))?;
    let value = match declaration.kind {
        SemanticDeclarationKind::Observable { value }
        | SemanticDeclarationKind::Objective { value, .. } => value,
        _ => {
            return Err(error(
                "POINT_FUNCTIONAL_UNSUPPORTED",
                "expected objective or observable",
            ));
        }
    };
    let expression = match &model.expressions[value.index()].kind {
        SemanticExprKind::Call { function, args } if function == "integrate" && args.len() == 1 => {
            args[0]
        }
        _ => {
            return Err(error(
                "POINT_MEASURE_UNSUPPORTED",
                "expected one cell integrate(integrand)",
            ));
        }
    };
    let domains = closure(model, expression)?
        .1
        .into_iter()
        .filter_map(|symbol| model.symbols[symbol.index()].domain)
        .collect::<BTreeSet<_>>();
    let domain = match domains.iter().copied().collect::<Vec<_>>().as_slice() {
        [domain] => *domain,
        [] if model.domains.len() == 1 => model.domains[0].id,
        _ => {
            return Err(error(
                "POINT_DOMAIN_AMBIGUOUS",
                "functional must have one unambiguous domain",
            ));
        }
    };
    compile_point_expression(module, model_name, declaration.id, expression, domain)
}

fn compile_node(
    module: &SemanticModule,
    model: &SemanticModel,
    declaration: &SemanticDeclaration,
    expression: ExprId,
    domain: DomainId,
    arguments: &mut BTreeMap<ExprId, PointExpressionNode>,
) -> Result<PointExpressionNode, PointExpressionError> {
    let (reachable, symbols) = closure(model, expression)?;
    for symbol in symbols {
        if model.symbols[symbol.index()]
            .domain
            .is_some_and(|support| support != domain)
        {
            return Err(error(
                "POINT_DOMAIN_MISMATCH",
                format!("symbol {symbol} has different support"),
            ));
        }
    }
    for id in reachable {
        match &model.expressions[id.index()].kind {
            SemanticExprKind::Call { function, .. }
                if !matches!(function.as_str(), "dot" | "inner" | "trace") =>
            {
                return Err(error(
                    "POINT_CONSTRUCT_UNSUPPORTED",
                    format!("call {function}"),
                ));
            }
            SemanticExprKind::FacetTrace { .. }
            | SemanticExprKind::Jump { .. }
            | SemanticExprKind::Average { .. }
            | SemanticExprKind::NormalComponent { .. }
            | SemanticExprKind::String { .. } => {
                return Err(error("POINT_MEASURE_UNSUPPORTED", "non-cell expression"));
            }
            SemanticExprKind::ProviderCall { provider, .. } => {
                let provider = &model.providers[provider.index()];
                if !matches!(
                    provider.differentiability,
                    DerivativeContract::Symbolic
                        | DerivativeContract::Automatic
                        | DerivativeContract::AnalyticProvided
                ) {
                    return Err(error(
                        "POINT_PROVIDER_DERIVATIVE_UNAVAILABLE",
                        &provider.name,
                    ));
                }
            }
            _ => {}
        }
    }
    let form = derive_point_form(module, model, declaration, expression, domain, true)
        .map_err(|e| error("POINT_FORM_UNSUPPORTED", e))?;
    let requirements = infer_form_requirements(module, &form)
        .map_err(|e| error("POINT_REQUIREMENTS_UNSUPPORTED", e))?;
    let factorization = factor_operator(&form, &requirements)
        .map_err(|e| error("POINT_FACTORIZATION_UNSUPPORTED", e))?;
    let kernels =
        lower_operator_kernels(&factorization).map_err(|e| error("POINT_KERNEL_UNSUPPORTED", e))?;
    let mut captures = Vec::new();
    for integral in &factorization.integrals {
        for input in &integral.primal.inputs {
            if input.source == InputSourceRequirement::Basis {
                continue;
            }
            let (definition, provider, args, dependencies, disposition) = match input.source {
                InputSourceRequirement::ExternalValue => (
                    None,
                    None,
                    vec![],
                    vec![input.binding.symbol],
                    PointCaptureDisposition::ExternalInput,
                ),
                InputSourceRequirement::ModelDefinedProperty { definition } => {
                    let SemanticExprKind::ProviderCall { provider, args } =
                        &form.expressions[definition.index()].kind
                    else {
                        return Err(error(
                            "POINT_CAPTURE_UNSUPPORTED",
                            "opaque definition survived expansion",
                        ));
                    };
                    let dependencies = closure(model, definition)?.1.into_iter().collect();
                    (
                        Some(definition),
                        Some(*provider),
                        args.clone(),
                        dependencies,
                        PointCaptureDisposition::ProviderValueAndProductsRequired,
                    )
                }
                _ => {
                    return Err(error(
                        "POINT_CAPTURE_UNSUPPORTED",
                        "opaque constitutive/value input survived expansion",
                    ));
                }
            };
            for arg in &args {
                if !arguments.contains_key(arg) {
                    let compiled =
                        compile_node(module, model, declaration, *arg, domain, arguments)?;
                    arguments.insert(*arg, compiled);
                }
            }
            captures.push(PointCapture {
                input: input.id,
                symbol: input.binding.symbol,
                definition,
                provider,
                arguments: args,
                dependencies,
                disposition,
            });
        }
    }
    let parameter_vjp = kernels
        .bundles
        .iter()
        .map(|bundle| {
            malleus::differentiate(
                &bundle.module.kernels[bundle.primal_kernel_index],
                &malleus::DerivativeRequest {
                    mode: malleus::DerivativeMode::Vjp,
                    independent_operands: bundle
                        .parameter
                        .independent_operands
                        .iter()
                        .map(|operand| operand.primal)
                        .collect(),
                    dependent_operands: vec![bundle.primal_output],
                },
            )
            .map_err(|e| error("POINT_CAPTURE_VJP_UNSUPPORTED", e))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PointExpressionNode {
        expression,
        form,
        requirements,
        factorization,
        kernels,
        parameter_vjp,
        captures,
    })
}

/// Caller evidence for polynomial degree in physical space at fixed time. Coordinates and time
/// have no name-based special cases: bind spatial coordinates with degree 1 and a fixed-time
/// datum with degree 0 explicitly. Provider evidence is call-specific, not provider-wide.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolynomialDegreeBindings {
    pub symbols: BTreeMap<SymbolId, u16>,
    pub provider_calls: BTreeMap<ExprId, u16>,
}

/// Conservative degree bound; `None` means unknown, never an exactness claim. Provider/case
/// values require caller evidence. Spatial differentiation reduces degree; time differentiation
/// remains unknown because a fixed-time degree does not bound the rate's spatial degree.
pub fn infer_expression_polynomial_degree(
    model: &SemanticModel,
    expression: ExprId,
    degrees: &PolynomialDegreeBindings,
) -> Option<u16> {
    fn infer(
        model: &SemanticModel,
        id: ExprId,
        degrees: &PolynomialDegreeBindings,
        stack: &mut BTreeSet<ExprId>,
    ) -> Option<u16> {
        if !stack.insert(id) {
            return None;
        }
        let expr = model.expressions.get(id.index())?;
        let result = match &expr.kind {
            SemanticExprKind::Number { .. } => Some(0),
            SemanticExprKind::Symbol { symbol } => match symbol_definition(model, *symbol) {
                Some(value) => infer(model, value, degrees, stack),
                None => degrees.symbols.get(symbol).copied(),
            },
            SemanticExprKind::ProviderCall { .. } => degrees.provider_calls.get(&id).copied(),
            SemanticExprKind::Unary { arg, .. }
            | SemanticExprKind::Conjugate { value: arg }
            | SemanticExprKind::TensorTrace { value: arg, .. } => {
                infer(model, *arg, degrees, stack)
            }
            SemanticExprKind::Binary { op, lhs, rhs } => {
                let left = infer(model, *lhs, degrees, stack);
                let right = infer(model, *rhs, degrees, stack);
                match op {
                    BinaryOp::Add | BinaryOp::Sub => left.zip(right).map(|(a, b)| a.max(b)),
                    BinaryOp::Mul => left.zip(right).and_then(|(a, b)| a.checked_add(b)),
                    BinaryOp::Div if matches!(model.expressions[rhs.index()].kind, SemanticExprKind::Number { value, .. } if value != 0.0) => {
                        left
                    }
                    BinaryOp::Pow => match model.expressions[rhs.index()].kind {
                        SemanticExprKind::Number { value, .. }
                            if value >= 0.0 && value <= u16::MAX as f64 && value.fract() == 0.0 =>
                        {
                            left.and_then(|degree| degree.checked_mul(value as u16))
                        }
                        _ => None,
                    },
                    _ => None,
                }
            }
            SemanticExprKind::Contraction { lhs, rhs, .. } => infer(model, *lhs, degrees, stack)
                .zip(infer(model, *rhs, degrees, stack))
                .and_then(|(a, b)| a.checked_add(b)),
            SemanticExprKind::Differential { operator, arg }
                if *operator != DifferentialOperator::TimeDerivative =>
            {
                infer(model, *arg, degrees, stack).map(|degree| degree.saturating_sub(1))
            }
            SemanticExprKind::Vector { elements } => elements
                .iter()
                .map(|arg| infer(model, *arg, degrees, stack))
                .collect::<Option<Vec<_>>>()
                .and_then(|values| values.into_iter().max()),
            SemanticExprKind::Index { value, indices }
                if indices.iter().all(|id| {
                    matches!(
                        model.expressions[id.index()].kind,
                        SemanticExprKind::Number { .. }
                    )
                }) =>
            {
                infer(model, *value, degrees, stack)
            }
            SemanticExprKind::Call { function, args }
                if matches!(function.as_str(), "dot" | "inner") && args.len() == 2 =>
            {
                infer(model, args[0], degrees, stack)
                    .zip(infer(model, args[1], degrees, stack))
                    .and_then(|(a, b)| a.checked_add(b))
            }
            _ => None,
        };
        stack.remove(&id);
        result
    }
    infer(model, expression, degrees, &mut BTreeSet::new())
}
