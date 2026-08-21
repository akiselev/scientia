//! FC5 lowering from indexed QFunctions to Malleus structured kernel bundles.

use crate::id::{Digest, span_independent_digest};
use crate::tensor::{
    DerivativeStateSemantics, OperatorFactorization, QFunctionInput, QFunctionProgram, TensorAxis,
    TensorAxisId, TensorAxisRole, TensorBinaryOp, TensorInputId, TensorInputRole,
    TensorReductionOp, TensorScalarExpr, TensorScalarSemantics, TensorUnaryOp,
};
use malleus::{
    AccessMode, AxisId, BinaryOp, CompareOp, DerivativeOperand, DerivativeRequest, FmaPolicy,
    IndexExpr, IndexingMap, IterationDomain, IteratorKind, KernelOperand, KernelRegion,
    NumericPolicy, OperandId, Predicate, Reassociation, ReductionOp, ReductionOrder, ScalarExpr,
    ScalarType, Statement, StructuredKernel, StructuredModule, UnaryOp, differentiate,
    validate_module,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

pub const STRUCTURED_KERNEL_BUNDLE_SCHEMA: &str = "scientia-structured-kernel-bundle/1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredKernelLoweringMethod {
    IndexedQFunctionMalleusAd,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuredInputOperand {
    pub input: TensorInputId,
    pub operand: OperandId,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuredDerivativeContract {
    pub kernel_index: usize,
    pub mode: malleus::DerivativeMode,
    pub purpose: StructuredDerivativePurpose,
    pub independent_operands: Vec<DerivativeOperand>,
    pub dependent_operands: Vec<DerivativeOperand>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredDerivativePurpose {
    StateDirection,
    StateAdjoint,
    FrozenParameterDirection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredDerivativeEvidence {
    SourceSymbolicJvpReceipt,
    StructuredChainRuleIdentity,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuredKernelReceipt {
    pub schema: String,
    pub artifact_digest: Digest,
    pub source_factorization_digest: Digest,
    pub source_primal_digest: Digest,
    pub source_symbolic_jvp_digest: Digest,
    pub integral_index: usize,
    pub output_index: usize,
    pub lowering: StructuredKernelLoweringMethod,
    pub complex_convention: crate::FormComplexConvention,
    pub state_semantics: DerivativeStateSemantics,
    pub derivative_evidence: Vec<StructuredDerivativeEvidence>,
    pub active_inputs: Vec<TensorInputId>,
    pub parameter_inputs: Vec<TensorInputId>,
    pub numeric_policy: StructuredNumericPolicyReceipt,
}

/// Serializable projection of the Malleus-owned policy used in artifact identity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuredNumericPolicyReceipt {
    pub scalar_type: String,
    pub fma: String,
    pub reassociation: String,
    pub reduction_order: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructuredPointKernelBundle {
    pub integral_index: usize,
    pub output_index: usize,
    pub module: StructuredModule,
    pub primal_kernel_index: usize,
    pub primal_inputs: Vec<StructuredInputOperand>,
    pub primal_output: OperandId,
    pub jvp: StructuredDerivativeContract,
    pub vjp: StructuredDerivativeContract,
    pub parameter: StructuredDerivativeContract,
    pub receipt: StructuredKernelReceipt,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructuredOperatorKernels {
    pub schema: String,
    pub source_factorization_digest: Digest,
    pub artifact_digest: Digest,
    pub bundles: Vec<StructuredPointKernelBundle>,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum StructuredLoweringError {
    #[error("KERNEL_SCALAR_POLICY: QFunction `{0}` does not use supported real64 semantics")]
    ScalarPolicy(String),
    #[error("KERNEL_AXIS: {0}")]
    Axis(String),
    #[error("KERNEL_SHAPE: {0}")]
    Shape(String),
    #[error("KERNEL_INPUT: QFunction input {0:?} is missing")]
    MissingInput(TensorInputId),
    #[error("KERNEL_INPUT: QFunction input {0:?} is declared more than once")]
    DuplicateInput(TensorInputId),
    #[error("KERNEL_SOURCE_JVP: {0}")]
    SourceJvp(String),
    #[error("KERNEL_VALIDATION: {0}")]
    Validation(#[from] malleus::ValidationError),
    #[error("KERNEL_DIFFERENTIATION: {0}")]
    Differentiation(#[from] malleus::DifferentiationError),
}

/// Lower each supported point output in an FC4 factorization into primal, state-JVP, state-VJP,
/// and frozen-input parameter-JVP kernels. Each output gets one module so its iteration domain
/// and reduction effects remain explicit and independently schedulable.
pub fn lower_operator_kernels(
    factorization: &OperatorFactorization,
) -> Result<StructuredOperatorKernels, StructuredLoweringError> {
    lower_operator_kernels_with_policy(factorization, NumericPolicy::default())
}

pub fn lower_operator_kernels_with_policy(
    factorization: &OperatorFactorization,
    numeric_policy: NumericPolicy,
) -> Result<StructuredOperatorKernels, StructuredLoweringError> {
    if numeric_policy.scalar_type != ScalarType::F64 {
        return Err(StructuredLoweringError::ScalarPolicy(format!(
            "{} requires real64 but the requested Malleus policy is {:?}",
            factorization.form, numeric_policy.scalar_type
        )));
    }
    let mut bundles = Vec::new();
    for integral in &factorization.integrals {
        for output_index in 0..integral.primal.outputs.len() {
            bundles.push(lower_output(
                factorization,
                integral.integral_index,
                output_index,
                &integral.primal,
                &integral.jvp,
                numeric_policy,
            )?);
        }
    }
    let artifact_digest = span_independent_digest(&OperatorBundleDigestPayload {
        schema: STRUCTURED_KERNEL_BUNDLE_SCHEMA,
        source_factorization_digest: &factorization.artifact_digest,
        bundles: bundles.iter().map(|bundle| &bundle.receipt).collect(),
    });
    Ok(StructuredOperatorKernels {
        schema: STRUCTURED_KERNEL_BUNDLE_SCHEMA.into(),
        source_factorization_digest: factorization.artifact_digest.clone(),
        artifact_digest,
        bundles,
    })
}

#[derive(Serialize)]
struct OperatorBundleDigestPayload<'a> {
    schema: &'static str,
    source_factorization_digest: &'a Digest,
    bundles: Vec<&'a StructuredKernelReceipt>,
}

fn lower_output(
    factorization: &OperatorFactorization,
    integral_index: usize,
    output_index: usize,
    primal_program: &QFunctionProgram,
    symbolic_jvp: &QFunctionProgram,
    numeric_policy: NumericPolicy,
) -> Result<StructuredPointKernelBundle, StructuredLoweringError> {
    if primal_program.scalar_semantics != TensorScalarSemantics::Real64 {
        return Err(StructuredLoweringError::ScalarPolicy(
            primal_program.name.clone(),
        ));
    }
    let derivative_receipt = symbolic_jvp.derivative_receipt.as_ref().ok_or_else(|| {
        StructuredLoweringError::SourceJvp(format!(
            "QFunction `{}` has no derivative receipt",
            symbolic_jvp.name
        ))
    })?;
    if derivative_receipt.primal_artifact_digest != primal_program.artifact_digest
        || symbolic_jvp.receipt.source_tensor_program_digest
            != primal_program.receipt.source_tensor_program_digest
        || symbolic_jvp.outputs.len() != primal_program.outputs.len()
        || symbolic_jvp
            .outputs
            .iter()
            .zip(&primal_program.outputs)
            .any(|(derivative, primal)| {
                derivative.binding != primal.binding
                    || derivative.side != primal.side
                    || derivative.shape != primal.shape
                    || derivative.free_axes != primal.free_axes
                    || derivative.basis_adjoint != primal.basis_adjoint
            })
    {
        return Err(StructuredLoweringError::SourceJvp(format!(
            "QFunction `{}` is not digest-linked and shape-aligned with `{}`",
            symbolic_jvp.name, primal_program.name
        )));
    }
    let output = primal_program.outputs.get(output_index).ok_or_else(|| {
        StructuredLoweringError::Shape(format!(
            "QFunction `{}` has no output at index {output_index}",
            primal_program.name
        ))
    })?;
    let (mut primal, primal_inputs, primal_output) = lower_primal_output(
        primal_program,
        output_index,
        &output.shape,
        &output.free_axes,
        &output.expression,
        numeric_policy,
    )?;
    let primal_base_name = primal_program
        .name
        .strip_suffix("::primal")
        .unwrap_or(&primal_program.name);
    let base_name = format!("{primal_base_name}::output_{output_index}");
    primal.name = format!("{base_name}::primal");

    let mut roles = BTreeMap::new();
    for input in &primal_program.inputs {
        if roles.insert(input.id, input.role).is_some() {
            return Err(StructuredLoweringError::DuplicateInput(input.id));
        }
    }
    let input_by_operand = primal_inputs
        .iter()
        .map(|binding| (binding.operand, binding.input))
        .collect::<BTreeMap<_, _>>();
    let active_operands = primal_inputs
        .iter()
        .filter(|binding| roles[&binding.input] == TensorInputRole::Active)
        .map(|binding| binding.operand)
        .collect::<Vec<_>>();
    let parameter_operands = primal_inputs
        .iter()
        .filter(|binding| roles[&binding.input] != TensorInputRole::Active)
        .map(|binding| binding.operand)
        .collect::<Vec<_>>();
    let dependent = vec![primal_output];

    let mut jvp = differentiate(
        &primal,
        &DerivativeRequest {
            mode: malleus::DerivativeMode::Jvp,
            independent_operands: active_operands.clone(),
            dependent_operands: dependent.clone(),
        },
    )?;
    jvp.kernel.name = format!("{base_name}::jvp");
    let mut vjp = differentiate(
        &primal,
        &DerivativeRequest {
            mode: malleus::DerivativeMode::Vjp,
            independent_operands: active_operands.clone(),
            dependent_operands: dependent.clone(),
        },
    )?;
    vjp.kernel.name = format!("{base_name}::vjp");
    let mut parameter = differentiate(
        &primal,
        &DerivativeRequest {
            mode: malleus::DerivativeMode::Jvp,
            independent_operands: parameter_operands.clone(),
            dependent_operands: dependent,
        },
    )?;
    parameter.kernel.name = format!("{base_name}::parameter");

    let module = StructuredModule {
        name: base_name,
        kernels: vec![primal, jvp.kernel, vjp.kernel, parameter.kernel],
    };
    validate_module(module.clone())?;

    let active_inputs = active_operands
        .iter()
        .map(|operand| input_by_operand[operand])
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let parameter_inputs = parameter_operands
        .iter()
        .map(|operand| input_by_operand[operand])
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let numeric_policy_receipt = policy_receipt(numeric_policy);
    let artifact_digest = span_independent_digest(&KernelReceiptDigestPayload {
        schema: STRUCTURED_KERNEL_BUNDLE_SCHEMA,
        source_factorization_digest: &factorization.artifact_digest,
        source_primal_digest: &primal_program.artifact_digest,
        source_symbolic_jvp_digest: &symbolic_jvp.artifact_digest,
        integral_index,
        output_index,
        lowering: StructuredKernelLoweringMethod::IndexedQFunctionMalleusAd,
        complex_convention: factorization.receipt.complex_convention,
        state_semantics: DerivativeStateSemantics::Stateless,
        derivative_evidence: &[
            StructuredDerivativeEvidence::SourceSymbolicJvpReceipt,
            StructuredDerivativeEvidence::StructuredChainRuleIdentity,
        ],
        active_inputs: &active_inputs,
        parameter_inputs: &parameter_inputs,
        numeric_policy: numeric_policy_receipt.clone(),
    });
    let receipt = StructuredKernelReceipt {
        schema: STRUCTURED_KERNEL_BUNDLE_SCHEMA.into(),
        artifact_digest,
        source_factorization_digest: factorization.artifact_digest.clone(),
        source_primal_digest: primal_program.artifact_digest.clone(),
        source_symbolic_jvp_digest: symbolic_jvp.artifact_digest.clone(),
        integral_index,
        output_index,
        lowering: StructuredKernelLoweringMethod::IndexedQFunctionMalleusAd,
        complex_convention: factorization.receipt.complex_convention,
        state_semantics: DerivativeStateSemantics::Stateless,
        derivative_evidence: vec![
            StructuredDerivativeEvidence::SourceSymbolicJvpReceipt,
            StructuredDerivativeEvidence::StructuredChainRuleIdentity,
        ],
        active_inputs,
        parameter_inputs,
        numeric_policy: numeric_policy_receipt,
    };

    Ok(StructuredPointKernelBundle {
        integral_index,
        output_index,
        module,
        primal_kernel_index: 0,
        primal_inputs,
        primal_output,
        jvp: StructuredDerivativeContract {
            kernel_index: 1,
            mode: jvp.mode,
            purpose: StructuredDerivativePurpose::StateDirection,
            independent_operands: jvp.independent_operands,
            dependent_operands: jvp.dependent_operands,
        },
        vjp: StructuredDerivativeContract {
            kernel_index: 2,
            mode: vjp.mode,
            purpose: StructuredDerivativePurpose::StateAdjoint,
            independent_operands: vjp.independent_operands,
            dependent_operands: vjp.dependent_operands,
        },
        parameter: StructuredDerivativeContract {
            kernel_index: 3,
            mode: parameter.mode,
            purpose: StructuredDerivativePurpose::FrozenParameterDirection,
            independent_operands: parameter.independent_operands,
            dependent_operands: parameter.dependent_operands,
        },
        receipt,
    })
}

#[derive(Serialize)]
struct KernelReceiptDigestPayload<'a> {
    schema: &'static str,
    source_factorization_digest: &'a Digest,
    source_primal_digest: &'a Digest,
    source_symbolic_jvp_digest: &'a Digest,
    integral_index: usize,
    output_index: usize,
    lowering: StructuredKernelLoweringMethod,
    complex_convention: crate::FormComplexConvention,
    state_semantics: DerivativeStateSemantics,
    derivative_evidence: &'a [StructuredDerivativeEvidence],
    active_inputs: &'a [TensorInputId],
    parameter_inputs: &'a [TensorInputId],
    numeric_policy: StructuredNumericPolicyReceipt,
}

fn policy_receipt(policy: NumericPolicy) -> StructuredNumericPolicyReceipt {
    StructuredNumericPolicyReceipt {
        scalar_type: match policy.scalar_type {
            ScalarType::F32 => "f32",
            ScalarType::F64 => "f64",
        }
        .into(),
        fma: match policy.fma {
            FmaPolicy::Forbidden => "forbidden",
            FmaPolicy::Allowed => "allowed",
        }
        .into(),
        reassociation: match policy.reassociation {
            Reassociation::Forbidden => "forbidden",
            Reassociation::Allowed => "allowed",
        }
        .into(),
        reduction_order: match policy.reduction_order {
            ReductionOrder::Canonical => "canonical",
            ReductionOrder::ScheduleDefined => "schedule_defined",
        }
        .into(),
    }
}

fn lower_primal_output(
    program: &QFunctionProgram,
    output_index: usize,
    output_shape: &[usize],
    free_axes: &[TensorAxis],
    expression: &TensorScalarExpr,
    numeric_policy: NumericPolicy,
) -> Result<(StructuredKernel, Vec<StructuredInputOperand>, OperandId), StructuredLoweringError> {
    let mut axes = Vec::new();
    let mut axis_definitions = BTreeMap::new();
    for axis in free_axes {
        insert_axis(*axis, &mut axes, &mut axis_definitions)?;
    }
    let expression = peel_reductions(expression, &mut axes, &mut axis_definitions)?;
    let axis_ids = axes
        .iter()
        .enumerate()
        .map(|(index, axis)| (axis.id, AxisId::new(index)))
        .collect::<BTreeMap<_, _>>();

    let mut uses = BTreeMap::<TensorInputId, Vec<Vec<TensorAxisId>>>::new();
    collect_input_uses(expression, &mut uses);
    let mut operands = Vec::new();
    let mut indexing_maps = Vec::new();
    let mut primal_inputs = Vec::new();
    let mut operand_by_access = BTreeMap::new();
    for input in &program.inputs {
        let Some(accesses) = uses.get(&input.id) else {
            continue;
        };
        for (access_index, indices) in accesses.iter().enumerate() {
            validate_input_shape(input, indices, &axis_definitions)?;
            let operand = OperandId::new(operands.len());
            operands.push(KernelOperand::tensor(
                format!("input_{}_access_{access_index}", input.id.0),
                input.shape.clone(),
                AccessMode::Read,
            ));
            indexing_maps.push(IndexingMap::new(
                operand,
                indices
                    .iter()
                    .map(|axis| {
                        axis_ids
                            .get(axis)
                            .copied()
                            .map(IndexExpr::axis)
                            .ok_or_else(|| {
                                StructuredLoweringError::Axis(format!(
                                    "input {:?} references undeclared tensor axis {}",
                                    input.id, axis.0
                                ))
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ));
            operand_by_access.insert((input.id, indices.clone()), operand);
            primal_inputs.push(StructuredInputOperand {
                input: input.id,
                operand,
            });
        }
    }

    if output_shape.len() != free_axes.len()
        || output_shape
            .iter()
            .zip(free_axes)
            .any(|(extent, axis)| *extent != axis.extent || axis.role != TensorAxisRole::Free)
    {
        return Err(StructuredLoweringError::Shape(format!(
            "QFunction `{}` output {output_index} shape/free-axis mismatch",
            program.name
        )));
    }
    let output = OperandId::new(operands.len());
    let has_reduction = axes
        .iter()
        .any(|axis| axis.role == TensorAxisRole::Reduction);
    operands.push(KernelOperand::tensor(
        format!("output_{output_index}"),
        output_shape.to_vec(),
        if has_reduction {
            AccessMode::Reduce(ReductionOp::Add)
        } else {
            AccessMode::Write
        },
    ));
    indexing_maps.push(IndexingMap::new(
        output,
        free_axes
            .iter()
            .map(|axis| IndexExpr::axis(axis_ids[&axis.id]))
            .collect::<Vec<_>>(),
    ));
    let value = lower_expression(expression, &operand_by_access, &axis_ids)?;
    let kernel = StructuredKernel {
        name: program.name.clone(),
        iteration_domain: IterationDomain::new(
            axes.iter().map(|axis| axis.extent).collect::<Vec<_>>(),
        ),
        iterators: axes
            .iter()
            .map(|axis| match axis.role {
                TensorAxisRole::Free => IteratorKind::Parallel,
                TensorAxisRole::Reduction => IteratorKind::Reduction,
            })
            .collect(),
        operands,
        indexing_maps,
        body: KernelRegion {
            statements: vec![Statement::Store {
                operand: output,
                value,
            }],
        },
        numeric_policy,
    };
    Ok((kernel, primal_inputs, output))
}

fn insert_axis(
    axis: TensorAxis,
    axes: &mut Vec<TensorAxis>,
    definitions: &mut BTreeMap<TensorAxisId, TensorAxis>,
) -> Result<(), StructuredLoweringError> {
    if axis.extent == 0 {
        return Err(StructuredLoweringError::Axis(format!(
            "tensor axis {} has zero extent",
            axis.id.0
        )));
    }
    if let Some(previous) = definitions.get(&axis.id) {
        if previous != &axis {
            return Err(StructuredLoweringError::Axis(format!(
                "tensor axis {} has conflicting definitions",
                axis.id.0
            )));
        }
        return Ok(());
    }
    definitions.insert(axis.id, axis);
    axes.push(axis);
    Ok(())
}

fn peel_reductions<'a>(
    mut expression: &'a TensorScalarExpr,
    axes: &mut Vec<TensorAxis>,
    definitions: &mut BTreeMap<TensorAxisId, TensorAxis>,
) -> Result<&'a TensorScalarExpr, StructuredLoweringError> {
    while let TensorScalarExpr::Reduction {
        op: TensorReductionOp::Sum,
        axis,
        expression: body,
    } = expression
    {
        if axis.role != TensorAxisRole::Reduction {
            return Err(StructuredLoweringError::Axis(format!(
                "reduction axis {} is not marked reduction",
                axis.id.0
            )));
        }
        insert_axis(*axis, axes, definitions)?;
        expression = body;
    }
    if contains_reduction(expression) {
        return Err(StructuredLoweringError::Axis(
            "only an enclosing nest of sum reductions can lower into one structured kernel".into(),
        ));
    }
    Ok(expression)
}

fn contains_reduction(expression: &TensorScalarExpr) -> bool {
    match expression {
        TensorScalarExpr::Reduction {
            op: TensorReductionOp::Sum,
            ..
        } => true,
        TensorScalarExpr::Unary { arg, .. } => contains_reduction(arg),
        TensorScalarExpr::Binary { lhs, rhs, .. } => {
            contains_reduction(lhs) || contains_reduction(rhs)
        }
        _ => false,
    }
}

fn collect_input_uses(
    expression: &TensorScalarExpr,
    uses: &mut BTreeMap<TensorInputId, Vec<Vec<TensorAxisId>>>,
) {
    match expression {
        TensorScalarExpr::Input { input, indices } => {
            let accesses = uses.entry(*input).or_default();
            if !accesses.contains(indices) {
                accesses.push(indices.clone());
                accesses.sort();
            }
        }
        TensorScalarExpr::Unary { arg, .. } => collect_input_uses(arg, uses),
        TensorScalarExpr::Binary { lhs, rhs, .. } => {
            collect_input_uses(lhs, uses);
            collect_input_uses(rhs, uses)
        }
        TensorScalarExpr::Reduction { expression, .. } => collect_input_uses(expression, uses),
        _ => {}
    }
}

fn validate_input_shape(
    input: &QFunctionInput,
    indices: &[TensorAxisId],
    axes: &BTreeMap<TensorAxisId, TensorAxis>,
) -> Result<(), StructuredLoweringError> {
    if input.shape.len() != indices.len() {
        return Err(StructuredLoweringError::Shape(format!(
            "input {:?} shape {:?} is indexed by {} axes",
            input.id,
            input.shape,
            indices.len()
        )));
    }
    for (dimension, (extent, axis)) in input.shape.iter().zip(indices).enumerate() {
        let axis = axes.get(axis).ok_or_else(|| {
            StructuredLoweringError::Axis(format!(
                "input {:?} dimension {dimension} references undeclared axis {}",
                input.id, axis.0
            ))
        })?;
        if *extent != axis.extent {
            return Err(StructuredLoweringError::Shape(format!(
                "input {:?} dimension {dimension} has extent {extent}, axis {} has extent {}",
                input.id, axis.id.0, axis.extent
            )));
        }
    }
    Ok(())
}

fn lower_expression(
    expression: &TensorScalarExpr,
    operands: &BTreeMap<(TensorInputId, Vec<TensorAxisId>), OperandId>,
    axes: &BTreeMap<TensorAxisId, AxisId>,
) -> Result<ScalarExpr, StructuredLoweringError> {
    Ok(match expression {
        TensorScalarExpr::Constant { value } => ScalarExpr::Constant(*value),
        TensorScalarExpr::Input { input, indices } => ScalarExpr::Load(
            operands
                .get(&(*input, indices.clone()))
                .copied()
                .ok_or(StructuredLoweringError::MissingInput(*input))?,
        ),
        TensorScalarExpr::Unary { op, arg } => ScalarExpr::unary(
            match op {
                TensorUnaryOp::Neg => UnaryOp::Neg,
                TensorUnaryOp::Abs => UnaryOp::Abs,
                TensorUnaryOp::Sqrt => UnaryOp::Sqrt,
                TensorUnaryOp::Exp => UnaryOp::Exp,
                TensorUnaryOp::Ln => UnaryOp::Ln,
                TensorUnaryOp::Sin => UnaryOp::Sin,
                TensorUnaryOp::Cos => UnaryOp::Cos,
                TensorUnaryOp::Tan => UnaryOp::Tan,
                TensorUnaryOp::Floor => UnaryOp::Floor,
                TensorUnaryOp::Ceil => UnaryOp::Ceil,
            },
            lower_expression(arg, operands, axes)?,
        ),
        TensorScalarExpr::Binary { op, lhs, rhs } => ScalarExpr::binary(
            match op {
                TensorBinaryOp::Add => BinaryOp::Add,
                TensorBinaryOp::Sub => BinaryOp::Sub,
                TensorBinaryOp::Mul => BinaryOp::Mul,
                TensorBinaryOp::Div => BinaryOp::Div,
                TensorBinaryOp::Pow => BinaryOp::Pow,
                TensorBinaryOp::Min => BinaryOp::Min,
                TensorBinaryOp::Max => BinaryOp::Max,
                TensorBinaryOp::Atan2 => BinaryOp::Atan2,
            },
            lower_expression(lhs, operands, axes)?,
            lower_expression(rhs, operands, axes)?,
        ),
        TensorScalarExpr::IndexEqual { lhs, rhs } => ScalarExpr::Select {
            condition: Box::new(Predicate::Compare {
                op: CompareOp::Eq,
                lhs: Box::new(ScalarExpr::Index(*axes.get(lhs).ok_or_else(|| {
                    StructuredLoweringError::Axis(format!(
                        "index equality references undeclared axis {}",
                        lhs.0
                    ))
                })?)),
                rhs: Box::new(ScalarExpr::Index(*axes.get(rhs).ok_or_else(|| {
                    StructuredLoweringError::Axis(format!(
                        "index equality references undeclared axis {}",
                        rhs.0
                    ))
                })?)),
            }),
            if_true: Box::new(ScalarExpr::Constant(1.0)),
            if_false: Box::new(ScalarExpr::Constant(0.0)),
        },
        TensorScalarExpr::Reduction { .. } => {
            return Err(StructuredLoweringError::Axis(
                "nested reduction reached scalar lowering".into(),
            ));
        }
    })
}
