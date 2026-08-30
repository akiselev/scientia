//! FC4 indexed tensor/QFunction programs and realization-neutral operator factorization.
//!
//! The compiler differentiates a scalar typed integrand with respect to its test-function
//! evaluations.  The resulting dual quantities are point QFunction outputs; basis transpose and
//! scatter remain explicit realization stages rather than being hidden in a physics kernel.

use crate::formulation::{
    FormArgumentRole, FormCaptureRole, FormComplexConvention, VariationalForm,
};
use crate::id::{Digest, span_independent_digest};
use crate::requirements::{
    BasisEvaluationRequirement, DerivativeEvaluation, EssentialConstraintRequirement,
    EvaluationSite, FormRequirements, GeometryPreprocessingRequirement, InputSourceRequirement,
    QuadratureIntent, TraceMapping,
};
use crate::scientific::{BinaryOp, DerivativeContract, FieldRole, UnaryOp, ValueShape};
use crate::semantic::{
    ExprId, SemanticExpr, SemanticExprKind, SemanticMeasure, SemanticProvider, SymbolId, TraceSide,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const QFUNCTION_SCHEMA: &str = "scientia-qfunction/1";
pub const TENSOR_PROGRAM_SCHEMA: &str = "scientia-tensor-program/1";
pub const OPERATOR_FACTORIZATION_SCHEMA: &str = "scientia-operator-factorization/1";

macro_rules! tensor_id {
    ($name:ident) => {
        #[derive(
            Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub u32);

        impl $name {
            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

tensor_id!(TensorInputId);
tensor_id!(TensorAxisId);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorScalarSemantics {
    Real64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorAxisRole {
    Free,
    Reduction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorAxis {
    pub id: TensorAxisId,
    pub extent: usize,
    pub role: TensorAxisRole,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorSide {
    Cell,
    Exterior,
    Minus,
    Plus,
    Point,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TensorBinding {
    pub symbol: SymbolId,
    pub evaluation: BasisEvaluationRequirement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorInputRole {
    Active,
    Passive,
    External,
    Direction {
        primal: TensorInputId,
    },
    /// GX-A3: an externally-supplied tangent value (`d[property]/d[wrt]`, evaluated by the
    /// property's own Malleus tangent kernel) that a JVP multiplies against `wrt`'s own
    /// directional input to produce the chain-rule term `d(F)/d[property] * d[property]/d[wrt] *
    /// delta[wrt]`. JVP-only: never appears in a primal `TensorProgram`.
    PropertyTangent {
        property: TensorInputId,
        wrt: TensorInputId,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QFunctionInput {
    pub id: TensorInputId,
    pub binding: TensorBinding,
    pub side: TensorSide,
    pub shape: Vec<usize>,
    pub role: TensorInputRole,
    pub source: InputSourceRequirement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorProgramInputRole {
    Test,
    Active,
    Passive,
    External,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorProgramInput {
    pub id: TensorInputId,
    pub binding: TensorBinding,
    pub side: TensorSide,
    pub shape: Vec<usize>,
    pub role: TensorProgramInputRole,
    pub source: InputSourceRequirement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorUnaryOp {
    Neg,
    Abs,
    Sqrt,
    Exp,
    Ln,
    Sin,
    Cos,
    Tan,
    Floor,
    Ceil,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Min,
    Max,
    Atan2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorReductionOp {
    Sum,
}

/// One explicitly indexed scalar expression. Tensor values are represented by free axes and
/// contractions by lexically scoped reduction axes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TensorScalarExpr {
    Constant {
        value: f64,
    },
    Input {
        input: TensorInputId,
        indices: Vec<TensorAxisId>,
    },
    Unary {
        op: TensorUnaryOp,
        arg: Box<Self>,
    },
    Binary {
        op: TensorBinaryOp,
        lhs: Box<Self>,
        rhs: Box<Self>,
    },
    IndexEqual {
        lhs: TensorAxisId,
        rhs: TensorAxisId,
    },
    Reduction {
        op: TensorReductionOp,
        axis: TensorAxis,
        expression: Box<Self>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QFunctionOutputRole {
    TestDual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BasisAdjoint {
    Transpose,
    ConjugateTranspose,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QFunctionOutput {
    pub role: QFunctionOutputRole,
    pub binding: TensorBinding,
    pub side: TensorSide,
    pub shape: Vec<usize>,
    pub free_axes: Vec<TensorAxis>,
    pub expression: TensorScalarExpr,
    pub basis_adjoint: BasisAdjoint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QFunctionConstruction {
    TestDirectionalDerivative,
    SymbolicJvp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorProgramConstruction {
    TypedFormIntegrand,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorProgramReceipt {
    pub source_form_digest: Digest,
    pub source_requirements_digest: Digest,
    pub integral_index: usize,
    pub construction: TensorProgramConstruction,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndexedTensorExpression {
    pub shape: Vec<usize>,
    pub free_axes: Vec<TensorAxis>,
    pub expression: TensorScalarExpr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TensorProgram {
    pub schema: String,
    pub name: String,
    pub artifact_digest: Digest,
    pub scalar_semantics: TensorScalarSemantics,
    pub inputs: Vec<TensorProgramInput>,
    pub output: IndexedTensorExpression,
    pub receipt: TensorProgramReceipt,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QFunctionReceipt {
    pub source_form_digest: Digest,
    pub source_requirements_digest: Digest,
    pub source_tensor_program_digest: Digest,
    pub integral_index: usize,
    pub construction: QFunctionConstruction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivativeMode {
    Jvp,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivativeEvaluationPoint {
    RuntimeBindings,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivativeStateSemantics {
    Stateless,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivativeConstructionMethod {
    SymbolicDirectionalDifferentiation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivativeEvidence {
    IndexedAlgebraIdentity,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivativeReceipt {
    pub primal_artifact_digest: Digest,
    pub active_inputs: Vec<TensorBinding>,
    pub frozen_inputs: Vec<TensorBinding>,
    /// GX-A3: symbols of every `ModelDefinedProperty` input whose chain-rule tangent (contract
    /// C7's per-input tangent kernel) was wired into this JVP via a `PropertyTangent` input,
    /// rather than left as a purely frozen coefficient. `frozen_inputs` still lists these
    /// properties truthfully -- their own *value* is still externally supplied -- this field is
    /// what distinguishes "frozen but chain-rule-corrected" from "genuinely ignored".
    pub inlined_properties: Vec<SymbolId>,
    pub evaluation_point: DerivativeEvaluationPoint,
    pub mode: DerivativeMode,
    pub complex_convention: FormComplexConvention,
    pub state_semantics: DerivativeStateSemantics,
    pub construction: DerivativeConstructionMethod,
    pub evidence: Vec<DerivativeEvidence>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QFunctionProgram {
    pub schema: String,
    pub name: String,
    pub artifact_digest: Digest,
    pub scalar_semantics: TensorScalarSemantics,
    pub inputs: Vec<QFunctionInput>,
    pub outputs: Vec<QFunctionOutput>,
    pub receipt: QFunctionReceipt,
    pub derivative_receipt: Option<DerivativeReceipt>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RestrictionDirection {
    Gather,
    Scatter,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorStage {
    Restriction {
        symbol: SymbolId,
        direction: RestrictionDirection,
    },
    BasisForward {
        binding: TensorBinding,
    },
    ExternalPreprocessing {
        binding: TensorBinding,
        source: InputSourceRequirement,
    },
    Geometry {
        requirement: GeometryPreprocessingRequirement,
    },
    QFunction {
        primal: Digest,
        jvp: Digest,
    },
    QuadratureWeight,
    BasisAdjoint {
        binding: TensorBinding,
        action: BasisAdjoint,
    },
    EssentialConstraints {
        constraints: Vec<EssentialConstraintRequirement>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IntegralOperatorFactorization {
    pub integral_index: usize,
    pub measure: SemanticMeasure,
    pub quadrature: QuadratureIntent,
    pub stages: Vec<OperatorStage>,
    pub tensor_program: TensorProgram,
    pub primal: QFunctionProgram,
    pub jvp: QFunctionProgram,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorFactorizationMethod {
    TypedIndexedFc4,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorFactorizationReceipt {
    pub source_form_digest: Digest,
    pub source_requirements_digest: Digest,
    pub method: OperatorFactorizationMethod,
    pub complex_convention: FormComplexConvention,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperatorFactorization {
    pub schema: String,
    pub model: String,
    pub form: String,
    pub artifact_digest: Digest,
    pub integrals: Vec<IntegralOperatorFactorization>,
    pub essential_constraints: Vec<EssentialConstraintRequirement>,
    pub receipt: OperatorFactorizationReceipt,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TensorCompileError {
    #[error("TENSOR_SOURCE_MISMATCH: requirements do not originate from form `{form}`")]
    SourceMismatch { form: String },
    #[error("TENSOR_MISSING_SIGNATURE: integral {0} has no FC3 kernel signature")]
    MissingSignature(usize),
    #[error("TENSOR_INVALID_EXPRESSION: expression {0} is outside the form arena")]
    InvalidExpression(ExprId),
    #[error("TENSOR_MISSING_INPUT: no input matches symbol {symbol} evaluation {evaluation:?}")]
    MissingInput {
        symbol: SymbolId,
        evaluation: BasisEvaluationRequirement,
    },
    #[error("TENSOR_SHAPE: {0}")]
    Shape(String),
    #[error("TENSOR_TEST_LINEARITY: test input {0:?} remains after test differentiation")]
    NonlinearTest(TensorBinding),
    #[error(
        "TENSOR_ACTIVE_SOURCE: active symbol {symbol} must be basis-backed, got {input_source:?}"
    )]
    ActiveInputRequiresBasis {
        symbol: SymbolId,
        input_source: InputSourceRequirement,
    },
    #[error("TENSOR_UNSUPPORTED: {0}")]
    Unsupported(String),
}

#[derive(Clone)]
struct FormalInput {
    input: QFunctionInput,
    test: bool,
}

/// Compile FC3 form requirements and their parent typed form into FC4 indexed point programs and
/// an explicit realization-neutral operator decomposition.
pub fn factor_operator(
    form: &VariationalForm,
    requirements: &FormRequirements,
) -> Result<OperatorFactorization, TensorCompileError> {
    if requirements.receipt.source_form_digest != form.artifact_digest
        || requirements.model != form.model
        || requirements.form != form.name
    {
        return Err(TensorCompileError::SourceMismatch {
            form: form.name.clone(),
        });
    }

    let mut integrals = Vec::with_capacity(form.integrals.len());
    for integral_index in 0..form.integrals.len() {
        let group = requirements
            .integral_groups
            .iter()
            .find(|group| {
                group
                    .occurrences
                    .iter()
                    .any(|occurrence| occurrence.integral_index == integral_index)
            })
            .ok_or(TensorCompileError::MissingSignature(integral_index))?;
        let (tensor_program, primal) =
            compile_primal_qfunction(form, requirements, integral_index, group)?;
        let jvp =
            compile_jvp_qfunction(form, requirements, integral_index, &tensor_program, &primal)?;
        let mut stages = factor_stages(&primal, &jvp, &group.signature.geometry);
        if !requirements.essential_constraints.is_empty() {
            stages.push(OperatorStage::EssentialConstraints {
                constraints: requirements.essential_constraints.clone(),
            });
        }
        integrals.push(IntegralOperatorFactorization {
            integral_index,
            measure: form.integrals[integral_index].measure.clone(),
            quadrature: group.signature.quadrature.clone(),
            stages,
            tensor_program,
            primal,
            jvp,
        });
    }
    let receipt = OperatorFactorizationReceipt {
        source_form_digest: form.artifact_digest.clone(),
        source_requirements_digest: requirements.artifact_digest.clone(),
        method: OperatorFactorizationMethod::TypedIndexedFc4,
        complex_convention: form.receipt.complex_convention,
    };
    let artifact_digest = span_independent_digest(&OperatorDigestPayload {
        schema: OPERATOR_FACTORIZATION_SCHEMA,
        model: &form.model,
        form: &form.name,
        integrals: &integrals,
        essential_constraints: &requirements.essential_constraints,
        receipt: &receipt,
    });
    Ok(OperatorFactorization {
        schema: OPERATOR_FACTORIZATION_SCHEMA.into(),
        model: form.model.clone(),
        form: form.name.clone(),
        artifact_digest,
        integrals,
        essential_constraints: requirements.essential_constraints.clone(),
        receipt,
    })
}

#[derive(Serialize)]
struct OperatorDigestPayload<'a> {
    schema: &'static str,
    model: &'a str,
    form: &'a str,
    integrals: &'a [IntegralOperatorFactorization],
    essential_constraints: &'a [EssentialConstraintRequirement],
    receipt: &'a OperatorFactorizationReceipt,
}

fn factor_stages(
    primal: &QFunctionProgram,
    jvp: &QFunctionProgram,
    geometry: &[GeometryPreprocessingRequirement],
) -> Vec<OperatorStage> {
    let mut stages = Vec::new();
    let mut gathered = BTreeSet::new();
    for input in &primal.inputs {
        if input.source == InputSourceRequirement::Basis {
            if gathered.insert(input.binding.symbol) {
                stages.push(OperatorStage::Restriction {
                    symbol: input.binding.symbol,
                    direction: RestrictionDirection::Gather,
                });
            }
            stages.push(OperatorStage::BasisForward {
                binding: input.binding.clone(),
            });
        } else {
            stages.push(OperatorStage::ExternalPreprocessing {
                binding: input.binding.clone(),
                source: input.source,
            });
        }
    }
    for requirement in geometry {
        stages.push(OperatorStage::Geometry {
            requirement: *requirement,
        });
    }
    stages.push(OperatorStage::QFunction {
        primal: primal.artifact_digest.clone(),
        jvp: jvp.artifact_digest.clone(),
    });
    stages.push(OperatorStage::QuadratureWeight);
    for output in &primal.outputs {
        stages.push(OperatorStage::BasisAdjoint {
            binding: output.binding.clone(),
            action: output.basis_adjoint,
        });
    }
    for symbol in primal
        .outputs
        .iter()
        .map(|output| output.binding.symbol)
        .collect::<BTreeSet<_>>()
    {
        stages.push(OperatorStage::Restriction {
            symbol,
            direction: RestrictionDirection::Scatter,
        });
    }
    stages
}

fn compile_primal_qfunction(
    form: &VariationalForm,
    requirements: &FormRequirements,
    integral_index: usize,
    group: &crate::requirements::NormalizedIntegralGroup,
) -> Result<(TensorProgram, QFunctionProgram), TensorCompileError> {
    let mut formal = Vec::new();
    let integral = &form.integrals[integral_index];
    let mut direct_symbols = BTreeSet::new();
    collect_direct_symbols(&form.expressions, integral.integrand, &mut direct_symbols)?;
    for requirement in &group.signature.inputs {
        if requirement.source != InputSourceRequirement::Basis
            && !direct_symbols.contains(&requirement.symbol)
        {
            continue;
        }
        for evaluation in &requirement.evaluations {
            let binding = TensorBinding {
                symbol: requirement.symbol,
                evaluation: evaluation.clone(),
            };
            let test = form.arguments.iter().any(|argument| {
                argument.symbol == requirement.symbol && argument.role == FormArgumentRole::Test
            });
            let role = if test {
                TensorInputRole::Passive
            } else if is_active(form, requirement.symbol) {
                TensorInputRole::Active
            } else if requirement.source == InputSourceRequirement::Basis {
                TensorInputRole::Passive
            } else {
                TensorInputRole::External
            };
            if role == TensorInputRole::Active
                && requirement.source != InputSourceRequirement::Basis
            {
                return Err(TensorCompileError::ActiveInputRequiresBasis {
                    symbol: requirement.symbol,
                    input_source: requirement.source,
                });
            }
            let id = TensorInputId(formal.len() as u32);
            formal.push(FormalInput {
                input: QFunctionInput {
                    id,
                    side: side_for_site(evaluation.site),
                    shape: evaluation_shape(form, requirements, requirement.symbol, evaluation)?,
                    binding,
                    role,
                    source: requirement.source,
                },
                test,
            });
        }
    }
    let mut lowerer = Lowerer {
        form,
        inputs: &formal,
        next_axis: 0,
        default_site: site_for_measure(&integral.measure),
    };
    let integrand = lowerer.lower(integral.integrand, &[], EvalContext::default())?;
    let tensor_inputs = formal
        .iter()
        .map(|input| TensorProgramInput {
            id: input.input.id,
            binding: input.input.binding.clone(),
            side: input.input.side,
            shape: input.input.shape.clone(),
            role: if input.test {
                TensorProgramInputRole::Test
            } else {
                match input.input.role {
                    TensorInputRole::Active => TensorProgramInputRole::Active,
                    TensorInputRole::Passive => TensorProgramInputRole::Passive,
                    TensorInputRole::External => TensorProgramInputRole::External,
                    TensorInputRole::Direction { .. } => unreachable!("no directions in primal"),
                    TensorInputRole::PropertyTangent { .. } => {
                        unreachable!("no property tangents in primal")
                    }
                }
            },
            source: input.input.source,
        })
        .collect::<Vec<_>>();
    let tensor_receipt = TensorProgramReceipt {
        source_form_digest: form.artifact_digest.clone(),
        source_requirements_digest: requirements.artifact_digest.clone(),
        integral_index,
        construction: TensorProgramConstruction::TypedFormIntegrand,
    };
    let tensor_name = format!(
        "{}::{}::integral_{integral_index}::tensor",
        form.model, form.name
    );
    let tensor_program = make_tensor_program(
        tensor_name,
        tensor_inputs,
        IndexedTensorExpression {
            shape: Vec::new(),
            free_axes: Vec::new(),
            expression: integrand.clone(),
        },
        tensor_receipt,
    );

    let test_inputs = formal
        .iter()
        .filter(|input| input.test)
        .cloned()
        .collect::<Vec<_>>();
    if test_inputs.is_empty() {
        return Err(TensorCompileError::Unsupported(
            "an operator integral must contain a test-function evaluation".into(),
        ));
    }

    let mut outputs = Vec::new();
    for test in &test_inputs {
        let free_axes = test
            .input
            .shape
            .iter()
            .map(|extent| lowerer.axis(*extent, TensorAxisRole::Free))
            .collect::<Vec<_>>();
        let indices = free_axes.iter().map(|axis| axis.id).collect::<Vec<_>>();
        let expression = simplify(differentiate(&integrand, test.input.id, &indices)?);
        outputs.push(QFunctionOutput {
            role: QFunctionOutputRole::TestDual,
            binding: test.input.binding.clone(),
            side: test.input.side,
            shape: test.input.shape.clone(),
            free_axes,
            expression,
            basis_adjoint: BasisAdjoint::Transpose,
        });
    }

    let retained = formal
        .iter()
        .filter(|input| !input.test)
        .cloned()
        .collect::<Vec<_>>();
    let remap = retained
        .iter()
        .enumerate()
        .map(|(index, input)| (input.input.id, TensorInputId(index as u32)))
        .collect::<BTreeMap<_, _>>();
    let inputs = retained
        .into_iter()
        .enumerate()
        .map(|(index, mut input)| {
            input.input.id = TensorInputId(index as u32);
            input.input
        })
        .collect::<Vec<_>>();
    for output in &mut outputs {
        output.expression = remap_inputs(&output.expression, &remap).map_err(|old| {
            let binding = formal[old.index()].input.binding.clone();
            TensorCompileError::NonlinearTest(binding)
        })?;
    }
    let receipt = QFunctionReceipt {
        source_form_digest: form.artifact_digest.clone(),
        source_requirements_digest: requirements.artifact_digest.clone(),
        source_tensor_program_digest: tensor_program.artifact_digest.clone(),
        integral_index,
        construction: QFunctionConstruction::TestDirectionalDerivative,
    };
    let name = format!(
        "{}::{}::integral_{integral_index}::primal",
        form.model, form.name
    );
    let artifact_digest = qfunction_digest(&name, &inputs, &outputs, &receipt, None);
    let qfunction = QFunctionProgram {
        schema: QFUNCTION_SCHEMA.into(),
        name,
        artifact_digest,
        scalar_semantics: TensorScalarSemantics::Real64,
        inputs,
        outputs,
        receipt,
        derivative_receipt: None,
    };
    Ok((tensor_program, qfunction))
}

fn compile_jvp_qfunction(
    form: &VariationalForm,
    requirements: &FormRequirements,
    integral_index: usize,
    tensor_program: &TensorProgram,
    primal: &QFunctionProgram,
) -> Result<QFunctionProgram, TensorCompileError> {
    let mut inputs = primal.inputs.clone();
    let mut directions = BTreeMap::new();
    for input in &primal.inputs {
        if input.role != TensorInputRole::Active {
            continue;
        }
        let id = TensorInputId(inputs.len() as u32);
        directions.insert(input.id, id);
        let mut direction = input.clone();
        direction.id = id;
        direction.role = TensorInputRole::Direction { primal: input.id };
        inputs.push(direction);
    }

    // GX-A3: a `ModelDefinedProperty` input whose definition wraps only `Symbolic`/`Automatic`
    // provider calls is not expanded into the integrand (a provider has no closed form Scientia
    // can lower symbolically); instead each Active, Basis-`Value` input its definition depends
    // on gets a companion `PropertyTangent` input -- the property's own runtime-evaluated
    // tangent (contract C7's per-input tangent kernel) -- and the chain-rule term
    // `d(F)/d[property] * d[property]/d[wrt] * delta[wrt]` is added to every output alongside
    // the ordinary directional derivative. A property with no provider calls at all (pure
    // basis-field/parameter arithmetic) is not inlined by this package; it stays a frozen
    // (Picard) coefficient exactly as before GX-A3.
    let mut property_corrections: Vec<(&QFunctionInput, Vec<(TensorInputId, TensorInputId)>)> =
        Vec::new();
    let mut inlined_properties: Vec<SymbolId> = Vec::new();
    for input in &primal.inputs {
        let InputSourceRequirement::ModelDefinedProperty { definition } = input.source else {
            continue;
        };
        if input.role == TensorInputRole::Active || !input.shape.is_empty() {
            continue;
        }
        if !property_providers_all_differentiable(form, definition) {
            continue;
        }
        let mut direct_symbols = BTreeSet::new();
        collect_direct_symbols(&form.expressions, definition, &mut direct_symbols)?;
        let mut wrt_pairs = Vec::new();
        for candidate in primal.inputs.iter().filter(|candidate| {
            candidate.role == TensorInputRole::Active
                && candidate.binding.evaluation.derivative == DerivativeEvaluation::Value
                && direct_symbols.contains(&candidate.binding.symbol)
        }) {
            let tangent_id = TensorInputId(inputs.len() as u32);
            inputs.push(QFunctionInput {
                id: tangent_id,
                binding: input.binding.clone(),
                side: input.side,
                shape: input.shape.clone(),
                role: TensorInputRole::PropertyTangent {
                    property: input.id,
                    wrt: candidate.id,
                },
                source: input.source,
            });
            wrt_pairs.push((candidate.id, tangent_id));
        }
        if !wrt_pairs.is_empty() {
            property_corrections.push((input, wrt_pairs));
            inlined_properties.push(input.binding.symbol);
        }
    }

    let outputs = primal
        .outputs
        .iter()
        .map(|output| {
            let mut output = output.clone();
            let mut expression = directional_derivative(&output.expression, &directions)?;
            for (property, wrt_pairs) in &property_corrections {
                let partial = differentiate(&output.expression, property.id, &[])?;
                if is_zero(&partial) {
                    continue;
                }
                let mut chain = constant(0.0);
                for (wrt, tangent_id) in wrt_pairs {
                    let Some(direction_id) = directions.get(wrt) else {
                        continue;
                    };
                    chain = binary(
                        TensorBinaryOp::Add,
                        chain,
                        binary(
                            TensorBinaryOp::Mul,
                            TensorScalarExpr::Input {
                                input: *tangent_id,
                                indices: vec![],
                            },
                            TensorScalarExpr::Input {
                                input: *direction_id,
                                indices: vec![],
                            },
                        ),
                    );
                }
                expression = binary(
                    TensorBinaryOp::Add,
                    expression,
                    binary(TensorBinaryOp::Mul, partial, chain),
                );
            }
            output.expression = simplify(expression);
            Ok(output)
        })
        .collect::<Result<Vec<_>, TensorCompileError>>()?;
    let active_inputs = primal
        .inputs
        .iter()
        .filter(|input| input.role == TensorInputRole::Active)
        .map(|input| input.binding.clone())
        .collect();
    let frozen_inputs = primal
        .inputs
        .iter()
        .filter(|input| input.role != TensorInputRole::Active)
        .map(|input| input.binding.clone())
        .collect();
    let derivative_receipt = DerivativeReceipt {
        primal_artifact_digest: primal.artifact_digest.clone(),
        active_inputs,
        frozen_inputs,
        inlined_properties,
        evaluation_point: DerivativeEvaluationPoint::RuntimeBindings,
        mode: DerivativeMode::Jvp,
        complex_convention: form.receipt.complex_convention,
        state_semantics: DerivativeStateSemantics::Stateless,
        construction: DerivativeConstructionMethod::SymbolicDirectionalDifferentiation,
        evidence: vec![DerivativeEvidence::IndexedAlgebraIdentity],
    };
    let receipt = QFunctionReceipt {
        source_form_digest: form.artifact_digest.clone(),
        source_requirements_digest: requirements.artifact_digest.clone(),
        source_tensor_program_digest: tensor_program.artifact_digest.clone(),
        integral_index,
        construction: QFunctionConstruction::SymbolicJvp,
    };
    let name = format!(
        "{}::{}::integral_{integral_index}::jvp",
        form.model, form.name
    );
    let artifact_digest = qfunction_digest(
        &name,
        &inputs,
        &outputs,
        &receipt,
        Some(&derivative_receipt),
    );
    Ok(QFunctionProgram {
        schema: QFUNCTION_SCHEMA.into(),
        name,
        artifact_digest,
        scalar_semantics: TensorScalarSemantics::Real64,
        inputs,
        outputs,
        receipt,
        derivative_receipt: Some(derivative_receipt),
    })
}

fn make_tensor_program(
    name: String,
    inputs: Vec<TensorProgramInput>,
    output: IndexedTensorExpression,
    receipt: TensorProgramReceipt,
) -> TensorProgram {
    #[derive(Serialize)]
    struct Payload<'a> {
        schema: &'static str,
        name: &'a str,
        scalar_semantics: TensorScalarSemantics,
        inputs: &'a [TensorProgramInput],
        output: &'a IndexedTensorExpression,
        receipt: &'a TensorProgramReceipt,
    }
    let artifact_digest = span_independent_digest(&Payload {
        schema: TENSOR_PROGRAM_SCHEMA,
        name: &name,
        scalar_semantics: TensorScalarSemantics::Real64,
        inputs: &inputs,
        output: &output,
        receipt: &receipt,
    });
    TensorProgram {
        schema: TENSOR_PROGRAM_SCHEMA.into(),
        name,
        artifact_digest,
        scalar_semantics: TensorScalarSemantics::Real64,
        inputs,
        output,
        receipt,
    }
}

fn qfunction_digest(
    name: &str,
    inputs: &[QFunctionInput],
    outputs: &[QFunctionOutput],
    receipt: &QFunctionReceipt,
    derivative_receipt: Option<&DerivativeReceipt>,
) -> Digest {
    #[derive(Serialize)]
    struct Payload<'a> {
        schema: &'static str,
        name: &'a str,
        scalar_semantics: TensorScalarSemantics,
        inputs: &'a [QFunctionInput],
        outputs: &'a [QFunctionOutput],
        receipt: &'a QFunctionReceipt,
        derivative_receipt: Option<&'a DerivativeReceipt>,
    }
    span_independent_digest(&Payload {
        schema: QFUNCTION_SCHEMA,
        name,
        scalar_semantics: TensorScalarSemantics::Real64,
        inputs,
        outputs,
        receipt,
        derivative_receipt,
    })
}

fn is_active(form: &VariationalForm, symbol: SymbolId) -> bool {
    form.arguments
        .iter()
        .any(|argument| argument.symbol == symbol && argument.role == FormArgumentRole::Trial)
        || form.captures.iter().any(|capture| {
            capture.symbol == symbol
                && matches!(
                    capture.role,
                    FormCaptureRole::PhysicalField(
                        FieldRole::Unknown | FieldRole::State | FieldRole::Trial
                    )
                )
        })
}

fn evaluation_shape(
    form: &VariationalForm,
    requirements: &FormRequirements,
    symbol: SymbolId,
    evaluation: &BasisEvaluationRequirement,
) -> Result<Vec<usize>, TensorCompileError> {
    let base = requirements
        .spaces
        .iter()
        .find(|space| space.symbol == symbol)
        .map(|space| value_shape(&space.value_shape))
        .or_else(|| {
            form.expressions.iter().find_map(|expression| {
                matches!(expression.kind, SemanticExprKind::Symbol { symbol: candidate } if candidate == symbol)
                    .then(|| semantic_value_shape(&expression.ty.shape))
                    .flatten()
                    .filter(|shape| {
                        !shape.is_empty()
                            || matches!(
                                expression.ty.shape,
                                crate::semantic::SemanticShape::Numeric(ValueShape::Scalar)
                            )
                    })
            })
        })
        .or_else(|| {
            form.arguments
                .iter()
                .find(|argument| argument.symbol == symbol)
                .and_then(|argument| semantic_value_shape(&argument.ty.shape))
        })
        .or_else(|| {
            form.captures
                .iter()
                .find(|capture| capture.symbol == symbol)
                .and_then(|capture| semantic_value_shape(&capture.ty.shape))
        })
        .ok_or_else(|| {
            TensorCompileError::Shape(format!("symbol {symbol} has no declared value shape"))
        })?;
    if matches!(
        evaluation.derivative,
        DerivativeEvaluation::Value | DerivativeEvaluation::TimeDerivative
    ) {
        return trace_shape(base, evaluation.trace_mapping);
    }
    let dimension = requirements
        .elements
        .iter()
        .find(|element| element.symbol == symbol)
        .or_else(|| requirements.elements.first())
        .map(|element| element.topological_dimension as usize)
        .ok_or_else(|| {
            TensorCompileError::Shape(format!("symbol {symbol} has no domain extent"))
        })?;
    let evaluated = match evaluation.derivative {
        DerivativeEvaluation::Value | DerivativeEvaluation::TimeDerivative => unreachable!(),
        DerivativeEvaluation::Gradient => {
            let mut shape = base;
            shape.push(dimension);
            shape
        }
        DerivativeEvaluation::Divergence => {
            require_domain_vector(symbol, evaluation.derivative, &base, dimension)?;
            Vec::new()
        }
        DerivativeEvaluation::Curl => {
            require_domain_vector(symbol, evaluation.derivative, &base, dimension)?;
            if dimension == 2 {
                Vec::new()
            } else {
                vec![dimension]
            }
        }
        DerivativeEvaluation::RotatedGradient => {
            require_scalar(symbol, evaluation.derivative, &base)?;
            vec![dimension]
        }
        DerivativeEvaluation::SymmetricGradient => {
            require_domain_vector(symbol, evaluation.derivative, &base, dimension)?;
            vec![dimension, dimension]
        }
    };
    trace_shape(evaluated, evaluation.trace_mapping)
}

fn trace_shape(
    shape: Vec<usize>,
    mapping: Option<TraceMapping>,
) -> Result<Vec<usize>, TensorCompileError> {
    match (mapping, shape.as_slice()) {
        (Some(TraceMapping::Normal), [_]) => Ok(Vec::new()),
        (Some(TraceMapping::Normal), [rows, _]) => Ok(vec![*rows]),
        (Some(TraceMapping::Normal), _) => Err(TensorCompileError::Shape(
            "normal trace requires a vector or rank-two tensor".into(),
        )),
        (Some(TraceMapping::Value | TraceMapping::Tangential) | None, _) => Ok(shape),
    }
}

/// GX-A3: true when every `ProviderCall` reachable from `id` (walking through arithmetic,
/// function calls, and further provider calls) names a declared provider whose
/// `differentiability` is `Symbolic` or `Automatic` -- i.e. one whose own tangent kernel
/// (contract C7) can supply the chain-rule term this package wires into the JVP. An expression
/// with no provider calls at all (pure basis-field/parameter arithmetic) is vacuously `true`
/// here, but is handled by `compile_jvp_qfunction`'s own `wrt_pairs`-emptiness check, not by
/// this predicate, since such a property is not inlined by this package (see the module-level
/// GX-A3 note in `compile_jvp_qfunction`).
fn property_providers_all_differentiable(form: &VariationalForm, id: ExprId) -> bool {
    fn walk(expressions: &[SemanticExpr], id: ExprId, providers: &[SemanticProvider]) -> bool {
        let Some(expression) = expressions.get(id.index()) else {
            return false;
        };
        match &expression.kind {
            SemanticExprKind::ProviderCall { provider, args } => {
                let differentiable = providers
                    .iter()
                    .find(|candidate| candidate.id == *provider)
                    .is_some_and(|candidate| {
                        matches!(
                            candidate.differentiability,
                            DerivativeContract::Symbolic | DerivativeContract::Automatic
                        )
                    });
                differentiable && args.iter().all(|arg| walk(expressions, *arg, providers))
            }
            SemanticExprKind::Unary { arg, .. } | SemanticExprKind::Differential { arg, .. } => {
                walk(expressions, *arg, providers)
            }
            SemanticExprKind::Binary { lhs, rhs, .. } => {
                walk(expressions, *lhs, providers) && walk(expressions, *rhs, providers)
            }
            SemanticExprKind::Call { args, .. } | SemanticExprKind::Vector { elements: args } => {
                args.iter().all(|arg| walk(expressions, *arg, providers))
            }
            SemanticExprKind::Symbol { .. }
            | SemanticExprKind::Number { .. }
            | SemanticExprKind::String { .. } => true,
            // Any other construct (contraction, tensor/facet trace, jump, average, index, ...)
            // is outside what this package expands; refuse inlining rather than guess.
            _ => false,
        }
    }
    walk(&form.expressions, id, &form.providers)
}

fn collect_direct_symbols(
    expressions: &[crate::semantic::SemanticExpr],
    id: ExprId,
    symbols: &mut BTreeSet<SymbolId>,
) -> Result<(), TensorCompileError> {
    let expression = expressions
        .get(id.index())
        .ok_or(TensorCompileError::InvalidExpression(id))?;
    match &expression.kind {
        SemanticExprKind::Symbol { symbol } => {
            symbols.insert(*symbol);
        }
        SemanticExprKind::Unary { arg, .. }
        | SemanticExprKind::Differential { arg, .. }
        | SemanticExprKind::TensorTrace { value: arg, .. }
        | SemanticExprKind::FacetTrace { value: arg, .. }
        | SemanticExprKind::Jump { value: arg }
        | SemanticExprKind::Average { value: arg }
        | SemanticExprKind::Conjugate { value: arg }
        | SemanticExprKind::NormalComponent { value: arg, .. } => {
            collect_direct_symbols(expressions, *arg, symbols)?;
        }
        SemanticExprKind::Binary { lhs, rhs, .. }
        | SemanticExprKind::Contraction { lhs, rhs, .. } => {
            collect_direct_symbols(expressions, *lhs, symbols)?;
            collect_direct_symbols(expressions, *rhs, symbols)?;
        }
        SemanticExprKind::Call { args, .. }
        | SemanticExprKind::ProviderCall { args, .. }
        | SemanticExprKind::Vector { elements: args } => {
            for arg in args {
                collect_direct_symbols(expressions, *arg, symbols)?;
            }
        }
        SemanticExprKind::Index { value, indices } => {
            collect_direct_symbols(expressions, *value, symbols)?;
            for index in indices {
                collect_direct_symbols(expressions, *index, symbols)?;
            }
        }
        SemanticExprKind::Number { .. } | SemanticExprKind::String { .. } => {}
    }
    Ok(())
}

fn require_domain_vector(
    symbol: SymbolId,
    derivative: DerivativeEvaluation,
    shape: &[usize],
    dimension: usize,
) -> Result<(), TensorCompileError> {
    if shape == [dimension] {
        Ok(())
    } else {
        Err(TensorCompileError::Shape(format!(
            "{derivative:?} of symbol {symbol} requires shape [{dimension}], got {shape:?}"
        )))
    }
}

fn require_scalar(
    symbol: SymbolId,
    derivative: DerivativeEvaluation,
    shape: &[usize],
) -> Result<(), TensorCompileError> {
    if shape.is_empty() {
        Ok(())
    } else {
        Err(TensorCompileError::Shape(format!(
            "{derivative:?} of symbol {symbol} requires scalar shape, got {shape:?}"
        )))
    }
}

fn semantic_value_shape(shape: &crate::semantic::SemanticShape) -> Option<Vec<usize>> {
    match shape {
        crate::semantic::SemanticShape::Numeric(shape) => Some(value_shape(shape)),
        crate::semantic::SemanticShape::Deferred => Some(Vec::new()),
        _ => None,
    }
}

fn value_shape(shape: &ValueShape) -> Vec<usize> {
    match shape {
        ValueShape::Scalar => Vec::new(),
        ValueShape::Vector(extent) => vec![*extent as usize],
        ValueShape::Tensor { rows, cols } => vec![*rows as usize, *cols as usize],
        ValueShape::SymmetricTensor(extent) => vec![*extent as usize, *extent as usize],
    }
}

#[derive(Clone, Copy)]
struct EvalContext {
    derivative: DerivativeEvaluation,
    site: Option<EvaluationSite>,
    trace_mapping: Option<TraceMapping>,
}

impl Default for EvalContext {
    fn default() -> Self {
        Self {
            derivative: DerivativeEvaluation::Value,
            site: None,
            trace_mapping: None,
        }
    }
}

impl EvalContext {
    fn binding(self, default_site: EvaluationSite) -> BasisEvaluationRequirement {
        BasisEvaluationRequirement {
            derivative: self.derivative,
            site: self.site.unwrap_or(default_site),
            trace_mapping: self.trace_mapping,
        }
    }
}

struct Lowerer<'a> {
    form: &'a VariationalForm,
    inputs: &'a [FormalInput],
    next_axis: u32,
    default_site: EvaluationSite,
}

impl Lowerer<'_> {
    fn axis(&mut self, extent: usize, role: TensorAxisRole) -> TensorAxis {
        let axis = TensorAxis {
            id: TensorAxisId(self.next_axis),
            extent,
            role,
        };
        self.next_axis += 1;
        axis
    }

    fn expression(&self, id: ExprId) -> Result<&crate::semantic::SemanticExpr, TensorCompileError> {
        self.form
            .expressions
            .get(id.index())
            .ok_or(TensorCompileError::InvalidExpression(id))
    }

    fn shape(&self, id: ExprId, context: EvalContext) -> Result<Vec<usize>, TensorCompileError> {
        let expression = self.expression(id)?;
        match &expression.kind {
            SemanticExprKind::Symbol { symbol } => {
                let evaluation = context.binding(self.default_site);
                return self
                    .inputs
                    .iter()
                    .find(|input| input_matches(input, *symbol, &evaluation))
                    .map(|input| input.input.shape.clone())
                    .ok_or(TensorCompileError::MissingInput {
                        symbol: *symbol,
                        evaluation,
                    });
            }
            SemanticExprKind::Differential { operator, arg } => {
                if !matches!(self.expression(*arg)?.kind, SemanticExprKind::Symbol { .. }) {
                    return Err(TensorCompileError::Unsupported(
                        "differential of a non-symbol expression requires preprocessing expansion"
                            .into(),
                    ));
                }
                return self.shape(
                    *arg,
                    EvalContext {
                        derivative: derivative_evaluation(*operator),
                        ..context
                    },
                );
            }
            SemanticExprKind::FacetTrace { value, side } => {
                return self.shape(
                    *value,
                    EvalContext {
                        site: Some(site_for_trace(*side)),
                        ..context
                    },
                );
            }
            SemanticExprKind::Jump { value } | SemanticExprKind::Average { value } => {
                return self.shape(
                    *value,
                    EvalContext {
                        site: Some(EvaluationSite::MinusTrace),
                        ..context
                    },
                );
            }
            SemanticExprKind::NormalComponent { value, side } => {
                return self.shape(
                    *value,
                    EvalContext {
                        site: Some(site_for_trace(*side)),
                        trace_mapping: Some(TraceMapping::Normal),
                        ..context
                    },
                );
            }
            _ => {}
        }
        if let Some(shape) = semantic_value_shape(&expression.ty.shape)
            && (!shape.is_empty()
                || matches!(
                    expression.ty.shape,
                    crate::semantic::SemanticShape::Numeric(ValueShape::Scalar)
                ))
        {
            return Ok(shape);
        }
        Ok(match &expression.kind {
            SemanticExprKind::Number { .. } | SemanticExprKind::String { .. } => Vec::new(),
            SemanticExprKind::Unary { arg, .. } | SemanticExprKind::Conjugate { value: arg } => {
                self.shape(*arg, context)?
            }
            SemanticExprKind::Binary { lhs, rhs, .. } => {
                let lhs = self.shape(*lhs, context)?;
                let rhs = self.shape(*rhs, context)?;
                broadcast_shape(&lhs, &rhs)?
            }
            SemanticExprKind::Contraction { .. } | SemanticExprKind::TensorTrace { .. } => {
                expression
                    .ty
                    .axes
                    .iter()
                    .map(|axis| axis.extent as usize)
                    .collect()
            }
            SemanticExprKind::Call { args, .. } | SemanticExprKind::ProviderCall { args, .. } => {
                args.iter()
                    .map(|arg| self.shape(*arg, context))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .find(|shape| !shape.is_empty())
                    .unwrap_or_default()
            }
            SemanticExprKind::Index { .. } => Vec::new(),
            SemanticExprKind::Vector { elements } => vec![elements.len()],
            SemanticExprKind::Symbol { .. }
            | SemanticExprKind::Differential { .. }
            | SemanticExprKind::FacetTrace { .. }
            | SemanticExprKind::Jump { .. }
            | SemanticExprKind::Average { .. }
            | SemanticExprKind::NormalComponent { .. } => unreachable!("handled above"),
        })
    }

    fn lower(
        &mut self,
        id: ExprId,
        indices: &[TensorAxisId],
        context: EvalContext,
    ) -> Result<TensorScalarExpr, TensorCompileError> {
        let expression = self.expression(id)?.clone();
        match expression.kind {
            SemanticExprKind::Number {
                value, unit: None, ..
            } => Ok(constant(value)),
            SemanticExprKind::Number { unit: Some(_), .. } => Err(TensorCompileError::Unsupported(
                "unit-bearing literal before numeric canonicalization".into(),
            )),
            SemanticExprKind::Symbol { symbol } => {
                let evaluation = context.binding(self.default_site);
                let input = self
                    .inputs
                    .iter()
                    .find(|input| input_matches(input, symbol, &evaluation))
                    .ok_or_else(|| TensorCompileError::MissingInput {
                        symbol,
                        evaluation: evaluation.clone(),
                    })?;
                if input.input.shape.len() != indices.len() {
                    return Err(TensorCompileError::Shape(format!(
                        "symbol {symbol} expects rank {}, got {} indices",
                        input.input.shape.len(),
                        indices.len()
                    )));
                }
                Ok(TensorScalarExpr::Input {
                    input: input.input.id,
                    indices: indices.to_vec(),
                })
            }
            SemanticExprKind::Unary {
                op: UnaryOp::Neg,
                arg,
            } => Ok(unary(
                TensorUnaryOp::Neg,
                self.lower(arg, indices, context)?,
            )),
            SemanticExprKind::Binary { op, lhs, rhs } => {
                let lhs_shape = self.shape(lhs, context)?;
                let rhs_shape = self.shape(rhs, context)?;
                let output_shape = broadcast_shape(&lhs_shape, &rhs_shape)?;
                if output_shape.len() != indices.len() {
                    return Err(TensorCompileError::Shape(format!(
                        "binary expression {id} expects rank {}, got {} indices",
                        output_shape.len(),
                        indices.len()
                    )));
                }
                let lhs_indices = if lhs_shape.is_empty() { &[] } else { indices };
                let rhs_indices = if rhs_shape.is_empty() { &[] } else { indices };
                let op = match op {
                    BinaryOp::Add => TensorBinaryOp::Add,
                    BinaryOp::Sub => TensorBinaryOp::Sub,
                    BinaryOp::Mul => TensorBinaryOp::Mul,
                    BinaryOp::Div => TensorBinaryOp::Div,
                    BinaryOp::Pow => TensorBinaryOp::Pow,
                    BinaryOp::Eq | BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                        return Err(TensorCompileError::Unsupported(
                            "comparison in a QFunction integrand".into(),
                        ));
                    }
                };
                Ok(binary(
                    op,
                    self.lower(lhs, lhs_indices, context)?,
                    self.lower(rhs, rhs_indices, context)?,
                ))
            }
            SemanticExprKind::Differential { operator, arg } => {
                if !matches!(self.expression(arg)?.kind, SemanticExprKind::Symbol { .. }) {
                    return Err(TensorCompileError::Unsupported(
                        "differential of a non-symbol expression requires preprocessing expansion"
                            .into(),
                    ));
                }
                self.lower(
                    arg,
                    indices,
                    EvalContext {
                        derivative: derivative_evaluation(operator),
                        ..context
                    },
                )
            }
            SemanticExprKind::Contraction {
                lhs,
                rhs,
                axes,
                conjugate_lhs,
            } => {
                if conjugate_lhs {
                    return Err(TensorCompileError::Unsupported(
                        "complex contraction under real64 scalar semantics".into(),
                    ));
                }
                let lhs_shape = self.shape(lhs, context)?;
                let rhs_shape = self.shape(rhs, context)?;
                let contracted_lhs = axes
                    .iter()
                    .map(|axis| axis.lhs as usize)
                    .collect::<BTreeSet<_>>();
                let contracted_rhs = axes
                    .iter()
                    .map(|axis| axis.rhs as usize)
                    .collect::<BTreeSet<_>>();
                let contracted_axis_count = axes.len();
                let expected_rank = lhs_shape
                    .len()
                    .checked_add(rhs_shape.len())
                    .and_then(|combined_rank| combined_rank.checked_sub(2 * contracted_axis_count))
                    .ok_or_else(|| {
                        TensorCompileError::Shape(format!(
                            "contraction {id} contracts {contracted_axis_count} axis pairs, \
                             which exceeds the combined operand rank {} + {}",
                            lhs_shape.len(),
                            rhs_shape.len()
                        ))
                    })?;
                if expected_rank != indices.len() {
                    return Err(TensorCompileError::Shape(format!(
                        "contraction {id} expects rank {expected_rank}, got {} indices",
                        indices.len()
                    )));
                }
                let mut lhs_indices = vec![TensorAxisId(0); lhs_shape.len()];
                let mut rhs_indices = vec![TensorAxisId(0); rhs_shape.len()];
                let mut free = indices.iter().copied();
                for (position, slot) in lhs_indices.iter_mut().enumerate() {
                    if !contracted_lhs.contains(&position) {
                        *slot = free.next().expect("validated free rank");
                    }
                }
                for (position, slot) in rhs_indices.iter_mut().enumerate() {
                    if !contracted_rhs.contains(&position) {
                        *slot = free.next().expect("validated free rank");
                    }
                }
                let mut reductions = Vec::new();
                for pair in axes {
                    let lhs_axis = pair.lhs as usize;
                    let rhs_axis = pair.rhs as usize;
                    let lhs_extent = *lhs_shape.get(lhs_axis).ok_or_else(|| {
                        TensorCompileError::Shape(format!(
                            "invalid lhs contraction axis {lhs_axis}"
                        ))
                    })?;
                    let rhs_extent = *rhs_shape.get(rhs_axis).ok_or_else(|| {
                        TensorCompileError::Shape(format!(
                            "invalid rhs contraction axis {rhs_axis}"
                        ))
                    })?;
                    if lhs_extent != rhs_extent {
                        return Err(TensorCompileError::Shape(format!(
                            "contraction extents differ: {lhs_extent} and {rhs_extent}"
                        )));
                    }
                    let axis = self.axis(lhs_extent, TensorAxisRole::Reduction);
                    lhs_indices[lhs_axis] = axis.id;
                    rhs_indices[rhs_axis] = axis.id;
                    reductions.push(axis);
                }
                let mut result = binary(
                    TensorBinaryOp::Mul,
                    self.lower(lhs, &lhs_indices, context)?,
                    self.lower(rhs, &rhs_indices, context)?,
                );
                for axis in reductions.into_iter().rev() {
                    result = TensorScalarExpr::Reduction {
                        op: TensorReductionOp::Sum,
                        axis,
                        expression: Box::new(result),
                    };
                }
                Ok(result)
            }
            SemanticExprKind::FacetTrace { value, side } => self.lower(
                value,
                indices,
                EvalContext {
                    site: Some(site_for_trace(side)),
                    ..context
                },
            ),
            SemanticExprKind::Jump { value } => Ok(binary(
                TensorBinaryOp::Sub,
                self.lower(
                    value,
                    indices,
                    EvalContext {
                        site: Some(EvaluationSite::MinusTrace),
                        ..context
                    },
                )?,
                self.lower(
                    value,
                    indices,
                    EvalContext {
                        site: Some(EvaluationSite::PlusTrace),
                        ..context
                    },
                )?,
            )),
            SemanticExprKind::Average { value } => Ok(binary(
                TensorBinaryOp::Mul,
                constant(0.5),
                binary(
                    TensorBinaryOp::Add,
                    self.lower(
                        value,
                        indices,
                        EvalContext {
                            site: Some(EvaluationSite::MinusTrace),
                            ..context
                        },
                    )?,
                    self.lower(
                        value,
                        indices,
                        EvalContext {
                            site: Some(EvaluationSite::PlusTrace),
                            ..context
                        },
                    )?,
                ),
            )),
            SemanticExprKind::NormalComponent { value, side } => self.lower(
                value,
                indices,
                EvalContext {
                    site: Some(site_for_trace(side)),
                    trace_mapping: Some(TraceMapping::Normal),
                    ..context
                },
            ),
            SemanticExprKind::Call { function, args } => {
                self.lower_call(&function, &args, indices, context)
            }
            // A provider call directly inside an integrand needs property-kernel lowering
            // (GX-A2/A3, not implemented here); it still refuses cleanly rather than
            // attempting kernel lowering.
            SemanticExprKind::ProviderCall { .. }
            | SemanticExprKind::TensorTrace { .. }
            | SemanticExprKind::Conjugate { .. }
            | SemanticExprKind::Index { .. }
            | SemanticExprKind::Vector { .. }
            | SemanticExprKind::String { .. } => Err(TensorCompileError::Unsupported(format!(
                "semantic expression {id} requires a later tensor primitive"
            ))),
        }
    }

    fn lower_call(
        &mut self,
        function: &str,
        args: &[ExprId],
        indices: &[TensorAxisId],
        context: EvalContext,
    ) -> Result<TensorScalarExpr, TensorCompileError> {
        let unary_op = match function {
            "abs" => Some(TensorUnaryOp::Abs),
            "sqrt" => Some(TensorUnaryOp::Sqrt),
            "exp" => Some(TensorUnaryOp::Exp),
            "log" | "ln" => Some(TensorUnaryOp::Ln),
            "sin" => Some(TensorUnaryOp::Sin),
            "cos" => Some(TensorUnaryOp::Cos),
            "tan" => Some(TensorUnaryOp::Tan),
            "floor" => Some(TensorUnaryOp::Floor),
            "ceil" => Some(TensorUnaryOp::Ceil),
            _ => None,
        };
        if let (Some(op), [arg]) = (unary_op, args) {
            return Ok(unary(op, self.lower(*arg, indices, context)?));
        }
        let binary_op = match function {
            "min" => Some(TensorBinaryOp::Min),
            "max" => Some(TensorBinaryOp::Max),
            "atan2" => Some(TensorBinaryOp::Atan2),
            _ => None,
        };
        if let (Some(op), [lhs, rhs]) = (binary_op, args) {
            return Ok(binary(
                op,
                self.lower(*lhs, indices, context)?,
                self.lower(*rhs, indices, context)?,
            ));
        }
        Err(TensorCompileError::Unsupported(format!(
            "call to `{function}` in indexed QFunction"
        )))
    }
}

fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, TensorCompileError> {
    if lhs.is_empty() {
        Ok(rhs.to_vec())
    } else if rhs.is_empty() || lhs == rhs {
        Ok(lhs.to_vec())
    } else {
        Err(TensorCompileError::Shape(format!(
            "cannot broadcast shapes {lhs:?} and {rhs:?}"
        )))
    }
}

fn derivative_evaluation(operator: crate::semantic::DifferentialOperator) -> DerivativeEvaluation {
    match operator {
        crate::semantic::DifferentialOperator::Gradient => DerivativeEvaluation::Gradient,
        crate::semantic::DifferentialOperator::Divergence => DerivativeEvaluation::Divergence,
        crate::semantic::DifferentialOperator::Curl => DerivativeEvaluation::Curl,
        crate::semantic::DifferentialOperator::RotatedGradient => {
            DerivativeEvaluation::RotatedGradient
        }
        crate::semantic::DifferentialOperator::SymmetricGradient => {
            DerivativeEvaluation::SymmetricGradient
        }
        crate::semantic::DifferentialOperator::TimeDerivative => {
            DerivativeEvaluation::TimeDerivative
        }
    }
}

fn site_for_measure(measure: &SemanticMeasure) -> EvaluationSite {
    match measure {
        SemanticMeasure::Cell { .. } => EvaluationSite::Cell,
        SemanticMeasure::ExteriorFacet { .. } => EvaluationSite::ExteriorTrace,
        SemanticMeasure::InteriorFacet { .. } | SemanticMeasure::Interface { .. } => {
            EvaluationSite::MinusTrace
        }
        SemanticMeasure::Point { .. } => EvaluationSite::Point,
    }
}

fn site_for_trace(side: TraceSide) -> EvaluationSite {
    match side {
        TraceSide::Exterior => EvaluationSite::ExteriorTrace,
        TraceSide::Minus => EvaluationSite::MinusTrace,
        TraceSide::Plus => EvaluationSite::PlusTrace,
    }
}

fn side_for_site(site: EvaluationSite) -> TensorSide {
    match site {
        EvaluationSite::Cell => TensorSide::Cell,
        EvaluationSite::ExteriorTrace => TensorSide::Exterior,
        EvaluationSite::MinusTrace => TensorSide::Minus,
        EvaluationSite::PlusTrace => TensorSide::Plus,
        EvaluationSite::Point => TensorSide::Point,
    }
}

fn constant(value: f64) -> TensorScalarExpr {
    TensorScalarExpr::Constant { value }
}

fn unary(op: TensorUnaryOp, arg: TensorScalarExpr) -> TensorScalarExpr {
    simplify(TensorScalarExpr::Unary {
        op,
        arg: Box::new(arg),
    })
}

fn binary(op: TensorBinaryOp, lhs: TensorScalarExpr, rhs: TensorScalarExpr) -> TensorScalarExpr {
    simplify(TensorScalarExpr::Binary {
        op,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    })
}

fn is_zero(expression: &TensorScalarExpr) -> bool {
    matches!(expression, TensorScalarExpr::Constant { value } if *value == 0.0)
}

fn is_one(expression: &TensorScalarExpr) -> bool {
    matches!(expression, TensorScalarExpr::Constant { value } if *value == 1.0)
}

fn simplify(expression: TensorScalarExpr) -> TensorScalarExpr {
    match expression {
        TensorScalarExpr::Unary { op, arg } => {
            let arg = simplify(*arg);
            if op == TensorUnaryOp::Neg
                && let TensorScalarExpr::Reduction {
                    op: reduction,
                    axis,
                    expression,
                } = arg
            {
                return TensorScalarExpr::Reduction {
                    op: reduction,
                    axis,
                    expression: Box::new(unary(TensorUnaryOp::Neg, *expression)),
                };
            }
            if op == TensorUnaryOp::Neg && is_zero(&arg) {
                constant(0.0)
            } else if let TensorScalarExpr::Constant { value } = arg {
                let value = match op {
                    TensorUnaryOp::Neg => -value,
                    TensorUnaryOp::Abs => value.abs(),
                    TensorUnaryOp::Sqrt => value.sqrt(),
                    TensorUnaryOp::Exp => value.exp(),
                    TensorUnaryOp::Ln => value.ln(),
                    TensorUnaryOp::Sin => value.sin(),
                    TensorUnaryOp::Cos => value.cos(),
                    TensorUnaryOp::Tan => value.tan(),
                    TensorUnaryOp::Floor => value.floor(),
                    TensorUnaryOp::Ceil => value.ceil(),
                };
                constant(value)
            } else {
                TensorScalarExpr::Unary {
                    op,
                    arg: Box::new(arg),
                }
            }
        }
        TensorScalarExpr::Binary { op, lhs, rhs } => {
            let lhs = simplify(*lhs);
            let rhs = simplify(*rhs);
            match op {
                TensorBinaryOp::Add if is_zero(&lhs) => rhs,
                TensorBinaryOp::Add if is_zero(&rhs) => lhs,
                TensorBinaryOp::Sub if is_zero(&rhs) => lhs,
                TensorBinaryOp::Mul if is_zero(&lhs) || is_zero(&rhs) => constant(0.0),
                TensorBinaryOp::Mul if is_one(&lhs) => rhs,
                TensorBinaryOp::Mul if is_one(&rhs) => lhs,
                TensorBinaryOp::Div if is_zero(&lhs) => constant(0.0),
                TensorBinaryOp::Div if is_one(&rhs) => lhs,
                _ => TensorScalarExpr::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            }
        }
        TensorScalarExpr::Reduction {
            op,
            axis,
            expression,
        } => {
            let expression = simplify(*expression);
            if is_zero(&expression) {
                constant(0.0)
            } else {
                TensorScalarExpr::Reduction {
                    op,
                    axis,
                    expression: Box::new(expression),
                }
            }
        }
        other => other,
    }
}

fn input_matches(
    input: &FormalInput,
    symbol: SymbolId,
    evaluation: &BasisEvaluationRequirement,
) -> bool {
    let actual = &input.input.binding.evaluation;
    input.input.binding.symbol == symbol
        && actual.derivative == evaluation.derivative
        && actual.site == evaluation.site
        && (actual.trace_mapping == evaluation.trace_mapping
            || (evaluation.trace_mapping.is_none()
                && actual.trace_mapping == Some(TraceMapping::Value)))
}

fn depends_on(expression: &TensorScalarExpr, input: TensorInputId) -> bool {
    match expression {
        TensorScalarExpr::Input {
            input: candidate, ..
        } => *candidate == input,
        TensorScalarExpr::Unary { arg, .. } => depends_on(arg, input),
        TensorScalarExpr::Binary { lhs, rhs, .. } => {
            depends_on(lhs, input) || depends_on(rhs, input)
        }
        TensorScalarExpr::Reduction { expression, .. } => depends_on(expression, input),
        TensorScalarExpr::Constant { .. } | TensorScalarExpr::IndexEqual { .. } => false,
    }
}

fn differentiate(
    expression: &TensorScalarExpr,
    target: TensorInputId,
    target_indices: &[TensorAxisId],
) -> Result<TensorScalarExpr, TensorCompileError> {
    if !depends_on(expression, target) {
        return Ok(constant(0.0));
    }
    Ok(match expression {
        TensorScalarExpr::Constant { .. } | TensorScalarExpr::IndexEqual { .. } => constant(0.0),
        TensorScalarExpr::Input { input, indices } => {
            if *input != target {
                constant(0.0)
            } else if indices.len() != target_indices.len() {
                return Err(TensorCompileError::Shape(
                    "test derivative rank does not match its indexed access".into(),
                ));
            } else {
                indices
                    .iter()
                    .zip(target_indices)
                    .fold(constant(1.0), |value, (lhs, rhs)| {
                        binary(
                            TensorBinaryOp::Mul,
                            value,
                            TensorScalarExpr::IndexEqual {
                                lhs: *lhs,
                                rhs: *rhs,
                            },
                        )
                    })
            }
        }
        TensorScalarExpr::Unary { op, arg } => derivative_unary(
            *op,
            (**arg).clone(),
            differentiate(arg, target, target_indices)?,
        )?,
        TensorScalarExpr::Binary { op, lhs, rhs } => derivative_binary(
            *op,
            (**lhs).clone(),
            (**rhs).clone(),
            differentiate(lhs, target, target_indices)?,
            differentiate(rhs, target, target_indices)?,
        )?,
        TensorScalarExpr::Reduction {
            op,
            axis,
            expression,
        } => TensorScalarExpr::Reduction {
            op: *op,
            axis: *axis,
            expression: Box::new(differentiate(expression, target, target_indices)?),
        },
    })
}

fn derivative_unary(
    op: TensorUnaryOp,
    arg: TensorScalarExpr,
    derivative: TensorScalarExpr,
) -> Result<TensorScalarExpr, TensorCompileError> {
    if is_zero(&derivative) {
        return Ok(derivative);
    }
    let factor = match op {
        TensorUnaryOp::Neg => return Ok(unary(TensorUnaryOp::Neg, derivative)),
        TensorUnaryOp::Sqrt => binary(
            TensorBinaryOp::Div,
            constant(0.5),
            unary(TensorUnaryOp::Sqrt, arg),
        ),
        TensorUnaryOp::Exp => unary(TensorUnaryOp::Exp, arg),
        TensorUnaryOp::Ln => binary(TensorBinaryOp::Div, constant(1.0), arg),
        TensorUnaryOp::Sin => unary(TensorUnaryOp::Cos, arg),
        TensorUnaryOp::Cos => unary(TensorUnaryOp::Neg, unary(TensorUnaryOp::Sin, arg)),
        TensorUnaryOp::Tan => {
            let cos = unary(TensorUnaryOp::Cos, arg);
            binary(
                TensorBinaryOp::Div,
                constant(1.0),
                binary(TensorBinaryOp::Mul, cos.clone(), cos),
            )
        }
        TensorUnaryOp::Abs | TensorUnaryOp::Floor | TensorUnaryOp::Ceil => {
            return Err(TensorCompileError::Unsupported(format!(
                "symbolic derivative of {op:?}"
            )));
        }
    };
    Ok(binary(TensorBinaryOp::Mul, factor, derivative))
}

fn derivative_binary(
    op: TensorBinaryOp,
    lhs: TensorScalarExpr,
    rhs: TensorScalarExpr,
    dlhs: TensorScalarExpr,
    drhs: TensorScalarExpr,
) -> Result<TensorScalarExpr, TensorCompileError> {
    Ok(match op {
        TensorBinaryOp::Add => binary(TensorBinaryOp::Add, dlhs, drhs),
        TensorBinaryOp::Sub => binary(TensorBinaryOp::Sub, dlhs, drhs),
        TensorBinaryOp::Mul => binary(
            TensorBinaryOp::Add,
            binary(TensorBinaryOp::Mul, dlhs, rhs),
            binary(TensorBinaryOp::Mul, lhs, drhs),
        ),
        TensorBinaryOp::Div => binary(
            TensorBinaryOp::Div,
            binary(
                TensorBinaryOp::Sub,
                binary(TensorBinaryOp::Mul, dlhs, rhs.clone()),
                binary(TensorBinaryOp::Mul, lhs, drhs),
            ),
            binary(TensorBinaryOp::Mul, rhs.clone(), rhs),
        ),
        TensorBinaryOp::Pow => {
            let power = binary(TensorBinaryOp::Pow, lhs.clone(), rhs.clone());
            binary(
                TensorBinaryOp::Mul,
                power,
                binary(
                    TensorBinaryOp::Add,
                    binary(
                        TensorBinaryOp::Mul,
                        drhs,
                        unary(TensorUnaryOp::Ln, lhs.clone()),
                    ),
                    binary(
                        TensorBinaryOp::Mul,
                        rhs,
                        binary(TensorBinaryOp::Div, dlhs, lhs),
                    ),
                ),
            )
        }
        TensorBinaryOp::Min | TensorBinaryOp::Max | TensorBinaryOp::Atan2 => {
            return Err(TensorCompileError::Unsupported(format!(
                "symbolic derivative of {op:?}"
            )));
        }
    })
}

fn directional_derivative(
    expression: &TensorScalarExpr,
    directions: &BTreeMap<TensorInputId, TensorInputId>,
) -> Result<TensorScalarExpr, TensorCompileError> {
    if !directions
        .keys()
        .any(|input| depends_on(expression, *input))
    {
        return Ok(constant(0.0));
    }
    Ok(match expression {
        TensorScalarExpr::Constant { .. } | TensorScalarExpr::IndexEqual { .. } => constant(0.0),
        TensorScalarExpr::Input { input, indices } => directions.get(input).map_or_else(
            || constant(0.0),
            |direction| TensorScalarExpr::Input {
                input: *direction,
                indices: indices.clone(),
            },
        ),
        TensorScalarExpr::Unary { op, arg } => derivative_unary(
            *op,
            (**arg).clone(),
            directional_derivative(arg, directions)?,
        )?,
        TensorScalarExpr::Binary { op, lhs, rhs } => derivative_binary(
            *op,
            (**lhs).clone(),
            (**rhs).clone(),
            directional_derivative(lhs, directions)?,
            directional_derivative(rhs, directions)?,
        )?,
        TensorScalarExpr::Reduction {
            op,
            axis,
            expression,
        } => TensorScalarExpr::Reduction {
            op: *op,
            axis: *axis,
            expression: Box::new(directional_derivative(expression, directions)?),
        },
    })
}

fn remap_inputs(
    expression: &TensorScalarExpr,
    remap: &BTreeMap<TensorInputId, TensorInputId>,
) -> Result<TensorScalarExpr, TensorInputId> {
    Ok(match expression {
        TensorScalarExpr::Constant { value } => constant(*value),
        TensorScalarExpr::Input { input, indices } => TensorScalarExpr::Input {
            input: *remap.get(input).ok_or(*input)?,
            indices: indices.clone(),
        },
        TensorScalarExpr::Unary { op, arg } => TensorScalarExpr::Unary {
            op: *op,
            arg: Box::new(remap_inputs(arg, remap)?),
        },
        TensorScalarExpr::Binary { op, lhs, rhs } => TensorScalarExpr::Binary {
            op: *op,
            lhs: Box::new(remap_inputs(lhs, remap)?),
            rhs: Box::new(remap_inputs(rhs, remap)?),
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
            expression: Box::new(remap_inputs(expression, remap)?),
        },
    })
}
