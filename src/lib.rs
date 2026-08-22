//! `scientia` parses `.res` source and derives scientific compiler semantics.
//!
//! Source syntax is represented by [`scientific::ScientificModule`]; resolved scientific meaning
//! lives exclusively in the typed [`semantic::SemanticModule`] arena. Quantity values use
//! Quantitas directly, while numerical realization and solve strategy belong downstream.

#![forbid(unsafe_code)]

mod algebra;

pub mod derivative;
pub mod evidence;
pub mod form_interpreter;
pub mod formulation;
pub mod id;
pub mod kernel;
pub mod method;
pub mod property_tensor;
pub mod requirements;
pub mod scientific;
pub mod semantic;
pub mod source;
pub mod structural;
pub mod structured;
pub mod system;
pub mod tensor;
pub mod tensor_interpreter;
pub mod verification;

pub use derivative::{
    ActiveSet, Control, DERIVATIVE_REQUEST_SCHEMA, DerivativeConvention, DerivativeDependence,
    DerivativeLevel, DerivativeProductSpec, DerivativeRefusal, DerivativeRequest,
    DerivativeStateConvention, DesignVariable, DifferentiabilityDisposition, Objective,
    ObjectiveSense, ObservableFunctional, ScalarConvention, ShapeDerivativeConvention,
};

pub use evidence::{
    EmpiricalGrade, EvidenceArtifact, EvidenceAxis, EvidenceGrade, EvidenceItem, EvidenceProfile,
    FormalGrade, NumericalGrade, Obligation, ObligationStatus,
};
pub use form_interpreter::{
    FormEvaluation, FormEvaluationContext, FormEvaluationKey, FormInterpretError, FormSample,
    FormValue, interpret_form, interpret_integral, required_evaluations,
};
pub use formulation::{
    BoundaryTermDisposition, BoundaryTermReceipt, FormArgument, FormArgumentRole, FormArity,
    FormAssumption, FormCapture, FormCaptureRole, FormCompileError, FormComplexConvention,
    FormReceipt, FormSide, FormTransformation, VariationalForm, VariationalIntegral,
    compile_variational_form, derive_variational_form, derive_variational_form_for,
};
pub use id::{Digest, ObligationId};
pub use kernel::{
    InputEvaluation, KernelLoweringError, KernelLoweringMethod, KernelLoweringReceipt,
    LocalFactorizationReceipt, LocalFormProgram, LocalInput, LocalInputRole,
    LocalIterationContract, LocalOutput, LocalOutputRole, LocalTransformation, LoweredKernel,
    factor_local_integral, lower_local_program,
};
pub use method::{
    AFFINE_METHOD_KERNEL_SCHEMA, AffineMethodKernel, AffineMethodKernelSpec,
    BoundaryIntegralMethod, ConservationLawMethod, FiniteDifferenceMethod, METHOD_PROGRAM_SCHEMA,
    MethodCompileError, MethodFamily, MethodProgram, MethodProgramKind, MethodReceipt,
    MethodSelectionReceipt, MethodStateBinding, NetworkDaeMethod, ParticleMethod,
    compile_boundary_integral_method, compile_conservation_law_method,
    compile_finite_difference_method, compile_network_dae_method, compile_particle_method,
};
pub use property_tensor::SymmetricTensor2;
pub use requirements::{
    BasisEvaluationRequirement, BoundaryPartitionRequirement, DerivativeEvaluation,
    ElementFamilyRequirement, ElementRequirement, EssentialConstraintRequirement, EvaluationSite,
    FormRequirements, GeometryPreprocessingRequirement, InputPreprocessingRequirement,
    InputSourceRequirement, IntegralOccurrence, KernelSignature, MeasureRequirement,
    NormalizedIntegralGroup, OrientationRequirement, PullbackRequirement, QuadratureIntent,
    QuadraturePrecision, RequirementInferenceError, RequirementInferenceMethod,
    RequirementInferenceReceipt, SpaceBindingRole, SpaceComposition, SpaceRequirement,
    SpaceSystemRequirement, TraceMapping, TraceRequirement, infer_form_requirements,
};
pub use scientific::{
    CouplingGraph, PropertyDefinition, ScientificError, ScientificModel, ScientificModule,
    TimeStateSemantics, canonicalize_authored_quantity, derive_coupling_graph,
    format_scientific_module, parse_scientific_module, parse_scientific_module_diagnostics,
    resolve_modules, semantic_digest, validate_quantities,
};
pub use semantic::{
    Axis, AxisContraction, DeclarationId, DifferentialOperator, DomainId, ExprId, Frame, RegionId,
    RegionKind, SemanticCompilation, SemanticDeclaration, SemanticDeclarationKind, SemanticDomain,
    SemanticExpr, SemanticExprKind, SemanticIntegral, SemanticMeasure, SemanticModel,
    SemanticModule, SemanticRegion, SemanticRole, SemanticShape, SemanticSymbol, SemanticType,
    SymbolId, TraceSide, compile_semantics, elaborate_module, semantic_arena_digest,
};
pub use source::{RelatedSpan, SourceDiagnostic, SourceSeverity, SourceSpan, Spanned};
pub use structural::scc::{Digraph, GraphError, Sccs, tarjan_scc};
pub use structural::{
    AliasAnalysis, AliasClass, Block, BlockKind, DerivativeVariable, DifferentiationStep,
    EquationDerivativeProfile, IncidenceSystem, IndexReductionPlan, Matching, Schedule,
    StructuralCompileError, StructuralError, analyze_aliases, compile_schedule,
    compile_schedule_without_tearing, derivative_profile, maximum_matching, pantelides_plan,
};
pub use structured::{
    STRUCTURED_KERNEL_BUNDLE_SCHEMA, StructuredDerivativeContract, StructuredDerivativeEvidence,
    StructuredDerivativePurpose, StructuredInputOperand, StructuredKernelLoweringMethod,
    StructuredKernelReceipt, StructuredLoweringError, StructuredNumericPolicyReceipt,
    StructuredOperatorKernels, StructuredPointKernelBundle, lower_operator_kernels,
    lower_operator_kernels_with_policy,
};
pub use system::{
    OPERATOR_SYSTEM_SCHEMA, OperatorBlockCoordinate, OperatorSystem, OperatorSystemBlock,
    OperatorSystemError, compile_authored_operator_system, compile_operator_system,
};
pub use tensor::{
    BasisAdjoint, DerivativeConstructionMethod, DerivativeEvaluationPoint, DerivativeEvidence,
    DerivativeMode, DerivativeReceipt, DerivativeStateSemantics, IndexedTensorExpression,
    IntegralOperatorFactorization, OperatorFactorization, OperatorFactorizationMethod,
    OperatorFactorizationReceipt, OperatorStage, QFunctionConstruction, QFunctionInput,
    QFunctionOutput, QFunctionOutputRole, QFunctionProgram, QFunctionReceipt, RestrictionDirection,
    TensorAxis, TensorAxisId, TensorAxisRole, TensorBinaryOp, TensorBinding, TensorCompileError,
    TensorInputId, TensorInputRole, TensorProgram, TensorProgramConstruction, TensorProgramInput,
    TensorProgramInputRole, TensorProgramReceipt, TensorReductionOp, TensorScalarExpr,
    TensorScalarSemantics, TensorSide, TensorUnaryOp, factor_operator,
};
pub use tensor_interpreter::{
    DenseTensor, ElementExecutionContext, OperatorAction, TensorInterpretError,
    interpret_element_operator, interpret_qfunction,
};
pub use verification::{
    ConvergenceExpectation, DerivativeCheckSpec, FormalizableObligationRef, InvariantSpec,
    LimitingCaseSpec, ManufacturedSolutionSpec, ObservableDefinition, ToleranceClass,
    UnsupportedGeneration, VERIFICATION_OBLIGATION_SCHEMA, VERIFICATION_PROFILE_SCHEMA,
    ValidityCondition, VerificationEvidenceClass, VerificationObligation,
    VerificationObligationKind, VerificationProfile, VerificationProfileError,
    derive_verification_profiles,
};
