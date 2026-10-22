//! Conservation-law semantics and finite-volume face kernels (SV2-C1a).
//!
//! A [`ConservationLawSystem`] is the law read from `.res`: one conserved scalar in a `DG`
//! space, `dt(u) + div(F(u)) = S(u)` in conservative form, and the boundary closures the model
//! declares. An [`FvMethodIntent`] is the case's discretization choice (decision D1). The two
//! stay separate until [`lower_conservation_law`] validates the intent against the law and
//! lowers the semantic flux expression (not a caller-supplied coefficient table) into Malleus
//! kernels: `F(u)·n`, the derived characteristic speed `|∂(F·n)/∂u|`, one facet-pair numerical
//! flux, ghost or prescribed-flux boundary kernels, the source and an optional slope limiter.
//!
//! Only physics-neutral numerical-flux families live here (decision D3): `Central`,
//! `LocalLaxFriedrichs` (Rusanov) and `Hll`, all derived from `F` and the derived speed. Named
//! Riemann or entropy fluxes are authored in `.res` as a scalar `constitutive` two-point
//! expression over `trace_minus`/`trace_plus`, `jump`, `average` and `normal_component*`; their
//! consistency and antisymmetry claims are *checked* at caller-supplied samples, never trusted.
//!
//! Facet convention (shared with Malleus [`malleus::FACET_NORMAL_CONVENTION`]): `n` is the unit
//! normal oriented from the minus cell to the plus cell; the numerical flux is the flux leaving
//! the minus cell. `jump(e) = e⁻ − e⁺`, `average(e) = ½(e⁻ + e⁺)`, `normal_component(G)` and
//! `normal_component_minus(G)` are `G·n`, `normal_component_plus(G) = G⁺·(−n)`. On a boundary
//! facet the interior cell is the minus side and the ghost state is the plus side.
//!
//! Every derivative request on an FV kernel that would cross a nonsmooth kink (`abs`, `min`,
//! `max`, `floor`, `ceil` whose switch depends on a differentiated operand) is refused here with
//! `FV_NONSMOOTH_DERIVATIVE_REFUSED` (decision D6, scoped to the FV path); Scientia does not rely
//! on Malleus for it. A `select` is piecewise, not a kink: it differentiates the taken branch.
use crate::id::{Digest, span_independent_digest};
use crate::method::{ConservationLawMethod, MethodCompileError, select_conservation_law};
use crate::scientific::{BinaryOp, BoundaryConditionKind, ValueShape};
use crate::verification::{
    ConservationRelation, ToleranceClass, ValidityCondition, VerificationEvidenceClass,
    VerificationObligation, VerificationObligationKind, make_obligation,
};
use crate::{
    DeclarationId, DifferentialOperator, ExprId, ProviderId, RegionId, SemanticDeclaration,
    SemanticDeclarationKind, SemanticExprKind, SemanticModel, SemanticModule, SemanticRole,
    SemanticShape, SymbolId, TraceSide, semantic_arena_digest,
};
use malleus::{
    AccessMode, BinaryOp as MBinary, BufferBinding, DerivativeMode, DerivativeProduct,
    DerivativeRequest, Executable, FacetOperandRole, FacetPairKernel, FacetSide, IndexingMap,
    Interpreter, IterationDomain, KernelOperand, KernelRegion, LocalId, NumericPolicy, OperandId,
    Predicate, ScalarExpr, Statement, StructuredKernel, SwapParity, UnaryOp as MUnary,
    check_facet_swap_symmetry, differentiate, validate, validate_facet_pair,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const CONSERVATION_LAW_SCHEMA: &str = "scientia-conservation-law/1";
pub const CONSERVATION_LAW_PROGRAM_SCHEMA: &str = "scientia-conservation-law-program/1";
pub const FV_METHOD_INTENT_SCHEMA: &str = "scientia-fv-method-intent/1";
pub const FV_KERNEL_SCHEMA: &str = "scientia-fv-kernel/1";

// ---------------------------------------------------------------------------------------------
// Law
// ---------------------------------------------------------------------------------------------

/// One additive term of the residual moved to the right-hand side: `S = Σ sign·term`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedTerm {
    pub expression: ExprId,
    pub sign: i8,
}

/// Where the characteristic speed comes from. Stage 1 has scalar laws only, whose speed is the
/// symbolic tangent `∂(F·n)/∂u` lowered through Malleus differentiation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeedSource {
    DerivedScalarTangent,
    /// SV2-C1b: the `.res`-declared absolute maximum signal speed `λ_max(U, n)` of a system
    /// (`@wave_speeds(absolute_maximum = ...)`), used by LLF and the CFL bound.
    DeclaredMaximum,
    /// SV2-C1b: `.res`-declared signed extreme speeds `s_min(U, n)`, `s_max(U, n)`
    /// (`@wave_speeds(minimum = ..., maximum = ...)`) for HLL, with a declared absolute
    /// maximum for the CFL bound.
    DeclaredSigned,
}

/// The flux semantics of one declared boundary. Periodicity is a mesh property (decision D2)
/// and appears only in [`ConservationLawSystem::check_boundary_closure`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BoundaryFluxSemantics {
    /// `dirichlet u = g`: the plus-side state `u_g = g(u_in, data, t)`.
    GhostState {
        region: RegionId,
        region_name: String,
        declaration: DeclarationId,
        ghost: ExprId,
    },
    /// `neumann u = h`: the outward normal flux `F_b·n = h(u_in, data, t)` directly.
    PrescribedFlux {
        region: RegionId,
        region_name: String,
        declaration: DeclarationId,
        flux: ExprId,
    },
}

impl BoundaryFluxSemantics {
    pub fn region_name(&self) -> &str {
        match self {
            Self::GhostState { region_name, .. } | Self::PrescribedFlux { region_name, .. } => {
                region_name
            }
        }
    }
}

/// A scalar conservation law `dt(u) + div(flux_sign·F) = S` read from one `.res` equation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConservationLawSystem {
    pub schema: String,
    pub model: String,
    pub source_semantic_digest: Digest,
    /// The FC10 structural selection, unchanged; `method.flux` is the divergence argument.
    pub method: ConservationLawMethod,
    pub spatial_dimension: u8,
    /// `+1` or `-1`: the sign with which `div(method.flux)` enters the normalized law.
    pub flux_sign: i8,
    pub source_terms: Vec<SignedTerm>,
    pub characteristic_speed: SpeedSource,
    pub boundaries: Vec<BoundaryFluxSemantics>,
    pub artifact_digest: Digest,
}

impl ConservationLawSystem {
    /// Every exterior region of a realized mesh must be closed by a declared boundary or be
    /// periodic; a declared boundary that is neither exterior nor periodic is also refused.
    pub fn check_boundary_closure(
        &self,
        exterior_regions: &[&str],
        periodic_regions: &[&str],
    ) -> Result<(), ConservationLawError> {
        let declared = self
            .boundaries
            .iter()
            .map(BoundaryFluxSemantics::region_name)
            .collect::<BTreeSet<_>>();
        for region in exterior_regions {
            if !declared.contains(region) && !periodic_regions.contains(region) {
                return Err(fv_error(
                    "FV_BOUNDARY_UNCLOSED",
                    format!("exterior region `{region}` has no flux semantics and is not periodic"),
                ));
            }
        }
        for region in &declared {
            if periodic_regions.contains(region) {
                return Err(fv_error(
                    "FV_BOUNDARY_UNCLOSED",
                    format!("region `{region}` is both periodic and a declared boundary"),
                ));
            }
            if !exterior_regions.contains(region) {
                return Err(fv_error(
                    "FV_BOUNDARY_UNCLOSED",
                    format!("declared boundary `{region}` is not an exterior region of the mesh"),
                ));
            }
        }
        Ok(())
    }
}

/// Read a scalar conservation law from `equation_name` in `model_name`.
pub fn compile_conservation_law_system(
    module: &SemanticModule,
    model_name: &str,
    equation_name: &str,
    conserved_name: &str,
) -> Result<ConservationLawSystem, ConservationLawError> {
    let (model, equation, method) =
        select_conservation_law(module, model_name, equation_name, conserved_name)?;
    let conserved = &model.symbols[method.conserved.index()];
    if !matches!(
        conserved.ty.shape,
        SemanticShape::Numeric(ValueShape::Scalar)
    ) {
        return Err(fv_error(
            "FV_SYSTEM_UNSUPPORTED",
            format!(
                "conserved field `{}` is not scalar; systems need declared speeds (SV2-C1b)",
                conserved.name
            ),
        ));
    }
    let (flux_sign, source_terms) = normalize_law(model, equation, &method)?;
    let boundaries = read_boundaries(model, method.conserved)?;
    let source_semantic_digest = Digest {
        algorithm: "blake3".into(),
        hex: semantic_arena_digest(module),
    };
    let mut system = ConservationLawSystem {
        schema: CONSERVATION_LAW_SCHEMA.into(),
        model: model.name.clone(),
        source_semantic_digest,
        spatial_dimension: model.domains[method.domain.index()].spatial_dimension,
        method,
        flux_sign,
        source_terms,
        characteristic_speed: SpeedSource::DerivedScalarTangent,
        boundaries,
        artifact_digest: empty_digest(),
    };
    system.artifact_digest = span_independent_digest(&system);
    Ok(system)
}

/// Normalize `equation` to `dt(u) + div(flux_sign·F) = Σ sign·term` (shared by the scalar law
/// and SV2-C1b systems): the time derivative and the flux divergence must be unscaled additive
/// terms; every other term is a source moved to the right-hand side.
pub(crate) fn normalize_law(
    model: &SemanticModel,
    equation: &SemanticDeclaration,
    method: &ConservationLawMethod,
) -> Result<(i8, Vec<SignedTerm>), ConservationLawError> {
    let SemanticDeclarationKind::Equation { lhs, rhs } = equation.kind else {
        unreachable!("selection returns an equation");
    };
    let mut terms = Vec::new();
    additive_terms(model, lhs, 1, &mut terms);
    additive_terms(model, rhs, -1, &mut terms);
    let mut time_sign = None;
    let mut flux_sign = None;
    let mut others = Vec::new();
    for (expression, sign) in terms {
        if expression == method.time_derivative {
            if time_sign.replace(sign).is_some() {
                return Err(fv_error(
                    "FV_STORAGE_SCALED",
                    "the time derivative appears more than once",
                ));
            }
        } else if expression == method.flux_divergence {
            if flux_sign.replace(sign).is_some() {
                return Err(fv_error(
                    "FV_FLUX_NOT_CONSERVATIVE",
                    "the flux divergence appears more than once",
                ));
            }
        } else if !is_zero_literal(model, expression) {
            others.push((expression, sign));
        }
    }
    let time_sign = time_sign.ok_or_else(|| {
        fv_error(
            "FV_STORAGE_SCALED",
            "dt(u) must be an unscaled additive term of the equation",
        )
    })?;
    let flux_sign = flux_sign.ok_or_else(|| {
        fv_error(
            "FV_FLUX_NOT_CONSERVATIVE",
            "div(F) must be an unscaled additive term of the equation",
        )
    })?;
    for (expression, _) in &others {
        if contains_differential(model, *expression, |operator| {
            operator == DifferentialOperator::Divergence
                || operator == DifferentialOperator::TimeDerivative
        }) {
            return Err(fv_error(
                "FV_FLUX_NOT_CONSERVATIVE",
                format!("term {expression} holds a second divergence or time derivative"),
            ));
        }
    }
    let source_terms = others
        .into_iter()
        .map(|(expression, sign)| SignedTerm {
            expression,
            sign: -sign * time_sign,
        })
        .collect();
    Ok((flux_sign * time_sign, source_terms))
}

/// The `dirichlet` (ghost state) and `neumann` (prescribed normal flux) closures declared for
/// `conserved`, sorted by region name; one closure per region.
pub(crate) fn read_boundaries(
    model: &SemanticModel,
    conserved: SymbolId,
) -> Result<Vec<BoundaryFluxSemantics>, ConservationLawError> {
    let mut boundaries = Vec::new();
    for declaration in &model.declarations {
        let SemanticDeclarationKind::BoundaryCondition {
            region,
            target,
            condition,
            value,
            ..
        } = &declaration.kind
        else {
            continue;
        };
        if *target != Some(conserved) {
            continue;
        }
        let region_name = model.regions[region.index()].name.clone();
        boundaries.push(match condition {
            BoundaryConditionKind::Dirichlet => BoundaryFluxSemantics::GhostState {
                region: *region,
                region_name,
                declaration: declaration.id,
                ghost: *value,
            },
            BoundaryConditionKind::Neumann => BoundaryFluxSemantics::PrescribedFlux {
                region: *region,
                region_name,
                declaration: declaration.id,
                flux: *value,
            },
            other => {
                return Err(fv_error(
                    "FV_BOUNDARY_UNSUPPORTED",
                    format!(
                        "boundary `{}` uses {other:?}; finite volume closes boundaries by ghost state (dirichlet) or prescribed normal flux (neumann)",
                        declaration.name
                    ),
                ));
            }
        });
    }
    boundaries.sort_by(|a, b| a.region_name().cmp(b.region_name()));
    if boundaries
        .windows(2)
        .any(|pair| pair[0].region_name() == pair[1].region_name())
    {
        return Err(fv_error(
            "FV_BOUNDARY_UNSUPPORTED",
            "a region carries more than one boundary closure",
        ));
    }
    Ok(boundaries)
}

pub(crate) fn additive_terms(
    model: &SemanticModel,
    id: ExprId,
    sign: i8,
    out: &mut Vec<(ExprId, i8)>,
) {
    match model.expressions[id.index()].kind {
        SemanticExprKind::Binary {
            op: BinaryOp::Add,
            lhs,
            rhs,
        } => {
            additive_terms(model, lhs, sign, out);
            additive_terms(model, rhs, sign, out);
        }
        SemanticExprKind::Binary {
            op: BinaryOp::Sub,
            lhs,
            rhs,
        } => {
            additive_terms(model, lhs, sign, out);
            additive_terms(model, rhs, -sign, out);
        }
        SemanticExprKind::Unary { arg, .. } => additive_terms(model, arg, -sign, out),
        _ => out.push((id, sign)),
    }
}

pub(crate) fn is_zero_literal(model: &SemanticModel, id: ExprId) -> bool {
    matches!(model.expressions[id.index()].kind, SemanticExprKind::Number { value, .. } if value == 0.0)
}

pub(crate) fn contains_differential(
    model: &SemanticModel,
    root: ExprId,
    predicate: impl Fn(DifferentialOperator) -> bool,
) -> bool {
    let mut stack = vec![root];
    let mut seen = BTreeSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let kind = &model.expressions[id.index()].kind;
        if let SemanticExprKind::Differential { operator, .. } = kind
            && predicate(*operator)
        {
            return true;
        }
        if let SemanticExprKind::Symbol { symbol } = kind
            && let Some(value) = crate::point_expression::symbol_definition(model, *symbol)
        {
            stack.push(value);
        }
        stack.extend(crate::formulation::expression_children(kind));
    }
    false
}

// ---------------------------------------------------------------------------------------------
// Method intent
// ---------------------------------------------------------------------------------------------

/// The case's finite-volume discretization choice, canonical-JSON digested.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FvMethodIntent {
    pub numerical_flux: NumericalFluxIntent,
    pub reconstruction: ReconstructionIntent,
    pub source: SourceTreatment,
    pub positivity: PositivityIntent,
}

impl FvMethodIntent {
    pub fn digest(&self) -> Digest {
        span_independent_digest(&IntentPayload {
            schema: FV_METHOD_INTENT_SCHEMA,
            intent: self,
        })
    }
}

#[derive(Serialize)]
struct IntentPayload<'a> {
    schema: &'static str,
    intent: &'a FvMethodIntent,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NumericalFluxIntent {
    /// `½(F⁻ + F⁺)·n`: smooth problems and testing only; not monotone.
    Central,
    /// Rusanov: `½(F⁻ + F⁺)·n − ½ max(λ⁻, λ⁺)(u⁺ − u⁻)`.
    LocalLaxFriedrichs,
    /// HLLE with `s_min = min(s⁻, s⁺)`, `s_max = max(s⁻, s⁺)` from the signed derived speed.
    Hll,
    /// A scalar `constitutive` two-point expression of the model. `tolerance` is the absolute
    /// bound for both checked claims, recorded in the program identity.
    Authored { declaration: String, tolerance: f64 },
}

impl NumericalFluxIntent {
    const fn label(&self) -> &'static str {
        match self {
            Self::Central => "central",
            Self::LocalLaxFriedrichs => "local_lax_friedrichs",
            Self::Hll => "hll",
            Self::Authored { .. } => "authored",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReconstructionIntent {
    Constant,
    Muscl {
        variables: ReconstructionVariables,
        gradient: GradientScheme,
        limiter: LimiterFamily,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReconstructionVariables {
    Conserved,
    Primitive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GradientScheme {
    LeastSquares,
    GreenGauss,
    Axis1D,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimiterFamily {
    None,
    Minmod,
    VanLeer,
    MonotonizedCentral,
    Superbee,
    BarthJespersen,
    Venkatakrishnan { k: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceTreatment {
    Pointwise,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PositivityIntent {
    None,
    ZhangShuScaling { thresholds: Vec<f64> },
}

// ---------------------------------------------------------------------------------------------
// Lowered program
// ---------------------------------------------------------------------------------------------

/// What one kernel operand carries.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FvOperandMeaning {
    /// The conserved state; `side` is set on facet-pair kernels.
    State {
        side: Option<FacetSide>,
    },
    /// Component `axis` of the unit normal (oriented minus to plus, or outward at a boundary).
    Normal {
        axis: u8,
    },
    /// Model data, indexed into [`ConservationLawProgram::data`]. Uniform data binds stride 0.
    Datum {
        datum: usize,
    },
    /// Slope ratio input of a Sweby limiter.
    SlopeRatio,
    Output,
    /// SV2-C1b: slot `component` of a system's conserved state vector; `side` as for `State`.
    StateComponent {
        side: Option<FacetSide>,
        component: usize,
    },
    /// SV2-C1b: slot `component` of a system's primitive vector `W` (inverse-map input).
    PrimitiveComponent {
        component: usize,
    },
    /// SV2-C1b: output `component` of a multi-output kernel: a state-vector slot for face
    /// flux, ghost, prescribed-flux, source and inverse-map kernels; a primitive slot for the
    /// forward map.
    OutputComponent {
        component: usize,
    },
}

/// External model data read by the lowered kernels.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FvDatumSource {
    /// A valueless `parameter`, `constant`, `source` or `input value` component.
    Symbol { symbol: SymbolId, component: u8 },
    /// A provider call whose arguments do not depend on the state, sampled by the consumer.
    ProviderCall {
        expression: ExprId,
        provider: ProviderId,
    },
    /// Model time `t`.
    Time,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FvDatum {
    pub name: String,
    pub source: FvDatumSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NonsmoothKind {
    Abs,
    Min,
    Max,
    Floor,
    Ceil,
}

/// One kink of a lowered kernel and the operands its switch depends on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NonsmoothSite {
    pub statement: usize,
    pub kind: NonsmoothKind,
    pub switch_operands: Vec<OperandId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FvKernelRole {
    PhysicalFluxNormal,
    CharacteristicSpeed,
    FaceFlux,
    Ghost,
    PrescribedBoundaryFlux,
    Source,
    Limiter,
    /// SV2-C1b: declared signed minimum speed `s_min(U, n)` (HLL).
    SignalSpeedMinimum,
    /// SV2-C1b: declared signed maximum speed `s_max(U, n)` (HLL).
    SignalSpeedMaximum,
    /// SV2-C1b: `W(U)`, conserved to primitive.
    PrimitiveForward,
    /// SV2-C1b: `U(W)`, primitive to conserved.
    PrimitiveInverse,
    /// SV2-C1b: minimum margin of the declared strictly-positive expressions.
    Admissibility,
}

/// One lowered kernel. `facet_roles` is set exactly for the facet-pair face flux.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FvKernel {
    pub role: FvKernelRole,
    pub kernel: StructuredKernel,
    pub operands: Vec<FvOperandMeaning>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub facet_roles: Option<Vec<FacetOperandRole>>,
    pub nonsmooth: Vec<NonsmoothSite>,
    pub digest: Digest,
}

impl FvKernel {
    pub fn facet_pair(&self) -> Option<FacetPairKernel> {
        self.facet_roles.as_ref().map(|roles| FacetPairKernel {
            kernel: self.kernel.clone(),
            roles: roles.clone(),
        })
    }

    pub fn operand(&self, meaning: &FvOperandMeaning) -> Option<OperandId> {
        self.operands
            .iter()
            .position(|item| item == meaning)
            .map(OperandId::new)
    }

    /// Differentiate this kernel, refusing every request whose independents reach a kink
    /// switch (decision D6). Smooth kernels and smooth directions keep their derivatives.
    pub fn differentiate(
        &self,
        request: &DerivativeRequest,
    ) -> Result<DerivativeProduct, ConservationLawError> {
        check_smooth_request(&self.nonsmooth, request)?;
        differentiate(&self.kernel, request)
            .map_err(|error| fv_error("FV_DERIVATIVE_FAILED", format!("{error:?}")))
    }
}

/// Refuse a derivative request that crosses a nonsmooth kink (D6).
pub fn check_smooth_request(
    sites: &[NonsmoothSite],
    request: &DerivativeRequest,
) -> Result<(), ConservationLawError> {
    let independents = request
        .independent_operands
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let crossing = sites
        .iter()
        .filter(|site| {
            site.switch_operands
                .iter()
                .any(|operand| independents.contains(operand))
        })
        .collect::<Vec<_>>();
    if crossing.is_empty() {
        return Ok(());
    }
    Err(fv_error(
        "FV_NONSMOOTH_DERIVATIVE_REFUSED",
        format!(
            "derivative crosses {} nonsmooth site(s) ({}); nonsmooth derivatives are refused until frozen-limiter semantics are settled",
            crossing.len(),
            crossing
                .iter()
                .map(|site| format!("{:?}@{}", site.kind, site.statement))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    ))
}

/// A caller-supplied facet sample for checking flux claims. `data` is keyed by
/// [`FvDatum::name`]; every datum the face flux reads must be present.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FvFluxSample {
    pub minus: f64,
    pub plus: f64,
    pub normal: Vec<f64>,
    pub data: BTreeMap<String, f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FluxClaim {
    /// `F*(u, u, n) = F(u)·n`.
    Consistency,
    /// `F*(u⁻, u⁺, n) = −F*(u⁺, u⁻, −n)` via `malleus::check_facet_swap_symmetry`.
    Antisymmetry,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FluxClaimCheck {
    pub claim: FluxClaim,
    pub samples: usize,
    pub max_absolute: f64,
    pub tolerance: f64,
}

/// The lowered finite-volume method for one conservation law.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConservationLawProgram {
    pub schema: String,
    pub system_digest: Digest,
    pub intent: FvMethodIntent,
    pub intent_digest: Digest,
    pub spatial_dimension: u8,
    pub data: Vec<FvDatum>,
    pub physical_flux_normal: FvKernel,
    pub characteristic_speed: FvKernel,
    pub face_flux: FvKernel,
    /// Boundary kernels in [`ConservationLawSystem::boundaries`] order.
    pub boundaries: Vec<FvKernel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<FvKernel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limiter: Option<FvKernel>,
    pub checks: Vec<FluxClaimCheck>,
    pub obligations: Vec<VerificationObligation>,
    pub artifact_digest: Digest,
}

impl ConservationLawProgram {
    pub fn validate_identity(&self) -> Result<(), ConservationLawError> {
        let mut payload = self.clone();
        payload.artifact_digest = empty_digest();
        if self.schema != CONSERVATION_LAW_PROGRAM_SCHEMA
            || span_independent_digest(&payload) != self.artifact_digest
        {
            return Err(fv_error(
                "FV_IDENTITY_MISMATCH",
                "conservation-law program changed",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{code}: {message}")]
pub struct ConservationLawError {
    pub code: &'static str,
    pub message: String,
}

impl From<MethodCompileError> for ConservationLawError {
    fn from(error: MethodCompileError) -> Self {
        fv_error("FV_METHOD_SELECTION", error.to_string())
    }
}

pub(crate) fn fv_error(code: &'static str, message: impl ToString) -> ConservationLawError {
    ConservationLawError {
        code,
        message: message.to_string(),
    }
}

pub(crate) fn empty_digest() -> Digest {
    Digest {
        algorithm: "blake3".into(),
        hex: String::new(),
    }
}

/// Validate `intent` against `system` and lower every kernel. Authored fluxes need at least
/// one sample; generic fluxes check antisymmetry at the samples given (exactly, tolerance 0).
pub fn lower_conservation_law(
    module: &SemanticModule,
    system: &ConservationLawSystem,
    intent: &FvMethodIntent,
    samples: &[FvFluxSample],
) -> Result<ConservationLawProgram, ConservationLawError> {
    let model = module
        .models
        .iter()
        .find(|model| model.name == system.model)
        .ok_or_else(|| fv_error("FV_METHOD_SELECTION", "model is absent"))?;
    if semantic_arena_digest(module) != system.source_semantic_digest.hex {
        return Err(fv_error(
            "FV_IDENTITY_MISMATCH",
            "system was read from a different semantic module",
        ));
    }
    let limiter = validate_intent(intent)?;
    let dimension = system.spatial_dimension;
    let mut data = DataTable::default();
    let lower = Lower {
        model,
        components: vec![(system.method.conserved, 0, 1)],
        dimension,
    };

    let flux_components = {
        let mut probe = KernelBuilder::new("probe");
        let state = probe.operand("u", FvOperandMeaning::State { side: None });
        let components = lower.expr(
            &mut probe,
            &mut data,
            system.method.flux,
            &StateContext::Point(state),
        )?;
        components.len()
    };
    if flux_components != usize::from(dimension) {
        return Err(fv_error(
            "FV_FLUX_SHAPE",
            format!("flux has {flux_components} component(s) on a {dimension}-dimensional domain"),
        ));
    }

    // F(u)·n and its tangent ∂(F·n)/∂u, the base of every derived kernel.
    let physical = {
        let mut builder = KernelBuilder::new("physical_flux_normal");
        let state = builder.operand("u", FvOperandMeaning::State { side: None });
        let normal = builder.normals(dimension);
        let value = lower.flux_normal(
            &mut builder,
            &mut data,
            system,
            &StateContext::Point(state),
            &normal,
        )?;
        builder.finish(FvKernelRole::PhysicalFluxNormal, value, None)?
    };
    let state = physical
        .operand(&FvOperandMeaning::State { side: None })
        .expect("physical flux has a state operand");
    let output = physical
        .operand(&FvOperandMeaning::Output)
        .expect("physical flux has an output");
    let tangent = differentiate(
        &physical.kernel,
        &DerivativeRequest {
            mode: DerivativeMode::Jvp,
            independent_operands: vec![state],
            dependent_operands: vec![output],
        },
    )
    .map_err(|error| fv_error("FV_DERIVATIVE_FAILED", format!("{error:?}")))?;
    let derived = Derived {
        physical: &physical,
        tangent: &tangent,
    };

    let characteristic_speed = {
        let mut builder = KernelBuilder::new("characteristic_speed");
        let state = builder.operand("u", FvOperandMeaning::State { side: None });
        let normal = builder.normals(dimension);
        let signed = derived.inline_tangent(&mut builder, ScalarExpr::Load(state), &normal)?;
        builder.finish(
            FvKernelRole::CharacteristicSpeed,
            ScalarExpr::unary(MUnary::Abs, signed),
            None,
        )?
    };

    let face_flux = {
        let mut builder =
            KernelBuilder::new(&format!("face_flux_{}", intent.numerical_flux.label()));
        let minus = builder.operand(
            "u_minus",
            FvOperandMeaning::State {
                side: Some(FacetSide::Minus),
            },
        );
        let plus = builder.operand(
            "u_plus",
            FvOperandMeaning::State {
                side: Some(FacetSide::Plus),
            },
        );
        let normal = builder.normals(dimension);
        let um = ScalarExpr::Load(minus);
        let up = ScalarExpr::Load(plus);
        let value = match &intent.numerical_flux {
            NumericalFluxIntent::Central => {
                let fm = derived.inline_primal(&mut builder, um, &normal)?;
                let fp = derived.inline_primal(&mut builder, up, &normal)?;
                half(add(fm, fp))
            }
            NumericalFluxIntent::LocalLaxFriedrichs => {
                let fm = derived.inline_primal(&mut builder, um.clone(), &normal)?;
                let fp = derived.inline_primal(&mut builder, up.clone(), &normal)?;
                let sm = derived.inline_tangent(&mut builder, um.clone(), &normal)?;
                let sp = derived.inline_tangent(&mut builder, up.clone(), &normal)?;
                let lambda = builder.bind(ScalarExpr::binary(
                    MBinary::Max,
                    ScalarExpr::unary(MUnary::Abs, sm),
                    ScalarExpr::unary(MUnary::Abs, sp),
                ));
                ScalarExpr::binary(
                    MBinary::Sub,
                    half(add(fm, fp)),
                    ScalarExpr::binary(
                        MBinary::Mul,
                        half(lambda),
                        ScalarExpr::binary(MBinary::Sub, up, um),
                    ),
                )
            }
            NumericalFluxIntent::Hll => {
                let fm = derived.inline_primal(&mut builder, um.clone(), &normal)?;
                let fp = derived.inline_primal(&mut builder, up.clone(), &normal)?;
                let sm = derived.inline_tangent(&mut builder, um.clone(), &normal)?;
                let sp = derived.inline_tangent(&mut builder, up.clone(), &normal)?;
                let smin = builder.bind(ScalarExpr::binary(MBinary::Min, sm.clone(), sp.clone()));
                let smax = builder.bind(ScalarExpr::binary(MBinary::Max, sm, sp));
                let numerator = ScalarExpr::binary(
                    MBinary::Add,
                    ScalarExpr::binary(
                        MBinary::Sub,
                        ScalarExpr::binary(MBinary::Mul, smax.clone(), fm.clone()),
                        ScalarExpr::binary(MBinary::Mul, smin.clone(), fp.clone()),
                    ),
                    ScalarExpr::binary(
                        MBinary::Mul,
                        ScalarExpr::binary(MBinary::Mul, smin.clone(), smax.clone()),
                        ScalarExpr::binary(MBinary::Sub, up, um),
                    ),
                );
                let middle = ScalarExpr::binary(
                    MBinary::Div,
                    numerator,
                    ScalarExpr::binary(MBinary::Sub, smax.clone(), smin.clone()),
                );
                select(
                    compare(
                        malleus::CompareOp::GreaterEqual,
                        smin,
                        ScalarExpr::Constant(0.0),
                    ),
                    fm,
                    select(
                        compare(
                            malleus::CompareOp::LessEqual,
                            smax,
                            ScalarExpr::Constant(0.0),
                        ),
                        fp,
                        middle,
                    ),
                )
            }
            NumericalFluxIntent::Authored { declaration, .. } => {
                let authored = authored_declaration(model, declaration)?;
                let components = lower.expr(
                    &mut builder,
                    &mut data,
                    authored,
                    &StateContext::Facet {
                        minus: vec![minus],
                        plus: vec![plus],
                        side: None,
                        normal: normal.clone(),
                    },
                )?;
                let [value] = <[ScalarExpr; 1]>::try_from(components).map_err(|_| {
                    fv_error(
                        "FV_AUTHORED_FLUX_SHAPE",
                        format!("authored flux `{declaration}` is not scalar"),
                    )
                })?;
                value
            }
        };
        let roles = builder.facet_roles(minus, plus);
        builder.finish(FvKernelRole::FaceFlux, value, Some(roles))?
    };

    let mut boundaries = Vec::new();
    for boundary in &system.boundaries {
        let (role, name, expression) = match boundary {
            BoundaryFluxSemantics::GhostState {
                region_name, ghost, ..
            } => (FvKernelRole::Ghost, format!("ghost_{region_name}"), *ghost),
            BoundaryFluxSemantics::PrescribedFlux {
                region_name, flux, ..
            } => (
                FvKernelRole::PrescribedBoundaryFlux,
                format!("boundary_flux_{region_name}"),
                *flux,
            ),
        };
        let mut builder = KernelBuilder::new(&name);
        let state = builder.operand("u_in", FvOperandMeaning::State { side: None });
        builder.normals(dimension);
        let components = lower.expr(
            &mut builder,
            &mut data,
            expression,
            &StateContext::Point(state),
        )?;
        let [value] = <[ScalarExpr; 1]>::try_from(components).map_err(|_| {
            fv_error(
                "FV_BOUNDARY_UNSUPPORTED",
                format!("boundary `{}` value is not scalar", boundary.region_name()),
            )
        })?;
        boundaries.push(builder.finish(role, value, None)?);
    }

    let source = if system.source_terms.is_empty() {
        None
    } else {
        let mut builder = KernelBuilder::new("source");
        let state = builder.operand("u", FvOperandMeaning::State { side: None });
        let mut total: Option<ScalarExpr> = None;
        for term in &system.source_terms {
            let components = lower.expr(
                &mut builder,
                &mut data,
                term.expression,
                &StateContext::Point(state),
            )?;
            let [value] = <[ScalarExpr; 1]>::try_from(components)
                .map_err(|_| fv_error("FV_SOURCE_UNSUPPORTED", "source term is not scalar"))?;
            let value = if term.sign < 0 {
                ScalarExpr::unary(MUnary::Neg, value)
            } else {
                value
            };
            total = Some(match total {
                None => value,
                Some(total) => add(total, value),
            });
        }
        Some(builder.finish(FvKernelRole::Source, total.expect("nonempty source"), None)?)
    };

    let limiter = limiter.map(limiter_kernel).transpose()?;

    let mut checks = Vec::new();
    let tolerance = match &intent.numerical_flux {
        NumericalFluxIntent::Authored { tolerance, .. } => {
            if !(tolerance.is_finite() && *tolerance >= 0.0) {
                return Err(fv_error(
                    "FV_SAMPLE_INVALID",
                    "authored-flux tolerance must be finite and nonnegative",
                ));
            }
            if samples.is_empty() {
                return Err(fv_error(
                    "FV_AUTHORED_FLUX_UNCHECKED",
                    "an authored numerical flux needs samples to check its claims",
                ));
            }
            *tolerance
        }
        _ => 0.0,
    };
    let data = data.into_data();
    if !samples.is_empty() {
        let consistency = check_consistency(&face_flux, &physical, &data, samples, dimension)?;
        let antisymmetry = check_antisymmetry(&face_flux, &data, samples, dimension)?;
        if consistency > tolerance {
            return Err(fv_error(
                "FV_AUTHORED_FLUX_INCONSISTENT",
                format!("F*(u,u,n) deviates from F(u)·n by {consistency:e} > {tolerance:e}"),
            ));
        }
        if antisymmetry > tolerance {
            return Err(fv_error(
                "FV_AUTHORED_FLUX_NOT_ANTISYMMETRIC",
                format!("F*(u⁻,u⁺,n) + F*(u⁺,u⁻,−n) deviates by {antisymmetry:e} > {tolerance:e}"),
            ));
        }
        checks.push(FluxClaimCheck {
            claim: FluxClaim::Consistency,
            samples: samples.len(),
            max_absolute: consistency,
            tolerance,
        });
        checks.push(FluxClaimCheck {
            claim: FluxClaim::Antisymmetry,
            samples: samples.len(),
            max_absolute: antisymmetry,
            tolerance,
        });
    }

    let obligations = obligations(model, system, intent, &checks);
    let mut program = ConservationLawProgram {
        schema: CONSERVATION_LAW_PROGRAM_SCHEMA.into(),
        system_digest: system.artifact_digest.clone(),
        intent: intent.clone(),
        intent_digest: intent.digest(),
        spatial_dimension: dimension,
        data,
        physical_flux_normal: physical,
        characteristic_speed,
        face_flux,
        boundaries,
        source,
        limiter,
        checks,
        obligations,
        artifact_digest: empty_digest(),
    };
    program.artifact_digest = span_independent_digest(&program);
    Ok(program)
}

pub(crate) fn validate_intent(intent: &FvMethodIntent) -> Result<Option<LimiterFamily>, ConservationLawError> {
    if let PositivityIntent::ZhangShuScaling { .. } = intent.positivity {
        return Err(fv_error(
            "FV_POSITIVITY_UNSUPPORTED",
            "Zhang-Shu positivity scaling needs declared admissible sets (SV2-C1b)",
        ));
    }
    match &intent.reconstruction {
        ReconstructionIntent::Constant => Ok(None),
        ReconstructionIntent::Muscl {
            variables, limiter, ..
        } => {
            if *variables == ReconstructionVariables::Primitive {
                return Err(fv_error(
                    "FV_PRIMITIVE_MAP_MISSING",
                    "a scalar law declares no primitive map",
                ));
            }
            match limiter {
                LimiterFamily::None => Ok(None),
                LimiterFamily::Minmod
                | LimiterFamily::VanLeer
                | LimiterFamily::MonotonizedCentral
                | LimiterFamily::Superbee => Ok(Some(limiter.clone())),
                LimiterFamily::BarthJespersen | LimiterFamily::Venkatakrishnan { .. } => {
                    Err(fv_error(
                        "FV_LIMITER_UNSUPPORTED",
                        format!("{limiter:?} needs neighbour bounds; it is not lowered yet"),
                    ))
                }
            }
        }
    }
}

/// Sweby limiter `φ(r)` as a point kernel.
pub(crate) fn limiter_kernel(limiter: LimiterFamily) -> Result<FvKernel, ConservationLawError> {
    let mut builder = KernelBuilder::new(&format!("limiter_{limiter:?}").to_lowercase());
    let r = ScalarExpr::Load(builder.operand("r", FvOperandMeaning::SlopeRatio));
    let c = ScalarExpr::Constant;
    let min = |a, b| ScalarExpr::binary(MBinary::Min, a, b);
    let max = |a, b| ScalarExpr::binary(MBinary::Max, a, b);
    let mul = |a, b| ScalarExpr::binary(MBinary::Mul, a, b);
    let value = match limiter {
        LimiterFamily::Minmod => max(c(0.0), min(c(1.0), r)),
        LimiterFamily::VanLeer => {
            let abs = ScalarExpr::unary(MUnary::Abs, r.clone());
            ScalarExpr::binary(MBinary::Div, add(r, abs.clone()), add(c(1.0), abs))
        }
        LimiterFamily::MonotonizedCentral => max(
            c(0.0),
            min(min(mul(c(2.0), r.clone()), half(add(c(1.0), r))), c(2.0)),
        ),
        LimiterFamily::Superbee => max(
            c(0.0),
            max(min(mul(c(2.0), r.clone()), c(1.0)), min(r, c(2.0))),
        ),
        _ => unreachable!("validated limiter family"),
    };
    builder.finish(FvKernelRole::Limiter, value, None)
}

pub(crate) fn authored_declaration(model: &SemanticModel, name: &str) -> Result<ExprId, ConservationLawError> {
    model
        .declarations
        .iter()
        .find_map(|declaration| match declaration.kind {
            SemanticDeclarationKind::ConstitutiveLaw { value } if declaration.name == name => {
                Some(value)
            }
            _ => None,
        })
        .ok_or_else(|| {
            fv_error(
                "FV_AUTHORED_FLUX_MISSING",
                format!("model `{}` has no constitutive `{name}`", model.name),
            )
        })
}

// ---------------------------------------------------------------------------------------------
// Semantic expression lowering
// ---------------------------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) enum StateContext {
    /// A scalar law's state at one point.
    Point(OperandId),
    /// A facet pair; `minus`/`plus` hold one operand per state slot (one for a scalar law).
    Facet {
        minus: Vec<OperandId>,
        plus: Vec<OperandId>,
        side: Option<FacetSide>,
        normal: Vec<OperandId>,
    },
    /// SV2-C1b: a system's state vector at one point, with the face normal available to
    /// `normal_component` when `normal` is set.
    System {
        state: Vec<OperandId>,
        normal: Option<Vec<OperandId>>,
    },
    /// SV2-C1b: the inverse primitive map; `variables[i]` binds the i-th primitive symbol.
    Primitive {
        variables: Vec<(SymbolId, OperandId)>,
    },
}

impl StateContext {
    fn with_side(&self, new: FacetSide) -> Result<Self, ConservationLawError> {
        match self {
            Self::Facet {
                minus,
                plus,
                normal,
                ..
            } => Ok(Self::Facet {
                minus: minus.clone(),
                plus: plus.clone(),
                side: Some(new),
                normal: normal.clone(),
            }),
            Self::Point(_) | Self::System { .. } | Self::Primitive { .. } => Err(fv_error(
                "FV_FLUX_UNSUPPORTED",
                "facet traces are only meaningful in an authored numerical flux",
            )),
        }
    }

    fn normal(&self) -> Option<&[OperandId]> {
        match self {
            Self::Facet { normal, .. } => Some(normal),
            Self::System { normal, .. } => normal.as_deref(),
            Self::Point(_) | Self::Primitive { .. } => None,
        }
    }
}

#[derive(Default)]
pub(crate) struct DataTable {
    data: Vec<FvDatum>,
}

impl DataTable {
    pub(crate) fn index(&mut self, name: String, source: FvDatumSource) -> usize {
        if let Some(index) = self.data.iter().position(|datum| datum.source == source) {
            return index;
        }
        self.data.push(FvDatum { name, source });
        self.data.len() - 1
    }

    pub(crate) fn into_data(self) -> Vec<FvDatum> {
        self.data
    }
}

pub(crate) struct Lower<'a> {
    pub(crate) model: &'a SemanticModel,
    /// `(field, offset, components)` of every conserved field in state-vector order.
    pub(crate) components: Vec<(SymbolId, usize, usize)>,
    pub(crate) dimension: u8,
}

impl Lower<'_> {
    /// `(offset, components)` of a conserved field.
    fn state_slot(&self, symbol: SymbolId) -> Option<(usize, usize)> {
        self.components
            .iter()
            .find(|(field, _, _)| *field == symbol)
            .map(|(_, offset, count)| (*offset, *count))
    }

    /// `flux_sign · Σ F_i n_i`, left to right.
    fn flux_normal(
        &self,
        builder: &mut KernelBuilder,
        data: &mut DataTable,
        system: &ConservationLawSystem,
        context: &StateContext,
        normal: &[OperandId],
    ) -> Result<ScalarExpr, ConservationLawError> {
        let components = self.expr(builder, data, system.method.flux, context)?;
        let value = dot(components, normal);
        Ok(if system.flux_sign < 0 {
            ScalarExpr::unary(MUnary::Neg, value)
        } else {
            value
        })
    }

    pub(crate) fn expr(
        &self,
        builder: &mut KernelBuilder,
        data: &mut DataTable,
        id: ExprId,
        context: &StateContext,
    ) -> Result<Vec<ScalarExpr>, ConservationLawError> {
        let expression = &self.model.expressions[id.index()];
        let unsupported = |what: &str| {
            Err(fv_error(
                "FV_FLUX_UNSUPPORTED",
                format!("{what} (expression {id}) is not lowered by the scalar finite-volume path"),
            ))
        };
        match &expression.kind {
            SemanticExprKind::Number { value, .. } => Ok(vec![ScalarExpr::Constant(*value)]),
            SemanticExprKind::Symbol { symbol } => self.symbol(builder, data, *symbol, context),
            SemanticExprKind::Unary { arg, .. } => Ok(self
                .expr(builder, data, *arg, context)?
                .into_iter()
                .map(|value| ScalarExpr::unary(MUnary::Neg, value))
                .collect()),
            SemanticExprKind::Binary { op, lhs, rhs } => {
                let lhs = self.expr(builder, data, *lhs, context)?;
                let rhs = self.expr(builder, data, *rhs, context)?;
                let op = match op {
                    BinaryOp::Add => MBinary::Add,
                    BinaryOp::Sub => MBinary::Sub,
                    BinaryOp::Mul => MBinary::Mul,
                    BinaryOp::Div => MBinary::Div,
                    BinaryOp::Pow => MBinary::Pow,
                    _ => return unsupported("a comparison"),
                };
                broadcast(op, lhs, rhs).ok_or_else(|| {
                    fv_error(
                        "FV_FLUX_SHAPE",
                        format!("operand shapes of expression {id} do not combine componentwise"),
                    )
                })
            }
            SemanticExprKind::Vector { elements } => {
                // Scalars make a vector; equal-extent vectors make a row-major tensor
                // (SV2-C1b: `[[F00, F01], [F10, F11]]`, row = conserved component).
                let mut values = Vec::with_capacity(elements.len());
                let mut extent = None;
                for element in elements {
                    let element = self.expr(builder, data, *element, context)?;
                    if element.is_empty() || *extent.get_or_insert(element.len()) != element.len()
                    {
                        return unsupported("a ragged vector literal");
                    }
                    values.extend(element);
                }
                Ok(values)
            }
            SemanticExprKind::Index { value, indices } => {
                // SV2-C1b: literal indices into the row-major components of `value`.
                let components = self.expr(builder, data, *value, context)?;
                let extents = self.model.expressions[value.index()]
                    .ty
                    .axes
                    .iter()
                    .map(|axis| usize::from(axis.extent))
                    .collect::<Vec<_>>();
                if extents.is_empty()
                    || extents.len() < indices.len()
                    || extents.iter().product::<usize>() != components.len()
                {
                    return Err(fv_error(
                        "FV_FLUX_SHAPE",
                        format!("expression {id} indexes a value whose axes do not match its components"),
                    ));
                }
                let mut offset = 0;
                let mut stride = components.len();
                for (index, extent) in indices.iter().zip(&extents) {
                    let position = match self.model.expressions[index.index()].kind {
                        SemanticExprKind::Number { value, .. }
                            if value >= 0.0 && value.fract() == 0.0 && (value as usize) < *extent =>
                        {
                            value as usize
                        }
                        _ => return unsupported("a non-literal or out-of-range index"),
                    };
                    stride /= extent;
                    offset += position * stride;
                }
                Ok(components[offset..offset + stride].to_vec())
            }
            SemanticExprKind::Contraction {
                lhs,
                rhs,
                axes,
                conjugate_lhs: false,
            } if axes.len() == 1 => {
                let lhs = self.expr(builder, data, *lhs, context)?;
                let rhs = self.expr(builder, data, *rhs, context)?;
                if lhs.len() != rhs.len() || lhs.is_empty() {
                    return unsupported("a contraction of unequal vectors");
                }
                Ok(vec![
                    lhs.into_iter()
                        .zip(rhs)
                        .map(|(a, b)| ScalarExpr::binary(MBinary::Mul, a, b))
                        .reduce(add)
                        .expect("nonempty"),
                ])
            }
            SemanticExprKind::Call { function, args } => {
                if function == "time" && args.is_empty() {
                    let datum = data.index("t".into(), FvDatumSource::Time);
                    return Ok(vec![builder.datum(datum, "t")]);
                }
                let mut values = Vec::with_capacity(args.len());
                for arg in args {
                    let value = self.expr(builder, data, *arg, context)?;
                    match <[ScalarExpr; 1]>::try_from(value) {
                        Ok([value]) => values.push(value),
                        Err(_) => return unsupported("a vector argument of an intrinsic"),
                    }
                }
                let unary = |op| -> Result<Vec<ScalarExpr>, ConservationLawError> {
                    match <[ScalarExpr; 1]>::try_from(values.clone()) {
                        Ok([value]) => Ok(vec![ScalarExpr::unary(op, value)]),
                        Err(_) => unsupported("an intrinsic with the wrong arity"),
                    }
                };
                match function.as_str() {
                    "sqrt" => unary(MUnary::Sqrt),
                    "exp" => unary(MUnary::Exp),
                    "ln" | "log" => unary(MUnary::Ln),
                    "sin" => unary(MUnary::Sin),
                    "cos" => unary(MUnary::Cos),
                    "abs" => unary(MUnary::Abs),
                    "min" | "max" if values.len() >= 2 => {
                        let op = if function == "min" {
                            MBinary::Min
                        } else {
                            MBinary::Max
                        };
                        Ok(vec![
                            values
                                .into_iter()
                                .reduce(|a, b| ScalarExpr::binary(op, a, b))
                                .expect("two or more"),
                        ])
                    }
                    _ => unsupported(&format!("call `{function}`")),
                }
            }
            SemanticExprKind::ProviderCall { provider, args } => {
                if args.iter().any(|arg| self.depends_on_state(*arg)) {
                    return Err(fv_error(
                        "FV_FLUX_PROVIDER_UNSUPPORTED",
                        format!(
                            "provider call {id} depends on the conserved state; state-dependent provider products are not lowered into face kernels"
                        ),
                    ));
                }
                if !matches!(
                    expression.ty.shape,
                    SemanticShape::Numeric(ValueShape::Scalar)
                ) {
                    return unsupported("a non-scalar provider value");
                }
                let name = format!(
                    "{}@{}",
                    self.model.providers[provider.index()].name,
                    id.index()
                );
                let datum = data.index(
                    name.clone(),
                    FvDatumSource::ProviderCall {
                        expression: id,
                        provider: *provider,
                    },
                );
                Ok(vec![builder.datum(datum, &name)])
            }
            SemanticExprKind::FacetTrace { value, side } => {
                let side = match side {
                    TraceSide::Minus => FacetSide::Minus,
                    TraceSide::Plus => FacetSide::Plus,
                    TraceSide::Exterior => return unsupported("an unsided trace"),
                };
                self.expr(builder, data, *value, &context.with_side(side)?)
            }
            SemanticExprKind::Jump { value } => {
                let minus =
                    self.expr(builder, data, *value, &context.with_side(FacetSide::Minus)?)?;
                let plus =
                    self.expr(builder, data, *value, &context.with_side(FacetSide::Plus)?)?;
                Ok(minus
                    .into_iter()
                    .zip(plus)
                    .map(|(a, b)| ScalarExpr::binary(MBinary::Sub, a, b))
                    .collect())
            }
            SemanticExprKind::Average { value } => {
                let minus =
                    self.expr(builder, data, *value, &context.with_side(FacetSide::Minus)?)?;
                let plus =
                    self.expr(builder, data, *value, &context.with_side(FacetSide::Plus)?)?;
                Ok(minus
                    .into_iter()
                    .zip(plus)
                    .map(|(a, b)| half(add(a, b)))
                    .collect())
            }
            SemanticExprKind::NormalComponent { value, side } => {
                let Some(normal) = context.normal() else {
                    return unsupported("a normal component outside an authored flux or a declared face expression");
                };
                let normal = normal.to_vec();
                let (inner, negate) = match side {
                    TraceSide::Exterior => (context.clone(), false),
                    TraceSide::Minus => (context.with_side(FacetSide::Minus)?, false),
                    TraceSide::Plus => (context.with_side(FacetSide::Plus)?, true),
                };
                let components = self.expr(builder, data, *value, &inner)?;
                let dimension = usize::from(self.dimension);
                if components.is_empty() || components.len() % dimension != 0 {
                    return unsupported("a normal component of a vector of the wrong extent");
                }
                // A vector gives `G·n`; a row-major tensor gives one `row·n` per row (SV2-C1b).
                let mut rows = Vec::with_capacity(components.len() / dimension);
                let mut components = components.into_iter();
                while rows.len() < rows.capacity() {
                    let row = components.by_ref().take(dimension).collect::<Vec<_>>();
                    let value = dot(row, &normal);
                    rows.push(if negate {
                        ScalarExpr::unary(MUnary::Neg, value)
                    } else {
                        value
                    });
                }
                Ok(rows)
            }
            SemanticExprKind::Differential { operator, .. } => {
                if *operator == DifferentialOperator::Gradient {
                    Err(fv_error(
                        "FV_DIFFUSIVE_FLUX_UNSUPPORTED",
                        format!(
                            "expression {id} differentiates the state; diffusive fluxes need two-point gradients (SV2-C2b)"
                        ),
                    ))
                } else {
                    unsupported("a nested differential")
                }
            }
            _ => unsupported("this construct"),
        }
    }

    fn symbol(
        &self,
        builder: &mut KernelBuilder,
        data: &mut DataTable,
        symbol: SymbolId,
        context: &StateContext,
    ) -> Result<Vec<ScalarExpr>, ConservationLawError> {
        if let Some((offset, count)) = self.state_slot(symbol) {
            let loads = |operands: &[OperandId]| {
                operands[offset..offset + count]
                    .iter()
                    .map(|operand| ScalarExpr::Load(*operand))
                    .collect::<Vec<_>>()
            };
            return match context {
                StateContext::Point(state) => Ok(vec![ScalarExpr::Load(*state)]),
                StateContext::System { state, .. } => Ok(loads(state)),
                StateContext::Facet {
                    minus, plus, side, ..
                } => match side {
                    Some(FacetSide::Minus) => Ok(loads(minus)),
                    Some(FacetSide::Plus) => Ok(loads(plus)),
                    None => Err(fv_error(
                        "FV_AUTHORED_FLUX_UNTRACED",
                        "an authored flux reads the state outside trace_minus/trace_plus/jump/average",
                    )),
                },
                StateContext::Primitive { .. } => Err(fv_error(
                    "FV_PRIMITIVE_MAP_INVALID",
                    format!(
                        "the inverse primitive map reads the conserved field `{}`",
                        self.model.symbols[symbol.index()].name
                    ),
                )),
            };
        }
        if let StateContext::Primitive { variables } = context
            && let Some((_, operand)) = variables.iter().find(|(variable, _)| *variable == symbol)
        {
            return Ok(vec![ScalarExpr::Load(*operand)]);
        }
        if let Some(value) = crate::point_expression::symbol_definition(self.model, symbol) {
            return self.expr(builder, data, value, context);
        }
        let source = &self.model.symbols[symbol.index()];
        match source.ty.role {
            SemanticRole::Parameter | SemanticRole::Constant | SemanticRole::Source => {
                let components = match &source.ty.shape {
                    SemanticShape::Numeric(ValueShape::Scalar) => 1,
                    SemanticShape::Numeric(ValueShape::Vector(extent)) => *extent,
                    _ => {
                        return Err(fv_error(
                            "FV_FLUX_UNSUPPORTED",
                            format!("datum `{}` has no scalar or vector shape", source.name),
                        ));
                    }
                };
                Ok((0..components)
                    .map(|component| {
                        let name = if components == 1 {
                            source.name.clone()
                        } else {
                            format!("{}[{component}]", source.name)
                        };
                        let datum =
                            data.index(name.clone(), FvDatumSource::Symbol { symbol, component });
                        builder.datum(datum, &name)
                    })
                    .collect())
            }
            _ => Err(fv_error(
                "FV_FLUX_FIELD_UNSUPPORTED",
                format!(
                    "`{}` is neither the conserved state nor uniform model data",
                    source.name
                ),
            )),
        }
    }

    fn depends_on_state(&self, root: ExprId) -> bool {
        let mut stack = vec![root];
        let mut seen = BTreeSet::new();
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            let kind = &self.model.expressions[id.index()].kind;
            if let SemanticExprKind::Symbol { symbol } = kind {
                if self.state_slot(*symbol).is_some() {
                    return true;
                }
                if let Some(value) = crate::point_expression::symbol_definition(self.model, *symbol)
                {
                    stack.push(value);
                }
            }
            stack.extend(crate::formulation::expression_children(kind));
        }
        false
    }
}

fn broadcast(op: MBinary, lhs: Vec<ScalarExpr>, rhs: Vec<ScalarExpr>) -> Option<Vec<ScalarExpr>> {
    let combine = |a, b| ScalarExpr::binary(op, a, b);
    match (lhs.len(), rhs.len(), op) {
        (l, r, MBinary::Add | MBinary::Sub) if l == r => Some(
            lhs.into_iter()
                .zip(rhs)
                .map(|(a, b)| combine(a, b))
                .collect(),
        ),
        (1, _, MBinary::Mul) => {
            let scalar = lhs.into_iter().next()?;
            Some(
                rhs.into_iter()
                    .map(|b| combine(scalar.clone(), b))
                    .collect(),
            )
        }
        (_, 1, MBinary::Mul | MBinary::Div) => {
            let scalar = rhs.into_iter().next()?;
            Some(
                lhs.into_iter()
                    .map(|a| combine(a, scalar.clone()))
                    .collect(),
            )
        }
        (1, 1, MBinary::Pow) => Some(vec![combine(
            lhs.into_iter().next()?,
            rhs.into_iter().next()?,
        )]),
        _ => None,
    }
}

pub(crate) fn add(a: ScalarExpr, b: ScalarExpr) -> ScalarExpr {
    ScalarExpr::binary(MBinary::Add, a, b)
}

pub(crate) fn half(value: ScalarExpr) -> ScalarExpr {
    ScalarExpr::binary(MBinary::Mul, ScalarExpr::Constant(0.5), value)
}

pub(crate) fn dot(components: Vec<ScalarExpr>, normal: &[OperandId]) -> ScalarExpr {
    components
        .into_iter()
        .zip(normal)
        .map(|(component, axis)| {
            ScalarExpr::binary(MBinary::Mul, component, ScalarExpr::Load(*axis))
        })
        .reduce(add)
        .expect("a spatial domain has at least one axis")
}

pub(crate) fn compare(op: malleus::CompareOp, lhs: ScalarExpr, rhs: ScalarExpr) -> Predicate {
    Predicate::Compare {
        op,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

pub(crate) fn select(condition: Predicate, if_true: ScalarExpr, if_false: ScalarExpr) -> ScalarExpr {
    ScalarExpr::Select {
        condition: Box::new(condition),
        if_true: Box::new(if_true),
        if_false: Box::new(if_false),
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel construction
// ---------------------------------------------------------------------------------------------

pub(crate) struct KernelBuilder {
    name: String,
    operands: Vec<KernelOperand>,
    pub(crate) meanings: Vec<FvOperandMeaning>,
    statements: Vec<Statement>,
    locals: usize,
}

impl KernelBuilder {
    pub(crate) fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            operands: Vec::new(),
            meanings: Vec::new(),
            statements: Vec::new(),
            locals: 0,
        }
    }

    pub(crate) fn operand(&mut self, name: &str, meaning: FvOperandMeaning) -> OperandId {
        if let Some(index) = self.meanings.iter().position(|item| *item == meaning) {
            return OperandId::new(index);
        }
        self.operands
            .push(KernelOperand::scalar(name, AccessMode::Read));
        self.meanings.push(meaning);
        OperandId::new(self.operands.len() - 1)
    }

    pub(crate) fn normals(&mut self, dimension: u8) -> Vec<OperandId> {
        (0..dimension)
            .map(|axis| self.operand(&format!("n{axis}"), FvOperandMeaning::Normal { axis }))
            .collect()
    }

    pub(crate) fn datum(&mut self, datum: usize, name: &str) -> ScalarExpr {
        ScalarExpr::Load(self.operand(name, FvOperandMeaning::Datum { datum }))
    }

    pub(crate) fn bind(&mut self, value: ScalarExpr) -> ScalarExpr {
        let local = LocalId::new(self.locals);
        self.locals += 1;
        self.statements.push(Statement::Let { local, value });
        ScalarExpr::Local(local)
    }

    fn facet_roles(&self, minus: OperandId, plus: OperandId) -> Vec<FacetOperandRole> {
        self.meanings
            .iter()
            .map(|meaning| match meaning {
                FvOperandMeaning::State {
                    side: Some(FacetSide::Minus),
                } => FacetOperandRole::Cell {
                    side: FacetSide::Minus,
                    partner: Some(plus),
                },
                FvOperandMeaning::State {
                    side: Some(FacetSide::Plus),
                } => FacetOperandRole::Cell {
                    side: FacetSide::Plus,
                    partner: Some(minus),
                },
                FvOperandMeaning::Normal { .. } => FacetOperandRole::Facet {
                    parity: SwapParity::Odd,
                },
                _ => FacetOperandRole::Facet {
                    parity: SwapParity::Even,
                },
            })
            .chain([FacetOperandRole::Facet {
                parity: SwapParity::Odd,
            }])
            .collect()
    }

    pub(crate) fn finish(
        self,
        role: FvKernelRole,
        value: ScalarExpr,
        facet_roles: Option<Vec<FacetOperandRole>>,
    ) -> Result<FvKernel, ConservationLawError> {
        self.finish_outputs(
            role,
            vec![("output".to_string(), FvOperandMeaning::Output, value)],
            facet_roles,
        )
    }

    /// SV2-C1b: one output operand per entry, in order, appended after the inputs. `facet_roles`
    /// covers every operand, outputs included.
    pub(crate) fn finish_outputs(
        mut self,
        role: FvKernelRole,
        outputs: Vec<(String, FvOperandMeaning, ScalarExpr)>,
        facet_roles: Option<Vec<FacetOperandRole>>,
    ) -> Result<FvKernel, ConservationLawError> {
        for (name, meaning, value) in outputs {
            let output = OperandId::new(self.operands.len());
            self.operands
                .push(KernelOperand::scalar(&name, AccessMode::Write));
            self.meanings.push(meaning);
            self.statements.push(Statement::Store {
                operand: output,
                value,
            });
        }
        let kernel = StructuredKernel {
            name: self.name,
            iteration_domain: IterationDomain::default(),
            iterators: Vec::new(),
            indexing_maps: (0..self.operands.len())
                .map(|index| IndexingMap::scalar(OperandId::new(index)))
                .collect(),
            operands: self.operands,
            body: KernelRegion {
                statements: self.statements,
            },
            numeric_policy: NumericPolicy::default(),
        };
        validate(kernel.clone())
            .map_err(|error| fv_error("FV_KERNEL_INVALID", format!("{error:?}")))?;
        if let Some(roles) = &facet_roles {
            validate_facet_pair(FacetPairKernel {
                kernel: kernel.clone(),
                roles: roles.clone(),
            })
            .map_err(|error| fv_error("FV_KERNEL_INVALID", error.to_string()))?;
        }
        let nonsmooth = nonsmooth_sites(&kernel);
        let digest = span_independent_digest(&KernelPayload {
            schema: FV_KERNEL_SCHEMA,
            role,
            kernel: &kernel,
            operands: &self.meanings,
            facet_roles: &facet_roles,
        });
        Ok(FvKernel {
            role,
            kernel,
            operands: self.meanings,
            facet_roles,
            nonsmooth,
            digest,
        })
    }
}

#[derive(Serialize)]
struct KernelPayload<'a> {
    schema: &'static str,
    role: FvKernelRole,
    kernel: &'a StructuredKernel,
    operands: &'a [FvOperandMeaning],
    facet_roles: &'a Option<Vec<FacetOperandRole>>,
}

/// `F(u)·n` and its Malleus JVP, inlined into composite kernels with substituted operands.
struct Derived<'a> {
    physical: &'a FvKernel,
    tangent: &'a DerivativeProduct,
}

impl Derived<'_> {
    fn inline_primal(
        &self,
        builder: &mut KernelBuilder,
        state: ScalarExpr,
        normal: &[OperandId],
    ) -> Result<ScalarExpr, ConservationLawError> {
        let map = |operand: OperandId| self.physical.operands[operand.index()].clone();
        let output = self
            .physical
            .operand(&FvOperandMeaning::Output)
            .expect("physical output");
        inline(builder, &self.physical.kernel, output, |operand| {
            substitute(&map(operand), &state, normal)
        })
    }

    fn inline_tangent(
        &self,
        builder: &mut KernelBuilder,
        state: ScalarExpr,
        normal: &[OperandId],
    ) -> Result<ScalarExpr, ConservationLawError> {
        let primal = self
            .tangent
            .primal_operands
            .iter()
            .map(|pair| (pair.derivative, pair.primal))
            .collect::<BTreeMap<_, _>>();
        let direction = self.tangent.independent_operands[0].derivative;
        let output = self.tangent.dependent_operands[0].derivative;
        inline(builder, &self.tangent.kernel, output, |operand| {
            if operand == direction {
                return Ok(Substitution::Value(ScalarExpr::Constant(1.0)));
            }
            let primal = primal.get(&operand).ok_or_else(|| {
                fv_error(
                    "FV_DERIVATIVE_FAILED",
                    format!("tangent operand {operand:?} has no primal binding"),
                )
            })?;
            substitute(&self.physical.operands[primal.index()], &state, normal)
        })
    }
}

/// How an inlined operand is bound: an expression of the target kernel, or a datum operand the
/// target kernel adds (or reuses) by meaning.
enum Substitution {
    Value(ScalarExpr),
    Datum(FvOperandMeaning),
}

fn substitute(
    meaning: &FvOperandMeaning,
    state: &ScalarExpr,
    normal: &[OperandId],
) -> Result<Substitution, ConservationLawError> {
    match meaning {
        FvOperandMeaning::State { .. } => Ok(Substitution::Value(state.clone())),
        FvOperandMeaning::Normal { axis } => Ok(Substitution::Value(ScalarExpr::Load(
            normal[usize::from(*axis)],
        ))),
        FvOperandMeaning::Datum { .. } => Ok(Substitution::Datum(meaning.clone())),
        other => Err(fv_error(
            "FV_KERNEL_INVALID",
            format!("cannot inline operand {other:?}"),
        )),
    }
}

/// Copy `source`'s statements into `builder`, mapping loads through `resolve` and renumbering
/// locals; the store to `output` becomes a fresh local, which is returned.
fn inline(
    builder: &mut KernelBuilder,
    source: &StructuredKernel,
    output: OperandId,
    mut resolve: impl FnMut(OperandId) -> Result<Substitution, ConservationLawError>,
) -> Result<ScalarExpr, ConservationLawError> {
    let mut loads = BTreeMap::new();
    for index in 0..source.operands.len() {
        let operand = OperandId::new(index);
        if !source.operands[index].access.can_read() {
            continue;
        }
        let value = match resolve(operand)? {
            Substitution::Value(value) => value,
            Substitution::Datum(meaning) => {
                let name = source.operands[index].name.clone();
                ScalarExpr::Load(builder.operand(&name, meaning))
            }
        };
        loads.insert(operand, value);
    }
    let mut locals = BTreeMap::new();
    let mut result = None;
    for statement in &source.body.statements {
        match statement {
            Statement::Let { local, value } => {
                let value = remap(value, &loads, &locals)?;
                let ScalarExpr::Local(fresh) = builder.bind(value) else {
                    unreachable!("bind returns a local");
                };
                locals.insert(*local, fresh);
            }
            Statement::Store { operand, value } if *operand == output => {
                result = Some(builder.bind(remap(value, &loads, &locals)?));
            }
            Statement::Store { .. } => {}
        }
    }
    result.ok_or_else(|| fv_error("FV_KERNEL_INVALID", "inlined kernel stores no output"))
}

fn remap(
    value: &ScalarExpr,
    loads: &BTreeMap<OperandId, ScalarExpr>,
    locals: &BTreeMap<LocalId, LocalId>,
) -> Result<ScalarExpr, ConservationLawError> {
    let missing = || fv_error("FV_KERNEL_INVALID", "inlined kernel reads an unbound value");
    Ok(match value {
        ScalarExpr::Constant(value) => ScalarExpr::Constant(*value),
        ScalarExpr::Index(_) => return Err(missing()),
        ScalarExpr::Load(operand) => loads.get(operand).cloned().ok_or_else(missing)?,
        ScalarExpr::Local(local) => ScalarExpr::Local(*locals.get(local).ok_or_else(missing)?),
        ScalarExpr::Unary { op, arg } => ScalarExpr::unary(*op, remap(arg, loads, locals)?),
        ScalarExpr::Binary { op, lhs, rhs } => {
            ScalarExpr::binary(*op, remap(lhs, loads, locals)?, remap(rhs, loads, locals)?)
        }
        ScalarExpr::Select {
            condition,
            if_true,
            if_false,
        } => select(
            remap_predicate(condition, loads, locals)?,
            remap(if_true, loads, locals)?,
            remap(if_false, loads, locals)?,
        ),
        ScalarExpr::Fma { lhs, rhs, addend } => ScalarExpr::fma(
            remap(lhs, loads, locals)?,
            remap(rhs, loads, locals)?,
            remap(addend, loads, locals)?,
        ),
    })
}

fn remap_predicate(
    predicate: &Predicate,
    loads: &BTreeMap<OperandId, ScalarExpr>,
    locals: &BTreeMap<LocalId, LocalId>,
) -> Result<Predicate, ConservationLawError> {
    Ok(match predicate {
        Predicate::Constant(value) => Predicate::Constant(*value),
        Predicate::Compare { op, lhs, rhs } => {
            compare(*op, remap(lhs, loads, locals)?, remap(rhs, loads, locals)?)
        }
        Predicate::Not(inner) => Predicate::Not(Box::new(remap_predicate(inner, loads, locals)?)),
        Predicate::And(a, b) => Predicate::And(
            Box::new(remap_predicate(a, loads, locals)?),
            Box::new(remap_predicate(b, loads, locals)?),
        ),
        Predicate::Or(a, b) => Predicate::Or(
            Box::new(remap_predicate(a, loads, locals)?),
            Box::new(remap_predicate(b, loads, locals)?),
        ),
    })
}

// ---------------------------------------------------------------------------------------------
// Nonsmooth analysis (D6)
// ---------------------------------------------------------------------------------------------

/// Every kink of `kernel` with the operands its switch depends on (through locals).
pub fn nonsmooth_sites(kernel: &StructuredKernel) -> Vec<NonsmoothSite> {
    let mut local_deps: BTreeMap<LocalId, BTreeSet<OperandId>> = BTreeMap::new();
    let mut sites = Vec::new();
    for (index, statement) in kernel.body.statements.iter().enumerate() {
        let (value, local) = match statement {
            Statement::Let { local, value } => (value, Some(*local)),
            Statement::Store { value, .. } => (value, None),
        };
        let deps = expr_sites(value, index, &local_deps, &mut sites);
        if let Some(local) = local {
            local_deps.insert(local, deps);
        }
    }
    sites
}

fn expr_sites(
    value: &ScalarExpr,
    statement: usize,
    locals: &BTreeMap<LocalId, BTreeSet<OperandId>>,
    sites: &mut Vec<NonsmoothSite>,
) -> BTreeSet<OperandId> {
    let site = |kind, deps: &BTreeSet<OperandId>, sites: &mut Vec<NonsmoothSite>| {
        sites.push(NonsmoothSite {
            statement,
            kind,
            switch_operands: deps.iter().copied().collect(),
        });
    };
    match value {
        ScalarExpr::Constant(_) | ScalarExpr::Index(_) => BTreeSet::new(),
        ScalarExpr::Load(operand) => BTreeSet::from([*operand]),
        ScalarExpr::Local(local) => locals.get(local).cloned().unwrap_or_default(),
        ScalarExpr::Unary { op, arg } => {
            let deps = expr_sites(arg, statement, locals, sites);
            match op {
                MUnary::Abs => site(NonsmoothKind::Abs, &deps, sites),
                MUnary::Floor => site(NonsmoothKind::Floor, &deps, sites),
                MUnary::Ceil => site(NonsmoothKind::Ceil, &deps, sites),
                _ => {}
            }
            deps
        }
        ScalarExpr::Binary { op, lhs, rhs } => {
            let mut deps = expr_sites(lhs, statement, locals, sites);
            deps.extend(expr_sites(rhs, statement, locals, sites));
            match op {
                MBinary::Min => site(NonsmoothKind::Min, &deps, sites),
                MBinary::Max => site(NonsmoothKind::Max, &deps, sites),
                _ => {}
            }
            deps
        }
        ScalarExpr::Select {
            condition,
            if_true,
            if_false,
        } => {
            // A select differentiates its taken branch (D6 clarification); it is no kink.
            let mut deps = predicate_sites(condition, statement, locals, sites);
            deps.extend(expr_sites(if_true, statement, locals, sites));
            deps.extend(expr_sites(if_false, statement, locals, sites));
            deps
        }
        ScalarExpr::Fma { lhs, rhs, addend } => {
            let mut deps = expr_sites(lhs, statement, locals, sites);
            deps.extend(expr_sites(rhs, statement, locals, sites));
            deps.extend(expr_sites(addend, statement, locals, sites));
            deps
        }
    }
}

fn predicate_sites(
    predicate: &Predicate,
    statement: usize,
    locals: &BTreeMap<LocalId, BTreeSet<OperandId>>,
    sites: &mut Vec<NonsmoothSite>,
) -> BTreeSet<OperandId> {
    match predicate {
        Predicate::Constant(_) => BTreeSet::new(),
        Predicate::Compare { lhs, rhs, .. } => {
            let mut deps = expr_sites(lhs, statement, locals, sites);
            deps.extend(expr_sites(rhs, statement, locals, sites));
            deps
        }
        Predicate::Not(inner) => predicate_sites(inner, statement, locals, sites),
        Predicate::And(a, b) | Predicate::Or(a, b) => {
            let mut deps = predicate_sites(a, statement, locals, sites);
            deps.extend(predicate_sites(b, statement, locals, sites));
            deps
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Claim checks
// ---------------------------------------------------------------------------------------------

fn sample_buffers(
    kernel: &FvKernel,
    data: &[FvDatum],
    sample: &FvFluxSample,
    dimension: u8,
    state: impl Fn(Option<FacetSide>) -> f64,
) -> Result<Vec<Vec<f64>>, ConservationLawError> {
    if sample.normal.len() != usize::from(dimension) {
        return Err(fv_error(
            "FV_SAMPLE_INVALID",
            format!("sample normal has {} components", sample.normal.len()),
        ));
    }
    kernel
        .operands
        .iter()
        .map(|meaning| {
            Ok(vec![match meaning {
                FvOperandMeaning::State { side } => state(*side),
                FvOperandMeaning::Normal { axis } => sample.normal[usize::from(*axis)],
                FvOperandMeaning::Datum { datum } => {
                    let name = &data[*datum].name;
                    *sample.data.get(name).ok_or_else(|| {
                        fv_error("FV_SAMPLE_INVALID", format!("sample lacks datum `{name}`"))
                    })?
                }
                FvOperandMeaning::SlopeRatio
                | FvOperandMeaning::Output
                | FvOperandMeaning::StateComponent { .. }
                | FvOperandMeaning::PrimitiveComponent { .. }
                | FvOperandMeaning::OutputComponent { .. } => 0.0,
            }])
        })
        .collect()
}

/// Run a point kernel on one-element buffers and return its output.
pub fn run_point(kernel: &FvKernel, buffers: &mut [Vec<f64>]) -> Result<f64, ConservationLawError> {
    let validated = validate(kernel.kernel.clone())
        .map_err(|error| fv_error("FV_KERNEL_INVALID", format!("{error:?}")))?;
    let executable = Executable::reference(validated);
    let mut bindings = buffers
        .iter_mut()
        .enumerate()
        .map(|(index, values)| BufferBinding::new(OperandId::new(index), values))
        .collect::<Vec<_>>();
    Interpreter::run(&executable, &mut bindings)
        .map_err(|error| fv_error("FV_KERNEL_INVALID", format!("{error:?}")))?;
    let output = kernel
        .operand(&FvOperandMeaning::Output)
        .expect("kernel output");
    Ok(buffers[output.index()][0])
}

fn check_consistency(
    face: &FvKernel,
    physical: &FvKernel,
    data: &[FvDatum],
    samples: &[FvFluxSample],
    dimension: u8,
) -> Result<f64, ConservationLawError> {
    let mut worst: f64 = 0.0;
    for sample in samples {
        for value in [sample.minus, sample.plus] {
            let mut face_buffers = sample_buffers(face, data, sample, dimension, |_| value)?;
            let mut physical_buffers =
                sample_buffers(physical, data, sample, dimension, |_| value)?;
            let numerical = run_point(face, &mut face_buffers)?;
            let exact = run_point(physical, &mut physical_buffers)?;
            let deviation = (numerical - exact).abs();
            if !deviation.is_finite() {
                return Ok(f64::INFINITY);
            }
            worst = worst.max(deviation);
        }
    }
    Ok(worst)
}

fn check_antisymmetry(
    face: &FvKernel,
    data: &[FvDatum],
    samples: &[FvFluxSample],
    dimension: u8,
) -> Result<f64, ConservationLawError> {
    let pair = face.facet_pair().expect("face flux is a facet pair");
    let validated = validate_facet_pair(pair)
        .map_err(|error| fv_error("FV_KERNEL_INVALID", error.to_string()))?;
    let mut worst: f64 = 0.0;
    for sample in samples {
        let buffers = sample_buffers(face, data, sample, dimension, |side| match side {
            Some(FacetSide::Plus) => sample.plus,
            _ => sample.minus,
        })?;
        let report = check_facet_swap_symmetry(&validated, &buffers, f64::MAX)
            .map_err(|error| fv_error("FV_SAMPLE_INVALID", error.to_string()))?;
        if !report.max_absolute.is_finite() {
            return Ok(f64::INFINITY);
        }
        worst = worst.max(report.max_absolute);
    }
    Ok(worst)
}

// ---------------------------------------------------------------------------------------------
// Obligations
// ---------------------------------------------------------------------------------------------

fn obligations(
    model: &SemanticModel,
    system: &ConservationLawSystem,
    intent: &FvMethodIntent,
    checks: &[FluxClaimCheck],
) -> Vec<VerificationObligation> {
    let parent = &system.source_semantic_digest.hex;
    let equation = model
        .declarations
        .iter()
        .find(|declaration| declaration.id == system.method.equation)
        .expect("selected equation");
    let field = system.method.conserved;
    let family = intent.numerical_flux.label();
    let authored = matches!(intent.numerical_flux, NumericalFluxIntent::Authored { .. });
    let checked = |claim| checks.iter().find(|check| check.claim == claim);
    let make =
        |meaning: String, kind, applicability: Vec<&str>, expected: &str, tolerance, evidence| {
            make_obligation(
                parent,
                &model.name,
                equation.span,
                meaning,
                kind,
                applicability
                    .into_iter()
                    .map(|statement| ValidityCondition {
                        statement: statement.into(),
                    })
                    .collect(),
                vec![],
                expected,
                tolerance,
                evidence,
                None,
            )
        };
    let claim_evidence = |claim| {
        if authored || checked(claim).is_some() {
            VerificationEvidenceClass::Numerical
        } else {
            VerificationEvidenceClass::Semantic
        }
    };
    let mut out = vec![
        make(
            format!(
                "{family} numerical flux is consistent with F(u)·n for {}",
                equation_name(equation)
            ),
            VerificationObligationKind::NumericalFluxConsistency {
                equation: equation.id,
            },
            vec![if authored {
                "authored flux: checked by Scientia at the caller's samples"
            } else {
                "generic family: holds by construction, checked at samples when given"
            }],
            "F*(u,u,n) = F(u)·n",
            if authored {
                ToleranceClass::Roundoff
            } else {
                ToleranceClass::Exact
            },
            claim_evidence(FluxClaim::Consistency),
        ),
        make(
            format!("{family} numerical flux is antisymmetric under side exchange"),
            VerificationObligationKind::NumericalFluxAntisymmetry {
                equation: equation.id,
            },
            vec!["checked by malleus::check_facet_swap_symmetry with n Odd and F* Odd"],
            "F*(u-,u+,n) = -F*(u+,u-,-n)",
            if authored {
                ToleranceClass::Roundoff
            } else {
                ToleranceClass::Exact
            },
            claim_evidence(FluxClaim::Antisymmetry),
        ),
        make(
            format!("discrete flux balance of {}", equation_name(equation)),
            VerificationObligationKind::Conservation {
                quantity: system.method.time_derivative,
                relation: ConservationRelation::DiscreteFluxBalance,
            },
            vec!["realized per step by Finitum; interior face fluxes cancel pairwise"],
            "d/dt sum_i |V_i| u_i = -sum_boundary |f| F*_f + sum_i |V_i| S_i",
            ToleranceClass::Roundoff,
            VerificationEvidenceClass::Numerical,
        ),
    ];
    let muscl = matches!(intent.reconstruction, ReconstructionIntent::Muscl { .. });
    let forward_euler_limit = if muscl { 0.5 } else { 1.0 };
    out.push(make(
        "explicit step obeys the CFL bound of the lowered scheme".into(),
        VerificationObligationKind::CflBound {
            field,
            forward_euler_limit,
        },
        vec![
            if matches!(intent.numerical_flux, NumericalFluxIntent::Central) {
                "central flux is not forward-Euler stable; for smooth problems and testing only"
            } else if muscl {
                "MUSCL limit 1/2 on 1-D Cartesian cells, times the integrator SSP coefficient"
            } else {
                "first-order monotone flux, times the integrator SSP coefficient"
            },
        ],
        "dt <= C min_i |V_i| / sum_{f in i} |f| lambda_f",
        ToleranceClass::Exact,
        VerificationEvidenceClass::Numerical,
    ));
    let monotone = matches!(
        intent.numerical_flux,
        NumericalFluxIntent::LocalLaxFriedrichs | NumericalFluxIntent::Hll
    );
    let tvd_limited = match &intent.reconstruction {
        ReconstructionIntent::Constant => true,
        ReconstructionIntent::Muscl { limiter, .. } => *limiter != LimiterFamily::None,
    };
    if monotone && tvd_limited {
        out.push(make(
            "total variation does not increase under the CFL bound".into(),
            VerificationObligationKind::TotalVariationDiminishing { field },
            vec!["Harten/Sweby: monotone flux and a limiter inside the TVD region"],
            "TV(u^{n+1}) <= TV(u^n)",
            ToleranceClass::Roundoff,
            VerificationEvidenceClass::Numerical,
        ));
    }
    out
}

fn equation_name(equation: &SemanticDeclaration) -> &str {
    &equation.name
}
