//! Diffusive finite-volume terms and multidimensional limiter kernels (SV2-C1c).
//!
//! A `.res` balance `dt(u) + div(F(u)) - div(k * grad(u)) = S` is read as a hyperbolic part `F`
//! (lowered by [`crate::conservation`]) plus one *diffusive term* whose coefficient `k` is a
//! `parameter`, a `property` (bound per cell by the realizer, e.g. per material region) or a
//! `constitutive` of `u`. Either part may be absent: a pure diffusion law lowers a zero
//! hyperbolic flux, a pure hyperbolic law has no [`DiffusiveTerm`] (and its W12 artifacts are
//! byte-identical, the new fields being omitted when absent).
//!
//! The diffusive normal flux leaving the minus cell is `-k_f · (grad u)·n`. The face-normal
//! gradient is an operand the realizer supplies ([`crate::FvOperandMeaning::FaceNormalGradient`]:
//! the two-point `(u⁺ − u⁻)/(δ·n)` on orthogonal meshes, `δ` the centre-to-centre vector; at a
//! `dirichlet` boundary `u⁺` is the ghost state and `δ·n` the centre-to-face distance). `k` is
//! evaluated on both sides and combined by the declared [`FaceAveraging`] (harmonic by default,
//! exact for piecewise-constant conductivity); a coefficient that reads neither the state nor
//! per-cell data is used directly, so the kernel is bitwise `-k·g` for constant `k`.
//!
//! Claims checked at caller samples ([`check_diffusive_claims`]): consistency with `-k(u)·g`
//! (exact for uniform `k`; otherwise within `8ε|k g|`, the harmonic mean of two equal values
//! costing three roundings), antisymmetry under side exchange (exact, through
//! `malleus::check_facet_swap_symmetry`), and dissipativity `F_d · (u⁺ − u⁻) ≤ 0` for `k > 0`.
//! Decision D6 applies unchanged: a smooth `k(u)` keeps its JVP, an `abs`/`min`/`max` in `k`
//! makes every state derivative of the face kernel a typed refusal.
//!
//! Multidimensional limiters ([`LimiterFamily::BarthJespersen`],
//! [`LimiterFamily::Venkatakrishnan`]) lower to the cell-face limiter function `φ(Δ⁺, Δ⁻, h)`
//! the realizer applies per (cell, face) and minimizes over the cell's faces: `Δ⁻` is the
//! unlimited reconstructed increment `∇u_i · (x_f − x_i)`, `Δ⁺` the admissible increment
//! `u_max − u_i` when `Δ⁻ > 0` and `u_min − u_i` when `Δ⁻ < 0` (neighbour extremes), `h` the
//! cell size. Barth–Jespersen is `min(1, Δ⁺/Δ⁻)` (`1` when `Δ⁻ = 0`) and refuses derivatives
//! (D6); Venkatakrishnan is the smooth rational form with `ε² = (K h)³` and admits them.
//!
//! Out of scope, refused or recorded: anisotropic/tensor conductivity
//! (`FV_DIFFUSION_ANISOTROPIC`), non-orthogonal corrections, cross-diffusion between components,
//! limiters on characteristic variables.
use crate::conservation::{
    ConservationLawError, ConservationLawProgram, ConservationLawSystem, DataTable, FvKernel,
    FvKernelRole, FvOperandMeaning, KernelBuilder, LimiterFamily, Lower, StateContext, add,
    compare, fv_error, half, run_point, select,
};
use crate::verification::{
    ToleranceClass, ValidityCondition, VerificationEvidenceClass, VerificationObligation,
    VerificationObligationKind, make_obligation,
};
use crate::{
    DifferentialOperator, ExprId, SemanticExprKind, SemanticModel, SemanticRole, SemanticShape,
    SymbolId, scientific::BinaryOp, scientific::ValueShape,
};
use malleus::{
    BinaryOp as MBinary, CompareOp, FacetSide, ScalarExpr, UnaryOp as MUnary,
    check_facet_swap_symmetry, validate_facet_pair,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------------------------
// Law side
// ---------------------------------------------------------------------------------------------

/// What the diffusion coefficient `k` is in the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffusionCoefficientKind {
    /// `div(grad(u))`: unit conductivity.
    Unit,
    /// A valueless `parameter` or `constant` symbol: uniform data.
    Parameter,
    /// A `property` symbol (its value is a provider product or an expression).
    Property,
    /// A `constitutive` symbol, typically `k(u)`.
    Constitutive,
    /// Any other scalar expression written in place.
    Expression,
}

/// The diffusive divergence `div(k * grad(u))` of a law, normalized to enter as
/// `dt(u) + ... - div(k grad u)`; an anti-diffusive sign is refused when the law is read.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusiveTerm {
    /// The `div(...)` expression. When it equals `method.flux_divergence`, the law has no
    /// hyperbolic part ([`ConservationLawSystem::hyperbolic_flux`] is `None`).
    pub divergence: ExprId,
    /// The `grad(u)` expression.
    pub gradient: ExprId,
    /// The coefficient expression; `None` is unit conductivity.
    pub coefficient: Option<ExprId>,
    pub coefficient_kind: DiffusionCoefficientKind,
    /// `k` reads the conserved state (`k(u)`).
    pub state_dependent: bool,
}

/// One top-level divergence term of the equation, classified.
pub(crate) enum DivergenceTerm {
    Diffusive(DiffusiveTerm),
    Hyperbolic { flux: ExprId },
}

/// Classify an additive term: `Some(Diffusive)` for `div(k * grad(u))`, `div(grad(u) * k)`,
/// `div(grad(u))` and their negations (through a single defining symbol), `Some(Hyperbolic)`
/// for every other `div(...)`, `None` for a non-divergence term. The returned sign factor is
/// `-1` when the argument was negated an odd number of times.
pub(crate) fn classify_divergence(
    model: &SemanticModel,
    conserved: SymbolId,
    expression: ExprId,
) -> Result<Option<(DivergenceTerm, i8)>, ConservationLawError> {
    let SemanticExprKind::Differential {
        operator: DifferentialOperator::Divergence,
        arg,
    } = model.expressions[expression.index()].kind
    else {
        return Ok(None);
    };
    let mut inner = arg;
    let mut sign = 1;
    // Follow one defining symbol (`constitutive q = -k * grad(u); ... div(q)`) and negations.
    loop {
        match &model.expressions[inner.index()].kind {
            SemanticExprKind::Unary { arg, .. } => {
                sign = -sign;
                inner = *arg;
            }
            SemanticExprKind::Symbol { symbol } if *symbol != conserved => {
                match crate::point_expression::symbol_definition(model, *symbol) {
                    Some(value) if contains_gradient_of(model, value, conserved) => inner = value,
                    _ => break,
                }
            }
            _ => break,
        }
    }
    let is_gradient = |id: ExprId| {
        matches!(
            model.expressions[id.index()].kind,
            SemanticExprKind::Differential {
                operator: DifferentialOperator::Gradient,
                arg,
            } if matches!(model.expressions[arg.index()].kind, SemanticExprKind::Symbol { symbol } if symbol == conserved)
        )
    };
    let (gradient, coefficient) = match &model.expressions[inner.index()].kind {
        _ if is_gradient(inner) => (inner, None),
        SemanticExprKind::Binary {
            op: BinaryOp::Mul,
            lhs,
            rhs,
        } if is_gradient(*rhs) => (*rhs, Some(*lhs)),
        SemanticExprKind::Binary {
            op: BinaryOp::Mul,
            lhs,
            rhs,
        } if is_gradient(*lhs) => (*lhs, Some(*rhs)),
        _ => {
            return Ok(Some((DivergenceTerm::Hyperbolic { flux: arg }, 1)));
        }
    };
    let (coefficient_kind, state_dependent) = match coefficient {
        None => (DiffusionCoefficientKind::Unit, false),
        Some(coefficient) => {
            let expression = &model.expressions[coefficient.index()];
            if !matches!(
                expression.ty.shape,
                SemanticShape::Numeric(ValueShape::Scalar)
            ) {
                return Err(fv_error(
                    "FV_DIFFUSION_ANISOTROPIC",
                    format!(
                        "diffusion coefficient {coefficient} is not scalar; tensor conductivity is not lowered (SV2-C1c scope)"
                    ),
                ));
            }
            if contains_differential(model, coefficient) {
                return Err(fv_error(
                    "FV_DIFFUSIVE_FLUX_UNSUPPORTED",
                    format!("diffusion coefficient {coefficient} holds a differential operator"),
                ));
            }
            let kind = match &expression.kind {
                SemanticExprKind::Symbol { symbol } => match model.symbols[symbol.index()].ty.role {
                    SemanticRole::Parameter | SemanticRole::Constant => {
                        DiffusionCoefficientKind::Parameter
                    }
                    SemanticRole::Property => DiffusionCoefficientKind::Property,
                    SemanticRole::ConstitutiveLaw => DiffusionCoefficientKind::Constitutive,
                    _ => DiffusionCoefficientKind::Expression,
                },
                _ => DiffusionCoefficientKind::Expression,
            };
            (kind, depends_on(model, coefficient, conserved))
        }
    };
    Ok(Some((
        DivergenceTerm::Diffusive(DiffusiveTerm {
            divergence: expression,
            gradient,
            coefficient,
            coefficient_kind,
            state_dependent,
        }),
        sign,
    )))
}

fn walk(model: &SemanticModel, root: ExprId, mut visit: impl FnMut(&SemanticExprKind) -> bool) -> bool {
    let mut stack = vec![root];
    let mut seen = BTreeSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let kind = &model.expressions[id.index()].kind;
        if visit(kind) {
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

fn depends_on(model: &SemanticModel, root: ExprId, symbol: SymbolId) -> bool {
    walk(model, root, |kind| {
        matches!(kind, SemanticExprKind::Symbol { symbol: found } if *found == symbol)
    })
}

fn contains_differential(model: &SemanticModel, root: ExprId) -> bool {
    walk(model, root, |kind| {
        matches!(kind, SemanticExprKind::Differential { .. })
    })
}

fn contains_gradient_of(model: &SemanticModel, root: ExprId, conserved: SymbolId) -> bool {
    walk(model, root, |kind| {
        matches!(
            kind,
            SemanticExprKind::Differential {
                operator: DifferentialOperator::Gradient,
                arg,
            } if matches!(model.expressions[arg.index()].kind, SemanticExprKind::Symbol { symbol } if symbol == conserved)
        )
    })
}

// ---------------------------------------------------------------------------------------------
// Intent
// ---------------------------------------------------------------------------------------------

/// How `k⁻` and `k⁺` combine into the face conductivity `k_f`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FaceAveraging {
    /// `2 k⁻ k⁺ / (k⁻ + k⁺)` (`0` when the sum is zero): exact flux continuity for
    /// piecewise-constant conductivity.
    #[default]
    Harmonic,
    /// `½ (k⁻ + k⁺)`.
    Arithmetic,
}

impl FaceAveraging {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Harmonic => "harmonic",
            Self::Arithmetic => "arithmetic",
        }
    }
}

/// The case's diffusion choices (`[discretization] diffusion = { averaging = "harmonic" }`).
/// The default is omitted from the serialized intent, so W12 intent digests are unchanged.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffusionIntent {
    #[serde(default)]
    pub averaging: FaceAveraging,
}

impl DiffusionIntent {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

// ---------------------------------------------------------------------------------------------
// Lowered program
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffusiveClaim {
    /// `F_d(u, u, g) = -k(u) g`.
    Consistency,
    /// `F_d(u⁻, u⁺, g) = -F_d(u⁺, u⁻, -g)` with `k⁻`, `k⁺` exchanged.
    Antisymmetry,
    /// `F_d · (u⁺ − u⁻) ≤ 0` when `k⁻, k⁺ > 0`.
    Dissipativity,
}

/// `max_absolute` is the worst `|deviation|` for consistency and antisymmetry and the largest
/// (signed) `F_d · (u⁺ − u⁻)` for dissipativity; each must not exceed `tolerance`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiffusiveClaimCheck {
    pub claim: DiffusiveClaim,
    pub samples: usize,
    pub max_absolute: f64,
    pub tolerance: f64,
}

/// The lowered diffusive part of a [`ConservationLawProgram`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiffusionProgram {
    pub averaging: FaceAveraging,
    /// `k` reads neither the state nor per-cell data: `k_f = k`, no averaging is applied.
    pub uniform: bool,
    pub state_dependent: bool,
    /// Point kernel `k(u)` (role `DiffusionCoefficient`): operands `u`, data.
    pub coefficient: FvKernel,
    /// Facet-pair kernel `-k_f g` (role `DiffusiveFaceFlux`): operands `u_minus`, `u_plus`,
    /// `grad_n` (`FaceNormalGradient`), uniform data and sided `CellDatum` data.
    pub face_flux: FvKernel,
    pub checks: Vec<DiffusiveClaimCheck>,
}

/// A caller-supplied facet sample for the diffusive claims. `distance` is `δ·n > 0`; the face
/// gradient is `(plus − minus)/distance`. `minus_data`/`plus_data` bind per-cell data by
/// [`crate::FvDatum::name`] (falling back to `data`), `data` binds uniform data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FvDiffusiveSample {
    pub minus: f64,
    pub plus: f64,
    pub distance: f64,
    #[serde(default)]
    pub data: BTreeMap<String, f64>,
    #[serde(default)]
    pub minus_data: BTreeMap<String, f64>,
    #[serde(default)]
    pub plus_data: BTreeMap<String, f64>,
}

impl FvDiffusiveSample {
    fn gradient(&self) -> Result<f64, ConservationLawError> {
        if !(self.distance.is_finite() && self.distance > 0.0) {
            return Err(fv_error(
                "FV_SAMPLE_INVALID",
                format!("face distance {} is not finite and positive", self.distance),
            ));
        }
        if !(self.minus.is_finite() && self.plus.is_finite()) {
            return Err(fv_error("FV_SAMPLE_INVALID", "sample states are not finite"));
        }
        Ok((self.plus - self.minus) / self.distance)
    }

    fn datum(&self, side: Option<FacetSide>, name: &str) -> Result<f64, ConservationLawError> {
        let sided = match side {
            Some(FacetSide::Minus) => self.minus_data.get(name),
            Some(FacetSide::Plus) => self.plus_data.get(name),
            None => None,
        };
        sided
            .or_else(|| self.data.get(name))
            .copied()
            .ok_or_else(|| fv_error("FV_SAMPLE_INVALID", format!("sample lacks datum `{name}`")))
    }
}

/// Lower the diffusive term of `system` with the case's averaging.
pub(crate) fn lower_diffusion(
    lower: &Lower<'_>,
    data: &mut DataTable,
    term: &DiffusiveTerm,
    intent: &DiffusionIntent,
) -> Result<DiffusionProgram, ConservationLawError> {
    let lower = Lower {
        cell_data: true,
        ..*lower
    };
    let coefficient = {
        let mut builder = KernelBuilder::new("diffusion_coefficient");
        let state = builder.operand("u", FvOperandMeaning::State { side: None });
        let value = match term.coefficient {
            None => ScalarExpr::Constant(1.0),
            Some(coefficient) => {
                scalar(lower.expr(&mut builder, data, coefficient, &StateContext::Point(state))?)?
            }
        };
        builder.finish(FvKernelRole::DiffusionCoefficient, value, None)?
    };
    let averaging = intent.averaging;
    let mut builder = KernelBuilder::new(&format!("diffusive_face_flux_{}", averaging.label()));
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
    let gradient = ScalarExpr::Load(builder.operand("grad_n", FvOperandMeaning::FaceNormalGradient));
    let sided = |builder: &mut KernelBuilder, data: &mut DataTable, side| {
        let context = StateContext::Facet {
            minus,
            plus,
            side: Some(side),
            normal: Vec::new(),
        };
        match term.coefficient {
            None => Ok(ScalarExpr::Constant(1.0)),
            Some(coefficient) => scalar(lower.expr(builder, data, coefficient, &context)?),
        }
    };
    let k_minus = sided(&mut builder, data, FacetSide::Minus)?;
    let k_plus = sided(&mut builder, data, FacetSide::Plus)?;
    let uniform = k_minus == k_plus;
    let face_conductivity = if uniform {
        k_minus
    } else {
        let km = builder.bind(k_minus);
        let kp = builder.bind(k_plus);
        match averaging {
            FaceAveraging::Harmonic => {
                let sum = builder.bind(add(km.clone(), kp.clone()));
                select(
                    compare(CompareOp::NotEq, sum.clone(), ScalarExpr::Constant(0.0)),
                    ScalarExpr::binary(
                        MBinary::Div,
                        ScalarExpr::binary(
                            MBinary::Mul,
                            ScalarExpr::binary(MBinary::Mul, ScalarExpr::Constant(2.0), km),
                            kp,
                        ),
                        sum,
                    ),
                    ScalarExpr::Constant(0.0),
                )
            }
            FaceAveraging::Arithmetic => half(add(km, kp)),
        }
    };
    let value = ScalarExpr::unary(
        MUnary::Neg,
        ScalarExpr::binary(MBinary::Mul, face_conductivity, gradient),
    );
    let roles = builder.facet_roles(minus, plus);
    let face_flux = builder.finish(FvKernelRole::DiffusiveFaceFlux, value, Some(roles))?;
    Ok(DiffusionProgram {
        averaging,
        uniform,
        state_dependent: term.state_dependent,
        coefficient,
        face_flux,
        checks: Vec::new(),
    })
}

fn scalar(components: Vec<ScalarExpr>) -> Result<ScalarExpr, ConservationLawError> {
    <[ScalarExpr; 1]>::try_from(components)
        .map(|[value]| value)
        .map_err(|_| fv_error("FV_DIFFUSION_ANISOTROPIC", "diffusion coefficient is not scalar"))
}

// ---------------------------------------------------------------------------------------------
// Claim checks
// ---------------------------------------------------------------------------------------------

/// Bind one sample to the operands of a diffusive kernel. `state` gives the value of each state
/// operand; `data_side` overrides the side used for `CellDatum` operands (consistency binds
/// both sides to the minus cell's data).
fn buffers(
    kernel: &FvKernel,
    data: &[crate::FvDatum],
    sample: &FvDiffusiveSample,
    gradient: f64,
    state: impl Fn(Option<FacetSide>) -> f64,
    data_side: impl Fn(FacetSide) -> FacetSide,
) -> Result<Vec<Vec<f64>>, ConservationLawError> {
    kernel
        .operands
        .iter()
        .map(|meaning| {
            Ok(vec![match meaning {
                FvOperandMeaning::State { side } => state(*side),
                FvOperandMeaning::FaceNormalGradient => gradient,
                FvOperandMeaning::Datum { datum } => {
                    sample.datum(Some(FacetSide::Minus), &data[*datum].name)?
                }
                FvOperandMeaning::CellDatum { datum, side } => {
                    sample.datum(Some(data_side(*side)), &data[*datum].name)?
                }
                FvOperandMeaning::Normal { .. }
                | FvOperandMeaning::SlopeRatio
                | FvOperandMeaning::LimiterBound
                | FvOperandMeaning::LimiterIncrement
                | FvOperandMeaning::CellSize
                | FvOperandMeaning::Output => 0.0,
            }])
        })
        .collect()
}

/// Check the three diffusive claims of `program` at `samples`. Consistency is exact for a
/// uniform coefficient and within `8ε · max|k g|` otherwise; antisymmetry and dissipativity are
/// exact. A failing claim is a typed refusal (`FV_DIFFUSIVE_FLUX_INCONSISTENT`,
/// `FV_DIFFUSIVE_FLUX_NOT_ANTISYMMETRIC`, `FV_DIFFUSIVE_FLUX_NOT_DISSIPATIVE`); dissipativity
/// needs `k > 0` on both sides of every sample (`FV_SAMPLE_INVALID` otherwise).
pub fn check_diffusive_claims(
    program: &ConservationLawProgram,
    samples: &[FvDiffusiveSample],
) -> Result<Vec<DiffusiveClaimCheck>, ConservationLawError> {
    let diffusion = program.diffusion.as_ref().ok_or_else(|| {
        fv_error(
            "FV_DIFFUSION_ABSENT",
            "the program has no diffusive term to check",
        )
    })?;
    if samples.is_empty() {
        return Err(fv_error(
            "FV_SAMPLE_INVALID",
            "diffusive claims need at least one sample",
        ));
    }
    let face = &diffusion.face_flux;
    let coefficient = &diffusion.coefficient;
    let data = &program.data;
    let pair = validate_facet_pair(face.facet_pair().expect("diffusive face flux is a facet pair"))
        .map_err(|error| fv_error("FV_KERNEL_INVALID", error.to_string()))?;
    let mut consistency: f64 = 0.0;
    let mut reference_scale: f64 = 0.0;
    let mut antisymmetry: f64 = 0.0;
    let mut dissipativity: f64 = f64::NEG_INFINITY;
    for sample in samples {
        let gradient = sample.gradient()?;
        // Consistency: equal states and the minus cell's data on both sides.
        for value in [sample.minus, sample.plus] {
            let mut face_buffers = buffers(face, data, sample, gradient, |_| value, |_| {
                FacetSide::Minus
            })?;
            let mut coefficient_buffers =
                buffers(coefficient, data, sample, gradient, |_| value, |_| FacetSide::Minus)?;
            let numerical = run_point(face, &mut face_buffers)?;
            let k = run_point(coefficient, &mut coefficient_buffers)?;
            let reference = -(k * gradient);
            let deviation = (numerical - reference).abs();
            if !deviation.is_finite() {
                return Err(fv_error(
                    "FV_DIFFUSIVE_FLUX_INCONSISTENT",
                    "the diffusive face flux is not finite at a sample",
                ));
            }
            consistency = consistency.max(deviation);
            reference_scale = reference_scale.max(reference.abs());
        }
        // Antisymmetry: Malleus exchanges cell operands (states and sided data) and negates
        // the odd facet operands (the gradient and the output).
        let sided = buffers(
            face,
            data,
            sample,
            gradient,
            |side| match side {
                Some(FacetSide::Plus) => sample.plus,
                _ => sample.minus,
            },
            |side| side,
        )?;
        let report = check_facet_swap_symmetry(&pair, &sided, f64::MAX)
            .map_err(|error| fv_error("FV_SAMPLE_INVALID", error.to_string()))?;
        if !report.max_absolute.is_finite() {
            return Err(fv_error(
                "FV_DIFFUSIVE_FLUX_NOT_ANTISYMMETRIC",
                "the swapped diffusive face flux is not finite",
            ));
        }
        antisymmetry = antisymmetry.max(report.max_absolute);
        // Dissipativity, with k > 0 on both sides.
        for (side, value) in [(FacetSide::Minus, sample.minus), (FacetSide::Plus, sample.plus)] {
            let mut side_buffers =
                buffers(coefficient, data, sample, gradient, |_| value, |_| side)?;
            // The coefficient kernel reads unsided data; bind it from this side's cell data.
            for (index, meaning) in coefficient.operands.iter().enumerate() {
                if let FvOperandMeaning::Datum { datum } = meaning {
                    side_buffers[index][0] = sample.datum(Some(side), &data[*datum].name)?;
                }
            }
            let k = run_point(coefficient, &mut side_buffers)?;
            if !(k.is_finite() && k > 0.0) {
                return Err(fv_error(
                    "FV_SAMPLE_INVALID",
                    format!("dissipativity needs k > 0 on both sides; k{side:?} = {k}"),
                ));
            }
        }
        let mut sided = sided;
        let flux = run_point(face, &mut sided)?;
        let product = flux * (sample.plus - sample.minus);
        if !product.is_finite() {
            return Err(fv_error(
                "FV_DIFFUSIVE_FLUX_NOT_DISSIPATIVE",
                "the diffusive face flux is not finite at a sample",
            ));
        }
        dissipativity = dissipativity.max(product);
    }
    let consistency_tolerance = if diffusion.uniform {
        0.0
    } else {
        8.0 * f64::EPSILON * reference_scale
    };
    if consistency > consistency_tolerance {
        return Err(fv_error(
            "FV_DIFFUSIVE_FLUX_INCONSISTENT",
            format!(
                "F_d(u,u,g) deviates from -k(u) g by {consistency:e} > {consistency_tolerance:e}"
            ),
        ));
    }
    if antisymmetry > 0.0 {
        return Err(fv_error(
            "FV_DIFFUSIVE_FLUX_NOT_ANTISYMMETRIC",
            format!("F_d(u⁻,u⁺,g) + F_d(u⁺,u⁻,−g) deviates by {antisymmetry:e}"),
        ));
    }
    if dissipativity > 0.0 {
        return Err(fv_error(
            "FV_DIFFUSIVE_FLUX_NOT_DISSIPATIVE",
            format!("F_d · (u⁺ − u⁻) = {dissipativity:e} > 0 at a sample with k > 0"),
        ));
    }
    Ok(vec![
        DiffusiveClaimCheck {
            claim: DiffusiveClaim::Consistency,
            samples: samples.len(),
            max_absolute: consistency,
            tolerance: consistency_tolerance,
        },
        DiffusiveClaimCheck {
            claim: DiffusiveClaim::Antisymmetry,
            samples: samples.len(),
            max_absolute: antisymmetry,
            tolerance: 0.0,
        },
        DiffusiveClaimCheck {
            claim: DiffusiveClaim::Dissipativity,
            samples: samples.len(),
            max_absolute: dissipativity,
            tolerance: 0.0,
        },
    ])
}

// ---------------------------------------------------------------------------------------------
// Multidimensional limiters
// ---------------------------------------------------------------------------------------------

/// Venkatakrishnan's `K` must be finite and positive (`ε² = (K h)³ > 0` keeps the rational
/// form total).
pub(crate) fn validate_limiter(limiter: &LimiterFamily) -> Result<(), ConservationLawError> {
    if let LimiterFamily::Venkatakrishnan { k } = limiter
        && !(k.is_finite() && *k > 0.0)
    {
        return Err(fv_error(
            "FV_LIMITER_PARAMETER_INVALID",
            format!("Venkatakrishnan k = {k} must be finite and positive"),
        ));
    }
    Ok(())
}

/// `φ(Δ⁺, Δ⁻, h)` as a point kernel the realizer applies per (cell, face).
pub(crate) fn multidimensional_limiter_kernel(
    limiter: &LimiterFamily,
) -> Result<FvKernel, ConservationLawError> {
    validate_limiter(limiter)?;
    let c = ScalarExpr::Constant;
    let mul = |a, b| ScalarExpr::binary(MBinary::Mul, a, b);
    match limiter {
        LimiterFamily::BarthJespersen => {
            let mut builder = KernelBuilder::new("limiter_barth_jespersen");
            let bound = ScalarExpr::Load(builder.operand("delta_plus", FvOperandMeaning::LimiterBound));
            let increment =
                ScalarExpr::Load(builder.operand("delta_minus", FvOperandMeaning::LimiterIncrement));
            let ratio = ScalarExpr::binary(MBinary::Div, bound, increment.clone());
            let value = select(
                compare(CompareOp::Eq, increment, c(0.0)),
                c(1.0),
                ScalarExpr::binary(MBinary::Min, c(1.0), ratio),
            );
            builder.finish(FvKernelRole::Limiter, value, None)
        }
        LimiterFamily::Venkatakrishnan { k } => {
            let mut builder = KernelBuilder::new("limiter_venkatakrishnan");
            let bound = ScalarExpr::Load(builder.operand("delta_plus", FvOperandMeaning::LimiterBound));
            let increment =
                ScalarExpr::Load(builder.operand("delta_minus", FvOperandMeaning::LimiterIncrement));
            let size = ScalarExpr::Load(builder.operand("h", FvOperandMeaning::CellSize));
            let kh = builder.bind(mul(c(*k), size));
            let eps2 = builder.bind(mul(mul(kh.clone(), kh.clone()), kh));
            let bound2 = builder.bind(mul(bound.clone(), bound.clone()));
            let cross = builder.bind(mul(increment.clone(), bound));
            let numerator = add(add(bound2.clone(), eps2.clone()), mul(c(2.0), cross.clone()));
            let denominator = add(
                add(
                    add(bound2, mul(c(2.0), mul(increment.clone(), increment))),
                    cross,
                ),
                eps2,
            );
            let value = ScalarExpr::binary(MBinary::Div, numerator, denominator);
            builder.finish(FvKernelRole::Limiter, value, None)
        }
        other => Err(fv_error(
            "FV_LIMITER_UNSUPPORTED",
            format!("{other:?} is a Sweby limiter, not a multidimensional one"),
        )),
    }
}

// ---------------------------------------------------------------------------------------------
// Obligations
// ---------------------------------------------------------------------------------------------

/// The diffusive obligations, appended after the W12 list.
pub(crate) fn obligations(
    model: &SemanticModel,
    system: &ConservationLawSystem,
    diffusion: &DiffusionProgram,
) -> Vec<VerificationObligation> {
    let parent = &system.source_semantic_digest.hex;
    let equation = model
        .declarations
        .iter()
        .find(|declaration| declaration.id == system.method.equation)
        .expect("selected equation");
    let checked = !diffusion.checks.is_empty();
    let condition = |statement: &str| ValidityCondition {
        statement: statement.into(),
    };
    vec![
        make_obligation(
            parent,
            &model.name,
            equation.span,
            format!(
                "{} diffusive face flux dissipates for {}",
                diffusion.averaging.label(),
                equation.name
            ),
            VerificationObligationKind::DiffusiveFluxDissipativity {
                equation: equation.id,
            },
            vec![
                condition("k > 0 on both sides of every face"),
                condition(if checked {
                    "two-point gradient (u+ - u-)/(delta.n) with delta.n > 0; checked by Scientia at the caller's samples"
                } else {
                    "two-point gradient (u+ - u-)/(delta.n) with delta.n > 0; holds by construction, checked at samples when given"
                }),
            ],
            vec![],
            "F_d(u-,u+,g) (u+ - u-) <= 0",
            ToleranceClass::Exact,
            if checked {
                VerificationEvidenceClass::Numerical
            } else {
                VerificationEvidenceClass::Semantic
            },
            None,
        ),
        make_obligation(
            parent,
            &model.name,
            equation.span,
            format!("grid Peclet number of {} is reported per face", equation.name),
            VerificationObligationKind::GridPeclet {
                field: system.method.conserved,
            },
            vec![
                condition("informational: realized per face by Finitum from the characteristic speed and k_f"),
                condition("Pe_h > 2 flags under-resolved diffusion; the first-order monotone flux stays bounded for every Pe_h"),
            ],
            vec![],
            "Pe_h = |lambda_f| delta_f / k_f",
            ToleranceClass::Exact,
            VerificationEvidenceClass::Numerical,
            None,
        ),
    ]
}
