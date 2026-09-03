//! Scientific verification intent derived from the canonical semantic model.
//!
//! These artifacts describe what downstream products must check. They do not
//! choose numerical tolerances, execute a solver, or promote product support.

use crate::id::{Digest, span_independent_digest};
use crate::scientific::{FieldRole, ValueShape, semantic_digest};
use crate::semantic::{
    DeclarationId, ExprId, SemanticCompilation, SemanticDeclarationKind, SemanticExprKind,
    SemanticModel, SemanticRole, SemanticShape, SymbolId,
};
use crate::source::SourceSpan;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

pub const VERIFICATION_PROFILE_SCHEMA: &str = "scientia-verification-profile/1";
pub const VERIFICATION_OBLIGATION_SCHEMA: &str = "scientia-verification-obligation/2";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationEvidenceClass {
    Semantic,
    Formal,
    Numerical,
    IndependentNumerical,
    Empirical,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToleranceClass {
    Exact,
    Roundoff,
    DirectionalTruncation,
    Discretization,
    Iterative,
    Experimental,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidityCondition {
    pub statement: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormalizableObligationRef {
    pub language_neutral_claim: String,
}

/// An `@mms` field's exact solution: either an authored expression, or a slot id (contract C2.1
/// grammar) naming the boundary/initial provider a model actually calls, since no corpus model
/// authors a standalone closed-form exact solution today.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum ExactSolutionSource {
    Authored(ExprId),
    Slot(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RefinementAxis {
    MeshSize,
    TimeStep,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum OrderBasis {
    SpaceOrderPlusOne,
    IntegratorOrder(u8),
    Declared(f64),
}

/// How a `Conservation` obligation's balance is stated. Every corpus conservation-family
/// annotation (`@conservation`, `@charge_conservation`, `@energy_balance`, `@power_balance`,
/// `@mass_conservation`, `@interface_conservation`) asserts the same shape of claim -- the
/// integral of `quantity` is preserved/balanced under the model's own dynamics -- so one variant
/// covers all of them; nothing in the annotation grammar distinguishes a different relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConservationRelation {
    GlobalBalance,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "spec", rename_all = "snake_case")]
pub enum VerificationObligationKind {
    Dimension {
        symbol: SymbolId,
    },
    ManufacturedSolution {
        field: SymbolId,
        exact: ExactSolutionSource,
    },
    Convergence {
        field: SymbolId,
        axis: RefinementAxis,
        order_basis: OrderBasis,
    },
    TemporalConvergence {
        field: SymbolId,
        order_basis: OrderBasis,
    },
    LimitingCase {
        name: String,
        relation: ExprId,
    },
    Invariant {
        declaration: DeclarationId,
        relation: ExprId,
    },
    Conservation {
        quantity: ExprId,
        relation: ConservationRelation,
    },
    PatchTest {
        field: SymbolId,
    },
    RigidBodyModes {
        field: SymbolId,
        count: u8,
    },
    DerivativeTaylor {
        block: Option<String>,
        active_inputs: Vec<SymbolId>,
    },
    /// `@inf_sup(pair = "...")`: the display pairing plus, when the model has exactly one
    /// unknown in a gradient-conforming space (H1/H(div)/H(curl)) and exactly one in L2/DG,
    /// the constrained field and its multiplier (SC-W1 follow-up for Finitum's inf-sup
    /// checker); `None` when the pairing cannot be read off the spaces.
    InfSup {
        pair: String,
        constrained: Option<SymbolId>,
        multiplier: Option<SymbolId>,
    },
    /// Not part of contract C6.1's kind list: a content-free carrier for obligations whose
    /// `unsupported` field is set (see `VERIFY_UNSUPPORTED_ANNOTATION`). Every other kind now
    /// requires a real `SymbolId`/`ExprId`, and a genuinely unsupported annotation (e.g.
    /// `@shock_tube`, `@bh_curve`) has none to offer honestly; this variant exists so Scientia
    /// never fabricates one. Flagged as a suggested C11 amendment in the landing report.
    Unsupported {
        name: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnsupportedGeneration {
    pub code: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerificationObligation {
    pub schema: String,
    pub id: Digest,
    pub parent_semantic_digest: String,
    pub model: String,
    pub source_span: SourceSpan,
    pub scientific_meaning: String,
    pub kind: VerificationObligationKind,
    pub applicability: Vec<ValidityCondition>,
    pub generated_inputs: Vec<String>,
    pub expected_relation: String,
    pub tolerance_class: ToleranceClass,
    pub evidence_class: VerificationEvidenceClass,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unsupported: Option<UnsupportedGeneration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub formalizable: Option<FormalizableObligationRef>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservableDefinition {
    pub name: String,
    pub scientific_meaning: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerificationProfile {
    pub schema: String,
    pub parent_semantic_digest: String,
    pub model: String,
    pub observables: Vec<ObservableDefinition>,
    pub obligations: Vec<VerificationObligation>,
    pub artifact_digest: Digest,
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("{code}: {message}")]
pub struct VerificationProfileError {
    pub code: String,
    pub message: String,
}

#[derive(Serialize)]
struct ObligationIdentity<'a> {
    schema: &'static str,
    parent_semantic_digest: &'a str,
    model: &'a str,
    scientific_meaning: &'a str,
    kind: &'a VerificationObligationKind,
    applicability: &'a [ValidityCondition],
    generated_inputs: &'a [String],
    expected_relation: &'a str,
    tolerance_class: ToleranceClass,
    evidence_class: VerificationEvidenceClass,
    unsupported: &'a Option<UnsupportedGeneration>,
    formalizable: &'a Option<FormalizableObligationRef>,
}

#[derive(Serialize)]
struct ProfileIdentity<'a> {
    schema: &'static str,
    parent_semantic_digest: &'a str,
    model: &'a str,
    observables: &'a [ObservableDefinition],
    obligations: &'a [VerificationObligation],
}

impl VerificationObligation {
    pub fn validate(&self) -> Result<(), VerificationProfileError> {
        if self.schema != VERIFICATION_OBLIGATION_SCHEMA
            || self.model.trim().is_empty()
            || self.parent_semantic_digest.trim().is_empty()
            || self.scientific_meaning.trim().is_empty()
            || self.expected_relation.trim().is_empty()
        {
            return Err(profile_error(
                "VERIFY_INVALID_OBLIGATION",
                "verification obligation is incomplete or has an unsupported schema",
            ));
        }
        if self
            .applicability
            .windows(2)
            .any(|pair| pair[0].statement >= pair[1].statement)
            || self
                .generated_inputs
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(profile_error(
                "VERIFY_NONCANONICAL_OBLIGATION",
                "applicability and generated inputs must be uniquely sorted",
            ));
        }
        let expected = span_independent_digest(&ObligationIdentity {
            schema: VERIFICATION_OBLIGATION_SCHEMA,
            parent_semantic_digest: &self.parent_semantic_digest,
            model: &self.model,
            scientific_meaning: &self.scientific_meaning,
            kind: &self.kind,
            applicability: &self.applicability,
            generated_inputs: &self.generated_inputs,
            expected_relation: &self.expected_relation,
            tolerance_class: self.tolerance_class,
            evidence_class: self.evidence_class,
            unsupported: &self.unsupported,
            formalizable: &self.formalizable,
        });
        if self.id != expected {
            return Err(profile_error(
                "VERIFY_OBLIGATION_IDENTITY_MISMATCH",
                "obligation identity does not match its scientific contents",
            ));
        }
        Ok(())
    }
}

impl VerificationProfile {
    pub fn validate(&self) -> Result<(), VerificationProfileError> {
        if self.schema != VERIFICATION_PROFILE_SCHEMA
            || self.model.trim().is_empty()
            || self.parent_semantic_digest.trim().is_empty()
        {
            return Err(profile_error(
                "VERIFY_INVALID_PROFILE",
                "verification profile is incomplete or has an unsupported schema",
            ));
        }
        if self
            .obligations
            .windows(2)
            .any(|pair| pair[0].id >= pair[1].id)
            || self
                .observables
                .windows(2)
                .any(|pair| pair[0].name >= pair[1].name)
        {
            return Err(profile_error(
                "VERIFY_NONCANONICAL_PROFILE",
                "profile obligations and observables must be uniquely sorted",
            ));
        }
        for obligation in &self.obligations {
            obligation.validate()?;
            if obligation.model != self.model
                || obligation.parent_semantic_digest != self.parent_semantic_digest
            {
                return Err(profile_error(
                    "VERIFY_PARENT_MISMATCH",
                    "obligation parent does not match its verification profile",
                ));
            }
        }
        let expected = span_independent_digest(&ProfileIdentity {
            schema: VERIFICATION_PROFILE_SCHEMA,
            parent_semantic_digest: &self.parent_semantic_digest,
            model: &self.model,
            observables: &self.observables,
            obligations: &self.obligations,
        });
        if self.artifact_digest != expected {
            return Err(profile_error(
                "VERIFY_PROFILE_IDENTITY_MISMATCH",
                "profile identity does not match its scientific contents",
            ));
        }
        Ok(())
    }
}

fn symbol_from_expr(model: &SemanticModel, expr: ExprId) -> Option<SymbolId> {
    match model.expressions.get(expr.index())?.kind {
        SemanticExprKind::Symbol { symbol } => Some(symbol),
        _ => None,
    }
}

fn number_arg(
    model: &SemanticModel,
    arguments: &BTreeMap<String, ExprId>,
    key: &str,
) -> Option<f64> {
    let expr = arguments.get(key)?;
    match model.expressions.get(expr.index())?.kind {
        SemanticExprKind::Number { value, .. } => Some(value),
        _ => None,
    }
}

fn sole_field_with_role(model: &SemanticModel, role: &FieldRole) -> Option<SymbolId> {
    let mut found = None;
    for symbol in &model.symbols {
        if let SemanticRole::PhysicalField(candidate) = &symbol.ty.role
            && candidate == role
        {
            if found.is_some() {
                return None;
            }
            found = Some(symbol.id);
        }
    }
    found
}

fn sole_vector_field_with_role(model: &SemanticModel, role: &FieldRole) -> Option<SymbolId> {
    let mut found = None;
    for symbol in &model.symbols {
        if let SemanticRole::PhysicalField(candidate) = &symbol.ty.role
            && candidate == role
            && matches!(
                symbol.ty.shape,
                SemanticShape::Numeric(ValueShape::Vector(_))
            )
        {
            if found.is_some() {
                return None;
            }
            found = Some(symbol.id);
        }
    }
    found
}

/// A `@rigid_body_modes` annotation carries no `field` argument in the corpus; the same model's
/// own `@patch_test(field = ...)` (when present) names the field these checks are paired with in
/// every corpus occurrence, so it takes priority over the shape-based fallback.
fn companion_patch_test_field(model: &SemanticModel) -> Option<SymbolId> {
    model.declarations.iter().find_map(|declaration| {
        let SemanticDeclarationKind::Verification { arguments } = &declaration.kind else {
            return None;
        };
        if declaration.name != "patch_test" {
            return None;
        }
        symbol_from_expr(model, *arguments.get("field")?)
    })
}

/// Slot id for an `@mms(field = ...)` field with no authored exact expression (contract C6.1):
/// the boundary/initial condition's own provider call when one exists, else `provider/exact_<field>`.
fn manufactured_exact_source(model: &SemanticModel, field: SymbolId) -> ExactSolutionSource {
    let mut value_expr = None;
    for declaration in &model.declarations {
        if let SemanticDeclarationKind::BoundaryCondition {
            target: Some(target),
            value,
            ..
        } = &declaration.kind
            && *target == field
        {
            value_expr = Some(*value);
            break;
        }
    }
    if value_expr.is_none() {
        for declaration in &model.declarations {
            if let SemanticDeclarationKind::InitialCondition {
                target: Some(target),
                value,
            } = &declaration.kind
                && *target == field
            {
                value_expr = Some(*value);
                break;
            }
        }
    }
    if let Some(value) = value_expr
        && let Some(expression) = model.expressions.get(value.index())
        && let SemanticExprKind::ProviderCall { provider, .. } = &expression.kind
        && let Some(found) = model
            .providers
            .iter()
            .find(|candidate| candidate.id == *provider)
    {
        return ExactSolutionSource::Slot(format!("provider/{}", found.name));
    }
    let field_name = model
        .symbols
        .get(field.index())
        .map(|symbol| symbol.name.as_str())
        .unwrap_or("field");
    ExactSolutionSource::Slot(format!("provider/exact_{field_name}"))
}

/// The saddle-point pairing readable from the spaces alone: the unique unknown/state field in
/// H1, H(div), or H(curl) is the constrained field and the unique one in L2/DG is the
/// multiplier; anything else is undecidable here (Finitum's `InfSupPairing::from_structure`
/// decides from the operator structure).
fn inf_sup_pairing(model: &SemanticModel) -> (Option<SymbolId>, Option<SymbolId>) {
    use crate::scientific::SpaceFamily;
    let mut constrained = Vec::new();
    let mut multipliers = Vec::new();
    for symbol in &model.symbols {
        if !matches!(
            symbol.ty.role,
            SemanticRole::PhysicalField(FieldRole::Unknown | FieldRole::State)
        ) {
            continue;
        }
        match symbol.space.as_ref().map(|space| &space.family) {
            Some(SpaceFamily::H1 | SpaceFamily::HDiv | SpaceFamily::HCurl) => {
                constrained.push(symbol.id)
            }
            Some(SpaceFamily::L2 | SpaceFamily::Dg) => multipliers.push(symbol.id),
            None => {}
        }
    }
    match (constrained.as_slice(), multipliers.as_slice()) {
        ([constrained], [multiplier]) => (Some(*constrained), Some(*multiplier)),
        _ => (None, None),
    }
}

fn conservation_quantity(arguments: &BTreeMap<String, ExprId>) -> Option<ExprId> {
    if arguments.len() == 1 {
        arguments.values().next().copied()
    } else {
        None
    }
}

/// Derive deterministic scientific verification intent without running any
/// numerical method. One profile is emitted per semantic model.
#[must_use]
pub fn derive_verification_profiles(compilation: &SemanticCompilation) -> Vec<VerificationProfile> {
    let parent = semantic_digest(&compilation.source);
    let mut profiles = compilation
        .semantic
        .models
        .iter()
        .map(|model| {
            let mut observables = Vec::new();
            let mut obligations = Vec::new();

            for symbol in &model.symbols {
                let meaning = format!("{}::{} has its elaborated scientific type", model.name, symbol.name);
                obligations.push(make_obligation(
                    &parent,
                    &model.name,
                    symbol.span,
                    meaning,
                    VerificationObligationKind::Dimension { symbol: symbol.id },
                    vec![],
                    vec![symbol.name.clone()],
                    "the supplied value has the exact elaborated shape, dimension, quantity kind, and frame",
                    ToleranceClass::Exact,
                    VerificationEvidenceClass::Semantic,
                    None,
                ));
            }

            let active_input_symbols = model
                .symbols
                .iter()
                .filter(|symbol| {
                    matches!(
                        symbol.ty.role,
                        SemanticRole::Parameter
                            | SemanticRole::Property
                            | SemanticRole::PhysicalField(FieldRole::Parameter)
                            | SemanticRole::PhysicalField(FieldRole::Coefficient)
                    )
                })
                .map(|symbol| symbol.id)
                .collect::<Vec<_>>();
            let active_inputs = model
                .symbols
                .iter()
                .filter(|symbol| {
                    matches!(
                        symbol.ty.role,
                        SemanticRole::Parameter
                            | SemanticRole::Property
                            | SemanticRole::PhysicalField(FieldRole::Parameter)
                            | SemanticRole::PhysicalField(FieldRole::Coefficient)
                    )
                })
                .map(|symbol| symbol.name.clone())
                .collect::<Vec<_>>();

            for declaration in &model.declarations {
                match &declaration.kind {
                    SemanticDeclarationKind::Observable { .. }
                    | SemanticDeclarationKind::Objective { .. } => observables.push(
                        ObservableDefinition {
                            name: declaration.name.clone(),
                            scientific_meaning: format!(
                                "authored observable {}::{}",
                                model.name, declaration.name
                            ),
                        },
                    ),
                    SemanticDeclarationKind::Invariant { value } => obligations.push(make_obligation(
                        &parent,
                        &model.name,
                        declaration.span,
                        format!("authored invariant {}::{}", model.name, declaration.name),
                        VerificationObligationKind::Invariant {
                            declaration: declaration.id,
                            relation: *value,
                        },
                        vec![],
                        vec![declaration.name.clone()],
                        "the authored invariant evaluates true",
                        ToleranceClass::Roundoff,
                        VerificationEvidenceClass::Numerical,
                        None,
                    )),
                    SemanticDeclarationKind::Equation { .. } if !active_inputs.is_empty() => {
                        obligations.push(make_obligation(
                            &parent,
                            &model.name,
                            declaration.span,
                            format!("directional derivative of equation {}::{}", model.name, declaration.name),
                            VerificationObligationKind::DerivativeTaylor {
                                block: Some(declaration.name.clone()),
                                active_inputs: active_input_symbols.clone(),
                            },
                            vec![ValidityCondition { statement: "the evaluation remains in the declared smooth stratum".into() }],
                            active_inputs.clone(),
                            "the derivative product matches the directional residual change; second-order truncation before roundoff",
                            ToleranceClass::DirectionalTruncation,
                            VerificationEvidenceClass::Numerical,
                            None,
                        ));
                    }
                    SemanticDeclarationKind::Verification { arguments } => {
                        let generated = arguments
                            .iter()
                            .map(|(key, expression)| {
                                format!("{key}={}", argument_label(model, expression.index()))
                            })
                            .collect::<Vec<_>>();
                        let name = declaration.name.as_str();
                        match name {
                            "mms" => {
                                let field = arguments
                                    .get("field")
                                    .and_then(|expression| symbol_from_expr(model, *expression));
                                match field {
                                    Some(field) => {
                                        obligations.push(make_obligation(
                                            &parent,
                                            &model.name,
                                            declaration.span,
                                            format!("manufactured solution annotation {}::{}", model.name, declaration.name),
                                            VerificationObligationKind::ManufacturedSolution {
                                                field,
                                                exact: manufactured_exact_source(model, field),
                                            },
                                            vec![],
                                            generated.clone(),
                                            "the realized residual and boundary data reproduce the manufactured field",
                                            ToleranceClass::Discretization,
                                            VerificationEvidenceClass::Numerical,
                                            None,
                                        ));
                                        obligations.push(make_obligation(
                                            &parent,
                                            &model.name,
                                            declaration.span,
                                            format!("mesh convergence for manufactured field in {}", model.name),
                                            VerificationObligationKind::Convergence {
                                                field,
                                                axis: RefinementAxis::MeshSize,
                                                order_basis: OrderBasis::SpaceOrderPlusOne,
                                            },
                                            vec![ValidityCondition { statement: "the same mathematical problem is solved on every refinement".into() }],
                                            vec!["at least three deterministic refinements".into()],
                                            "the observed error order meets the predeclared method expectation",
                                            ToleranceClass::Discretization,
                                            VerificationEvidenceClass::Numerical,
                                            None,
                                        ));
                                    }
                                    None => obligations.push(unsupported_obligation(
                                        &parent, model, declaration, name, generated,
                                        "no `field` argument resolves to a declared symbol",
                                    )),
                                }
                            }
                            "spatial_convergence" => {
                                let field = sole_field_with_role(model, &FieldRole::State);
                                let order = number_arg(model, arguments, "order");
                                match (field, order) {
                                    (Some(field), Some(order)) => obligations.push(make_obligation(
                                        &parent,
                                        &model.name,
                                        declaration.span,
                                        format!("spatial convergence annotation in {}", model.name),
                                        VerificationObligationKind::Convergence {
                                            field,
                                            axis: RefinementAxis::MeshSize,
                                            order_basis: OrderBasis::Declared(order),
                                        },
                                        vec![ValidityCondition { statement: "the same mathematical problem is solved on every refinement".into() }],
                                        generated.clone(),
                                        "the observed spatial error order meets the declared order",
                                        ToleranceClass::Discretization,
                                        VerificationEvidenceClass::Numerical,
                                        None,
                                    )),
                                    _ => obligations.push(unsupported_obligation(
                                        &parent, model, declaration, name, generated,
                                        "no unique state field or declared order to convergence-check",
                                    )),
                                }
                            }
                            "temporal_convergence" => {
                                let field = sole_field_with_role(model, &FieldRole::State);
                                let order = number_arg(model, arguments, "order");
                                match (field, order) {
                                    (Some(field), Some(order)) => obligations.push(make_obligation(
                                        &parent,
                                        &model.name,
                                        declaration.span,
                                        format!("temporal convergence annotation in {}", model.name),
                                        VerificationObligationKind::TemporalConvergence {
                                            field,
                                            order_basis: OrderBasis::Declared(order),
                                        },
                                        vec![ValidityCondition { statement: "the same mathematical problem is solved on every time-step refinement".into() }],
                                        generated.clone(),
                                        "the observed temporal error order meets the declared order",
                                        ToleranceClass::Discretization,
                                        VerificationEvidenceClass::Numerical,
                                        None,
                                    )),
                                    _ => obligations.push(unsupported_obligation(
                                        &parent, model, declaration, name, generated,
                                        "no unique state field or declared order to convergence-check",
                                    )),
                                }
                            }
                            "patch_test" => {
                                let field = arguments
                                    .get("field")
                                    .and_then(|expression| symbol_from_expr(model, *expression));
                                match field {
                                    Some(field) => obligations.push(make_obligation(
                                        &parent,
                                        &model.name,
                                        declaration.span,
                                        format!("patch test annotation in {}", model.name),
                                        VerificationObligationKind::PatchTest { field },
                                        vec![],
                                        generated.clone(),
                                        "a constant-strain patch reproduces exactly on any admissible mesh",
                                        ToleranceClass::Discretization,
                                        VerificationEvidenceClass::Numerical,
                                        None,
                                    )),
                                    None => obligations.push(unsupported_obligation(
                                        &parent, model, declaration, name, generated,
                                        "no `field` argument resolves to a declared symbol",
                                    )),
                                }
                            }
                            "rigid_body_modes" => {
                                let field = companion_patch_test_field(model)
                                    .or_else(|| sole_vector_field_with_role(model, &FieldRole::Unknown));
                                let count = number_arg(model, arguments, "count");
                                match (field, count) {
                                    (Some(field), Some(count)) => obligations.push(make_obligation(
                                        &parent,
                                        &model.name,
                                        declaration.span,
                                        format!("rigid-body-modes annotation in {}", model.name),
                                        VerificationObligationKind::RigidBodyModes {
                                            field,
                                            count: count as u8,
                                        },
                                        vec![],
                                        generated.clone(),
                                        "the declared number of zero-energy rigid-body modes are recovered",
                                        ToleranceClass::Roundoff,
                                        VerificationEvidenceClass::Numerical,
                                        None,
                                    )),
                                    _ => obligations.push(unsupported_obligation(
                                        &parent, model, declaration, name, generated,
                                        "no unique vector field or declared count for rigid-body modes",
                                    )),
                                }
                            }
                            "jvp_taylor" => {
                                let block = arguments
                                    .get("block")
                                    .map(|expression| argument_label(model, expression.index()));
                                obligations.push(make_obligation(
                                    &parent,
                                    &model.name,
                                    declaration.span,
                                    format!("JVP Taylor-remainder annotation in {}", model.name),
                                    VerificationObligationKind::DerivativeTaylor {
                                        block,
                                        active_inputs: active_input_symbols.clone(),
                                    },
                                    vec![ValidityCondition { statement: "the evaluation remains in the declared smooth stratum".into() }],
                                    generated.clone(),
                                    "the derivative product matches the directional residual change; second-order truncation before roundoff",
                                    ToleranceClass::DirectionalTruncation,
                                    VerificationEvidenceClass::Numerical,
                                    None,
                                ));
                            }
                            "inf_sup" => {
                                let pair = arguments
                                    .get("pair")
                                    .map(|expression| argument_label(model, expression.index()));
                                let (constrained, multiplier) = inf_sup_pairing(model);
                                match pair {
                                    Some(pair) => obligations.push(make_obligation(
                                        &parent,
                                        &model.name,
                                        declaration.span,
                                        format!("inf-sup stability annotation in {}", model.name),
                                        VerificationObligationKind::InfSup {
                                            pair,
                                            constrained,
                                            multiplier,
                                        },
                                        vec![],
                                        generated.clone(),
                                        "the discrete inf-sup constant stays bounded away from zero under refinement",
                                        ToleranceClass::Discretization,
                                        VerificationEvidenceClass::Numerical,
                                        None,
                                    )),
                                    None => obligations.push(unsupported_obligation(
                                        &parent, model, declaration, name, generated,
                                        "no `pair` argument to name the discrete space pairing",
                                    )),
                                }
                            }
                            "energy_balance" | "power_balance" | "charge_conservation"
                            | "mass_conservation" | "conservation" | "interface_conservation" => {
                                match conservation_quantity(arguments) {
                                    Some(quantity) => obligations.push(make_obligation(
                                        &parent,
                                        &model.name,
                                        declaration.span,
                                        format!("conservation annotation {}::{}", model.name, name),
                                        VerificationObligationKind::Conservation {
                                            quantity,
                                            relation: ConservationRelation::GlobalBalance,
                                        },
                                        vec![],
                                        generated.clone(),
                                        "the declared quantity's global balance is preserved",
                                        ToleranceClass::Discretization,
                                        VerificationEvidenceClass::Numerical,
                                        None,
                                    )),
                                    // A bare `@energy_balance()`/`@charge_conservation()`/... names
                                    // no expression at all; Scientia does not guess which model
                                    // expression is "the" conserved quantity (see the landing
                                    // report for the annotations this affects).
                                    None => obligations.push(unsupported_obligation(
                                        &parent, model, declaration, name, generated,
                                        "the bare annotation names no quantity expression to reference",
                                    )),
                                }
                            }
                            "limiting_case" => {
                                let relation = if arguments.len() == 1 {
                                    arguments.values().next().copied()
                                } else {
                                    None
                                };
                                match relation {
                                    Some(relation) => obligations.push(make_obligation(
                                        &parent,
                                        &model.name,
                                        declaration.span,
                                        format!("limiting-case annotation in {}", model.name),
                                        VerificationObligationKind::LimitingCase {
                                            name: declaration.name.clone(),
                                            relation,
                                        },
                                        vec![],
                                        generated,
                                        "the observable approaches the declared limiting relation",
                                        ToleranceClass::Discretization,
                                        VerificationEvidenceClass::Numerical,
                                        None,
                                    )),
                                    None => obligations.push(unsupported_obligation(
                                        &parent, model, declaration, name, generated,
                                        "no unique relation argument to state the limiting case",
                                    )),
                                }
                            }
                            // Validation annotations identify product evidence sources; they are
                            // not scientific verification checks generated by Scientia.
                            "validation" => {}
                            _ => obligations.push(unsupported_obligation(
                                &parent, model, declaration, name, generated,
                                &format!("Scientia has no safe generator for @{name}"),
                            )),
                        }
                    }
                    _ => {}
                }
            }

            observables.sort_by(|left, right| left.name.cmp(&right.name));
            obligations.sort_by(|left, right| left.id.cmp(&right.id));
            let mut profile = VerificationProfile {
                schema: VERIFICATION_PROFILE_SCHEMA.into(),
                parent_semantic_digest: parent.clone(),
                model: model.name.clone(),
                observables,
                obligations,
                artifact_digest: Digest::blake3(&[]),
            };
            profile.artifact_digest = span_independent_digest(&ProfileIdentity {
                schema: VERIFICATION_PROFILE_SCHEMA,
                parent_semantic_digest: &profile.parent_semantic_digest,
                model: &profile.model,
                observables: &profile.observables,
                obligations: &profile.obligations,
            });
            profile
                .validate()
                .expect("derived verification profiles are canonical");
            profile
        })
        .collect::<Vec<_>>();
    profiles.sort_by(|left, right| left.model.cmp(&right.model));
    profiles
}

#[allow(clippy::too_many_arguments)]
fn make_obligation(
    parent_semantic_digest: &str,
    model: &str,
    source_span: SourceSpan,
    scientific_meaning: String,
    kind: VerificationObligationKind,
    mut applicability: Vec<ValidityCondition>,
    mut generated_inputs: Vec<String>,
    expected_relation: &str,
    tolerance_class: ToleranceClass,
    evidence_class: VerificationEvidenceClass,
    unsupported: Option<UnsupportedGeneration>,
) -> VerificationObligation {
    applicability.sort_by(|left, right| left.statement.cmp(&right.statement));
    generated_inputs.sort();
    let id = span_independent_digest(&ObligationIdentity {
        schema: VERIFICATION_OBLIGATION_SCHEMA,
        parent_semantic_digest,
        model,
        scientific_meaning: &scientific_meaning,
        kind: &kind,
        applicability: &applicability,
        generated_inputs: &generated_inputs,
        expected_relation,
        tolerance_class,
        evidence_class,
        unsupported: &unsupported,
        formalizable: &None,
    });
    VerificationObligation {
        schema: VERIFICATION_OBLIGATION_SCHEMA.into(),
        id,
        parent_semantic_digest: parent_semantic_digest.into(),
        model: model.into(),
        source_span,
        scientific_meaning,
        kind,
        applicability,
        generated_inputs,
        expected_relation: expected_relation.into(),
        tolerance_class,
        evidence_class,
        unsupported,
        formalizable: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn unsupported_obligation(
    parent: &str,
    model: &SemanticModel,
    declaration: &crate::semantic::SemanticDeclaration,
    name: &str,
    generated: Vec<String>,
    reason: &str,
) -> VerificationObligation {
    make_obligation(
        parent,
        &model.name,
        declaration.span,
        format!(
            "unsupported verification annotation {}::{}",
            model.name, declaration.name
        ),
        VerificationObligationKind::Unsupported { name: name.into() },
        vec![],
        generated,
        "generation is structurally refused",
        ToleranceClass::Exact,
        VerificationEvidenceClass::Semantic,
        Some(UnsupportedGeneration {
            code: "VERIFY_UNSUPPORTED_ANNOTATION".into(),
            reason: reason.into(),
        }),
    )
}

fn argument_label(model: &SemanticModel, expression: usize) -> String {
    match &model.expressions[expression].kind {
        SemanticExprKind::Symbol { symbol } => model.symbols[symbol.index()].name.clone(),
        SemanticExprKind::String { value } => value.clone(),
        SemanticExprKind::Number { value, .. } => value.to_string(),
        _ => format!("semantic-expression-{expression}"),
    }
}

fn profile_error(code: &str, message: &str) -> VerificationProfileError {
    VerificationProfileError {
        code: code.into(),
        message: message.into(),
    }
}
