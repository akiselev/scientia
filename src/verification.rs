//! Scientific verification intent derived from the canonical semantic model.
//!
//! These artifacts describe what downstream products must check. They do not
//! choose numerical tolerances, execute a solver, or promote product support.

use crate::id::{Digest, span_independent_digest};
use crate::scientific::{FieldRole, semantic_digest};
use crate::semantic::{
    SemanticCompilation, SemanticDeclarationKind, SemanticExprKind, SemanticModel, SemanticRole,
};
use crate::source::SourceSpan;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const VERIFICATION_PROFILE_SCHEMA: &str = "scientia-verification-profile/1";
pub const VERIFICATION_OBLIGATION_SCHEMA: &str = "scientia-verification-obligation/1";

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
pub struct ManufacturedSolutionSpec {
    pub field: String,
    pub construction: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LimitingCaseSpec {
    pub name: String,
    pub relation: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvariantSpec {
    pub declaration: String,
    pub relation: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivativeCheckSpec {
    pub target: String,
    pub active_inputs: Vec<String>,
    pub construction: String,
    pub step_policy: String,
    pub expected_behavior: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConvergenceExpectation {
    pub field: String,
    pub refinement_axis: String,
    pub expected_order_basis: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservableDefinition {
    pub name: String,
    pub scientific_meaning: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidityCondition {
    pub statement: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormalizableObligationRef {
    pub language_neutral_claim: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "spec", rename_all = "snake_case")]
pub enum VerificationObligationKind {
    Dimension { symbol: String },
    ManufacturedSolution(ManufacturedSolutionSpec),
    LimitingCase(LimitingCaseSpec),
    Invariant(InvariantSpec),
    Derivative(DerivativeCheckSpec),
    Convergence(ConvergenceExpectation),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnsupportedGeneration {
    pub code: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
                    VerificationObligationKind::Dimension {
                        symbol: symbol.name.clone(),
                    },
                    vec![],
                    vec![symbol.name.clone()],
                    "the supplied value has the exact elaborated shape, dimension, quantity kind, and frame",
                    ToleranceClass::Exact,
                    VerificationEvidenceClass::Semantic,
                    None,
                ));
            }

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
                    SemanticDeclarationKind::Observable { .. } => observables.push(
                        ObservableDefinition {
                            name: declaration.name.clone(),
                            scientific_meaning: format!(
                                "authored observable {}::{}",
                                model.name, declaration.name
                            ),
                        },
                    ),
                    SemanticDeclarationKind::Invariant { .. } => obligations.push(make_obligation(
                        &parent,
                        &model.name,
                        declaration.span,
                        format!("authored invariant {}::{}", model.name, declaration.name),
                        VerificationObligationKind::Invariant(InvariantSpec {
                            declaration: declaration.name.clone(),
                            relation: "the invariant expression holds over its declared evaluation domain".into(),
                        }),
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
                            VerificationObligationKind::Derivative(DerivativeCheckSpec {
                                target: declaration.name.clone(),
                                active_inputs: active_inputs.clone(),
                                construction: "centered directional finite difference against the declared derivative product".into(),
                                step_policy: "evaluate a precommitted decreasing geometric sequence of symmetric positive and negative steps; reject perturbed points that leave the declared smooth stratum".into(),
                                expected_behavior: "second-order truncation before the roundoff-dominated regime; no single-step pass claim".into(),
                            }),
                            vec![ValidityCondition { statement: "the evaluation remains in the declared smooth stratum".into() }],
                            active_inputs.clone(),
                            "the derivative product matches the directional residual change",
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
                                    .map(|expression| argument_label(model, expression.index()))
                                    .unwrap_or_else(|| "authored field".into());
                                obligations.push(make_obligation(
                                    &parent,
                                    &model.name,
                                    declaration.span,
                                    format!("manufactured solution annotation {}::{}", model.name, declaration.name),
                                    VerificationObligationKind::ManufacturedSolution(ManufacturedSolutionSpec {
                                        field: field.clone(),
                                        construction: "substitute an authored exact field and derive consistent source and boundary data".into(),
                                    }),
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
                                    VerificationObligationKind::Convergence(ConvergenceExpectation {
                                        field,
                                        refinement_axis: "mesh_size".into(),
                                        expected_order_basis: "declared approximation space and measured norm".into(),
                                    }),
                                    vec![ValidityCondition { statement: "the same mathematical problem is solved on every refinement".into() }],
                                    vec!["at least three deterministic refinements".into()],
                                    "the observed error order meets the predeclared method expectation",
                                    ToleranceClass::Discretization,
                                    VerificationEvidenceClass::Numerical,
                                    None,
                                ));
                            }
                            "limiting_case" => obligations.push(make_obligation(
                                &parent,
                                &model.name,
                                declaration.span,
                                format!("limiting-case annotation in {}", model.name),
                                VerificationObligationKind::LimitingCase(LimitingCaseSpec {
                                    name: declaration.name.clone(),
                                    relation: "the authored limit recovers the declared reduced model".into(),
                                }),
                                vec![],
                                generated,
                                "the observable approaches the declared limiting relation",
                                ToleranceClass::Discretization,
                                VerificationEvidenceClass::Numerical,
                                None,
                            )),
                            // Validation annotations identify product evidence sources; they are
                            // not scientific verification checks generated by Scientia.
                            "validation" => {}
                            _ => obligations.push(make_obligation(
                                &parent,
                                &model.name,
                                declaration.span,
                                format!("unsupported verification annotation {}::{}", model.name, declaration.name),
                                VerificationObligationKind::LimitingCase(LimitingCaseSpec {
                                    name: declaration.name.clone(),
                                    relation: "generation is unavailable".into(),
                                }),
                                vec![],
                                generated,
                                "generation is structurally refused",
                                ToleranceClass::Exact,
                                VerificationEvidenceClass::Semantic,
                                Some(UnsupportedGeneration {
                                    code: "VERIFY_UNSUPPORTED_ANNOTATION".into(),
                                    reason: format!("Scientia has no safe generator for @{name}"),
                                }),
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
