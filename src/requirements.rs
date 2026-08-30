//! Mesh-free finite-element requirements inferred from typed variational forms.
//!
//! This is the FC3 boundary.  The types in this module describe mathematical needs; they do not
//! select a mesh, reference cell, basis table, quadrature rule, DOF layout, or assembly strategy.

use crate::formulation::{FormArgumentRole, FormAssumption, FormCaptureRole, VariationalForm};
use crate::id::{Digest, span_independent_digest};
use crate::scientific::{
    BinaryOp, BoundaryConditionKind, Continuity, FieldRole, SpaceFamily, SpaceSpec, UnaryOp,
    ValueShape,
};
use crate::semantic::{
    AxisContraction, DeclarationId, DifferentialOperator, DomainId, ExprId, ProviderId, RegionId,
    RegionKind, SemanticDeclarationKind, SemanticExpr, SemanticExprKind, SemanticMeasure,
    SemanticModel, SemanticModule, SemanticRole, SemanticShape, SemanticType, SymbolId, TraceSide,
};
use crate::source::SourceSpan;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const FORM_REQUIREMENTS_SCHEMA: &str = "scientia-form-requirements/1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpaceComposition {
    Empty,
    Single,
    Product,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpaceSystemRequirement {
    pub composition: SpaceComposition,
    pub argument_spaces: Vec<SymbolId>,
    pub coefficient_spaces: Vec<SymbolId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpaceBindingRole {
    Test,
    Trial,
    PhysicalField(FieldRole),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PullbackRequirement {
    H1Composition,
    L2Density,
    CovariantPiola,
    ContravariantPiola,
    Broken,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrientationRequirement {
    EdgeTangential,
    FacetTangential,
    FacetNormal,
    TwoSidedFacet,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceRequirement {
    ExteriorValue,
    MinusValue,
    PlusValue,
    ExteriorTangential,
    MinusTangential,
    PlusTangential,
    ExteriorNormal,
    MinusNormal,
    PlusNormal,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpaceRequirement {
    pub symbol: SymbolId,
    pub role: SpaceBindingRole,
    pub domain: DomainId,
    pub value_shape: ValueShape,
    pub space: SpaceSpec,
    pub pullback: PullbackRequirement,
    pub orientations: Vec<OrientationRequirement>,
    pub traces: Vec<TraceRequirement>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ElementFamilyRequirement {
    H1,
    L2,
    Hcurl,
    Hdiv,
    DiscontinuousGalerkin,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ElementRequirement {
    pub symbol: SymbolId,
    pub domain: DomainId,
    pub topological_dimension: u8,
    pub family: ElementFamilyRequirement,
    pub polynomial_order: u8,
    pub value_shape: ValueShape,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivativeEvaluation {
    Value,
    Gradient,
    Divergence,
    Curl,
    RotatedGradient,
    SymmetricGradient,
    TimeDerivative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationSite {
    Cell,
    ExteriorTrace,
    MinusTrace,
    PlusTrace,
    Point,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceMapping {
    Value,
    Tangential,
    Normal,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct BasisEvaluationRequirement {
    pub derivative: DerivativeEvaluation,
    pub site: EvaluationSite,
    pub trace_mapping: Option<TraceMapping>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputSourceRequirement {
    Basis,
    ExternalValue,
    ModelDefinedValue { definition: ExprId },
    ModelDefinedProperty { definition: ExprId },
    ModelDefinedConstitutive { definition: ExprId },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputPreprocessingRequirement {
    pub symbol: SymbolId,
    pub source: InputSourceRequirement,
    pub evaluations: Vec<BasisEvaluationRequirement>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryPreprocessingRequirement {
    Jacobian,
    InverseJacobian,
    JacobianDeterminant,
    FacetJacobian,
    FacetNormal,
    PointLocation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuadraturePrecision {
    PolynomialExact,
    CoefficientDependent,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuadratureIntent {
    /// A lower bound on exactness for the known polynomial part of the integrand.
    pub minimum_polynomial_degree: u16,
    pub precision: QuadraturePrecision,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeasureRequirement {
    pub measure: SemanticMeasure,
    pub domain: DomainId,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelSignature {
    pub measure: MeasureRequirement,
    pub output_type: SemanticType,
    pub inputs: Vec<InputPreprocessingRequirement>,
    pub geometry: Vec<GeometryPreprocessingRequirement>,
    pub quadrature: QuadratureIntent,
    /// Digest of the typed expression tree with arena IDs and source spans removed.
    pub integrand_digest: Digest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegralOccurrence {
    pub integral_index: usize,
    pub source_span: SourceSpan,
}

/// Integrals are combined only when this complete signature (including the integrand) matches.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizedIntegralGroup {
    pub signature_digest: Digest,
    pub signature: KernelSignature,
    pub occurrences: Vec<IntegralOccurrence>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EssentialConstraintRequirement {
    pub argument: SymbolId,
    pub region: RegionId,
    pub condition: DeclarationId,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryPartitionRequirement {
    pub domain: DomainId,
    pub exterior_regions: Vec<RegionId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequirementInferenceMethod {
    TypedFormFc3,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequirementInferenceReceipt {
    pub source_form_digest: Digest,
    pub source_declaration: DeclarationId,
    pub method: RequirementInferenceMethod,
}

/// Pure, realization-neutral requirements for satisfying a variational form.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormRequirements {
    pub schema: String,
    pub model: String,
    pub form: String,
    pub artifact_digest: Digest,
    pub space_system: SpaceSystemRequirement,
    pub spaces: Vec<SpaceRequirement>,
    pub elements: Vec<ElementRequirement>,
    pub integral_groups: Vec<NormalizedIntegralGroup>,
    pub essential_constraints: Vec<EssentialConstraintRequirement>,
    pub boundary_partitions: Vec<BoundaryPartitionRequirement>,
    pub receipt: RequirementInferenceReceipt,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum RequirementInferenceError {
    #[error(
        "REQ_SOURCE_MISMATCH: form `{form}` does not originate from the supplied semantic module"
    )]
    SourceMismatch { form: String },
    #[error("REQ_MISSING_MODEL: semantic module has no model named `{0}`")]
    MissingModel(String),
    #[error("REQ_INVALID_EXPRESSION: expression {0} is outside the form arena")]
    InvalidExpression(ExprId),
    #[error("REQ_INVALID_SYMBOL: form references unclassified symbol {0}")]
    InvalidSymbol(SymbolId),
    #[error("REQ_INVALID_DOMAIN: domain {0} is outside the semantic model")]
    InvalidDomain(DomainId),
    #[error("REQ_INVALID_REGION: region {0} is outside the semantic model")]
    InvalidRegion(RegionId),
    #[error("REQ_REGION_KIND: region {region} has kind {actual:?}, expected {expected:?}")]
    RegionKindMismatch {
        region: RegionId,
        expected: RegionKind,
        actual: RegionKind,
    },
    #[error("REQ_MEASURE_DOMAIN: integral {integral} has incompatible domains {domains:?}")]
    IncompatibleMeasureDomains {
        integral: usize,
        domains: Vec<DomainId>,
    },
    #[error("REQ_MEASURE_DOMAIN: integral {integral} has no domain-bearing input or region")]
    MissingMeasureDomain { integral: usize },
    #[error(
        "REQ_INTEGRAND_AXES: integral {integral} must be scalar, got {shape:?} with axes {axes}"
    )]
    NonScalarIntegrand {
        integral: usize,
        shape: SemanticShape,
        axes: usize,
    },
    #[error(
        "REQ_SPACE_CONTINUITY: symbol {symbol} has {family:?} with incompatible continuity {continuity:?}"
    )]
    IncompatibleContinuity {
        symbol: SymbolId,
        family: SpaceFamily,
        continuity: Continuity,
    },
    #[error(
        "REQ_SPACE_SHAPE: symbol {symbol} in {family:?} must be a vector of domain extent {dimension}, got {shape:?}"
    )]
    IncompatibleSpaceShape {
        symbol: SymbolId,
        family: SpaceFamily,
        dimension: u8,
        shape: SemanticShape,
    },
    #[error(
        "REQ_DIFFERENTIAL_SPACE: {operator:?} is incompatible with {family:?} for symbol {symbol}"
    )]
    IncompatibleDifferentialSpace {
        symbol: SymbolId,
        family: SpaceFamily,
        operator: DifferentialOperator,
    },
    #[error("REQ_TRACE_SPACE: normal trace is incompatible with {family:?} for symbol {symbol}")]
    IncompatibleNormalTrace {
        symbol: SymbolId,
        family: SpaceFamily,
    },
    #[error(
        "REQ_BOUNDARY_DATA: condition {condition} does not provide compatible essential data on region {region}"
    )]
    IncompatibleBoundaryData {
        condition: DeclarationId,
        region: RegionId,
    },
}

/// Infer the complete FC3 requirements for a typed form.
pub fn infer_form_requirements(
    module: &SemanticModule,
    form: &VariationalForm,
) -> Result<FormRequirements, RequirementInferenceError> {
    if form.source_semantic_digest.hex != crate::semantic::semantic_arena_digest(module) {
        return Err(RequirementInferenceError::SourceMismatch {
            form: form.name.clone(),
        });
    }
    let model = module
        .models
        .iter()
        .find(|model| model.name == form.model)
        .ok_or_else(|| RequirementInferenceError::MissingModel(form.model.clone()))?;
    let bindings = BindingCatalog::new(model, form);

    let mut spaces = infer_spaces(model, form)?;
    let mut integral_groups = infer_integral_groups(model, form, &bindings, &mut spaces)?;
    spaces.sort_by_key(|space| space.symbol);
    integral_groups
        .sort_by(|left, right| left.signature_digest.hex.cmp(&right.signature_digest.hex));

    let elements = spaces
        .iter()
        .map(|requirement| {
            let dimension = model
                .domains
                .get(requirement.domain.index())
                .ok_or(RequirementInferenceError::InvalidDomain(requirement.domain))?
                .spatial_dimension;
            Ok(ElementRequirement {
                symbol: requirement.symbol,
                domain: requirement.domain,
                topological_dimension: dimension,
                family: element_family(&requirement.space.family),
                polynomial_order: requirement.space.order,
                value_shape: requirement.value_shape.clone(),
            })
        })
        .collect::<Result<Vec<_>, RequirementInferenceError>>()?;
    let (essential_constraints, boundary_partitions) =
        infer_boundary_requirements(model, form, &spaces)?;
    let argument_spaces = spaces
        .iter()
        .filter(|space| matches!(space.role, SpaceBindingRole::Test | SpaceBindingRole::Trial))
        .map(|space| space.symbol)
        .collect::<Vec<_>>();
    let coefficient_spaces = spaces
        .iter()
        .filter(|space| matches!(space.role, SpaceBindingRole::PhysicalField(_)))
        .map(|space| space.symbol)
        .collect::<Vec<_>>();
    let space_system = SpaceSystemRequirement {
        composition: match argument_spaces.len() {
            0 => SpaceComposition::Empty,
            1 => SpaceComposition::Single,
            _ => SpaceComposition::Product,
        },
        argument_spaces,
        coefficient_spaces,
    };
    let receipt = RequirementInferenceReceipt {
        source_form_digest: form.artifact_digest.clone(),
        source_declaration: form.declaration,
        method: RequirementInferenceMethod::TypedFormFc3,
    };
    // Provenance indices and the parent digest are deliberately excluded: this identity names the
    // inferred mathematical requirements and therefore survives integral source reordering.
    let artifact_digest = span_independent_digest(&CanonicalRequirementsPayload {
        schema: FORM_REQUIREMENTS_SCHEMA,
        model: &form.model,
        form: &form.name,
        spaces: canonical_spaces(&spaces, &bindings),
        groups: canonical_groups(&integral_groups),
        essential_constraints: canonical_constraints(&essential_constraints, &bindings),
        boundary_partitions: canonical_partitions(&boundary_partitions, &bindings),
    });

    Ok(FormRequirements {
        schema: FORM_REQUIREMENTS_SCHEMA.into(),
        model: form.model.clone(),
        form: form.name.clone(),
        artifact_digest,
        space_system,
        spaces,
        elements,
        integral_groups,
        essential_constraints,
        boundary_partitions,
        receipt,
    })
}

fn infer_spaces(
    model: &SemanticModel,
    form: &VariationalForm,
) -> Result<Vec<SpaceRequirement>, RequirementInferenceError> {
    let mut spaces = Vec::new();
    for argument in &form.arguments {
        spaces.push(space_requirement(
            model,
            argument.symbol,
            match argument.role {
                FormArgumentRole::Test => SpaceBindingRole::Test,
                FormArgumentRole::Trial => SpaceBindingRole::Trial,
            },
            argument.domain,
            &argument.ty,
            &argument.space,
        )?);
    }
    for capture in &form.captures {
        let (Some(domain), Some(space), FormCaptureRole::PhysicalField(role)) =
            (capture.domain, capture.space.as_ref(), &capture.role)
        else {
            continue;
        };
        spaces.push(space_requirement(
            model,
            capture.symbol,
            SpaceBindingRole::PhysicalField(role.clone()),
            domain,
            &capture.ty,
            space,
        )?);
    }
    let mut transitive_fields = BTreeSet::new();
    for capture in &form.captures {
        collect_definition_fields(
            model,
            capture.symbol,
            &mut BTreeSet::new(),
            &mut transitive_fields,
        )?;
    }
    let existing = spaces
        .iter()
        .map(|requirement| requirement.symbol)
        .collect::<BTreeSet<_>>();
    for symbol_id in transitive_fields.difference(&existing) {
        let symbol = model
            .symbols
            .get(symbol_id.index())
            .filter(|symbol| symbol.id == *symbol_id)
            .ok_or(RequirementInferenceError::InvalidSymbol(*symbol_id))?;
        let SemanticRole::PhysicalField(role) = &symbol.ty.role else {
            continue;
        };
        let (Some(domain), Some(space)) = (symbol.domain, symbol.space.as_ref()) else {
            return Err(RequirementInferenceError::InvalidSymbol(*symbol_id));
        };
        spaces.push(space_requirement(
            model,
            *symbol_id,
            SpaceBindingRole::PhysicalField(role.clone()),
            domain,
            &symbol.ty,
            space,
        )?);
    }
    Ok(spaces)
}

fn collect_definition_fields(
    model: &SemanticModel,
    symbol: SymbolId,
    visited: &mut BTreeSet<SymbolId>,
    fields: &mut BTreeSet<SymbolId>,
) -> Result<(), RequirementInferenceError> {
    if !visited.insert(symbol) {
        return Ok(());
    }
    let semantic_symbol = model
        .symbols
        .get(symbol.index())
        .filter(|candidate| candidate.id == symbol)
        .ok_or(RequirementInferenceError::InvalidSymbol(symbol))?;
    if matches!(semantic_symbol.ty.role, SemanticRole::PhysicalField(_)) {
        fields.insert(symbol);
        return Ok(());
    }
    let definition = model
        .declarations
        .iter()
        .find(|declaration| declaration.symbol == Some(symbol))
        .and_then(|declaration| match declaration.kind {
            SemanticDeclarationKind::Value { value: Some(value) }
            | SemanticDeclarationKind::Property { value }
            | SemanticDeclarationKind::ConstitutiveLaw { value } => Some(value),
            _ => None,
        });
    let Some(definition) = definition else {
        return Ok(());
    };
    let mut dependencies = BTreeSet::new();
    collect_symbols(&model.expressions, definition, &mut dependencies)?;
    for dependency in dependencies {
        collect_definition_fields(model, dependency, visited, fields)?;
    }
    Ok(())
}

fn space_requirement(
    model: &SemanticModel,
    symbol: SymbolId,
    role: SpaceBindingRole,
    domain: DomainId,
    ty: &SemanticType,
    space: &SpaceSpec,
) -> Result<SpaceRequirement, RequirementInferenceError> {
    let dimension = model
        .domains
        .get(domain.index())
        .ok_or(RequirementInferenceError::InvalidDomain(domain))?
        .spatial_dimension;
    let expected = match space.family {
        SpaceFamily::H1 => Continuity::Continuous,
        SpaceFamily::L2 | SpaceFamily::Dg => Continuity::Discontinuous,
        SpaceFamily::HCurl => Continuity::Tangential,
        SpaceFamily::HDiv => Continuity::Normal,
    };
    if space.continuity != expected {
        return Err(RequirementInferenceError::IncompatibleContinuity {
            symbol,
            family: space.family.clone(),
            continuity: space.continuity.clone(),
        });
    }
    let SemanticShape::Numeric(value_shape) = &ty.shape else {
        return Err(RequirementInferenceError::IncompatibleSpaceShape {
            symbol,
            family: space.family.clone(),
            dimension,
            shape: ty.shape.clone(),
        });
    };
    if matches!(space.family, SpaceFamily::HCurl | SpaceFamily::HDiv)
        && *value_shape != ValueShape::Vector(dimension)
    {
        return Err(RequirementInferenceError::IncompatibleSpaceShape {
            symbol,
            family: space.family.clone(),
            dimension,
            shape: ty.shape.clone(),
        });
    }
    let pullback = match space.family {
        SpaceFamily::H1 => PullbackRequirement::H1Composition,
        SpaceFamily::L2 => PullbackRequirement::L2Density,
        SpaceFamily::HCurl => PullbackRequirement::CovariantPiola,
        SpaceFamily::HDiv => PullbackRequirement::ContravariantPiola,
        SpaceFamily::Dg => PullbackRequirement::Broken,
    };
    let orientations = match space.family {
        SpaceFamily::HCurl => vec![
            OrientationRequirement::EdgeTangential,
            OrientationRequirement::FacetTangential,
        ],
        SpaceFamily::HDiv => vec![OrientationRequirement::FacetNormal],
        _ => vec![],
    };
    Ok(SpaceRequirement {
        symbol,
        role,
        domain,
        value_shape: value_shape.clone(),
        space: space.clone(),
        pullback,
        orientations,
        traces: vec![],
    })
}

fn infer_integral_groups(
    model: &SemanticModel,
    form: &VariationalForm,
    bindings: &BindingCatalog,
    spaces: &mut [SpaceRequirement],
) -> Result<Vec<NormalizedIntegralGroup>, RequirementInferenceError> {
    let mut groups = BTreeMap::<String, NormalizedIntegralGroup>::new();
    for (integral_index, integral) in form.integrals.iter().enumerate() {
        let expression = expression(&form.expressions, integral.integrand)?;
        if !matches!(
            expression.ty.shape,
            SemanticShape::Numeric(ValueShape::Scalar) | SemanticShape::Deferred
        ) || !expression.ty.axes.is_empty()
        {
            return Err(RequirementInferenceError::NonScalarIntegrand {
                integral: integral_index,
                shape: expression.ty.shape.clone(),
                axes: expression.ty.axes.len(),
            });
        }
        let domain = measure_domain(model, form, integral_index, &integral.measure)?;
        let measure = MeasureRequirement {
            measure: integral.measure.clone(),
            domain,
        };
        let default_site = site_for_measure(&integral.measure);
        let mut inputs = BTreeMap::<SymbolId, BTreeSet<BasisEvaluationRequirement>>::new();
        collect_evaluations(
            &form.expressions,
            integral.integrand,
            DerivativeEvaluation::Value,
            default_site,
            None,
            bindings,
            &mut inputs,
        )?;
        complete_trace_mappings(&mut inputs, spaces);
        for (symbol, evaluations) in &inputs {
            let binding = bindings
                .get(*symbol)
                .ok_or(RequirementInferenceError::InvalidSymbol(*symbol))?;
            if let Some(binding_domain) = binding.domain
                && binding_domain != domain
            {
                return Err(RequirementInferenceError::IncompatibleMeasureDomains {
                    integral: integral_index,
                    domains: sorted_domains([domain, binding_domain]),
                });
            }
            if let Some(space) = spaces.iter_mut().find(|space| space.symbol == *symbol) {
                validate_evaluations(space, evaluations)?;
                add_trace_requirements(space, evaluations);
            }
        }
        let inputs = inputs
            .into_iter()
            .map(|(symbol, evaluations)| {
                Ok(InputPreprocessingRequirement {
                    symbol,
                    source: bindings.input_source(symbol)?,
                    evaluations: evaluations.into_iter().collect(),
                })
            })
            .collect::<Result<Vec<_>, RequirementInferenceError>>()?;
        let geometry = geometry_requirements(&integral.measure, &inputs, spaces);
        let degree = polynomial_degree(&form.expressions, integral.integrand, bindings)?;
        let quadrature = QuadratureIntent {
            minimum_polynomial_degree: degree.degree,
            precision: if degree.coefficient_dependent {
                QuadraturePrecision::CoefficientDependent
            } else {
                QuadraturePrecision::PolynomialExact
            },
        };
        let normalized = normalize_expression(&form.expressions, integral.integrand, bindings)?;
        let integrand_digest = span_independent_digest(&normalized);
        let signature = KernelSignature {
            measure,
            output_type: expression.ty.clone(),
            inputs,
            geometry,
            quadrature,
            integrand_digest,
        };
        let signature_digest = canonical_signature_digest(&signature, bindings)?;
        let occurrence = IntegralOccurrence {
            integral_index,
            source_span: integral.source_span,
        };
        if let Some(group) = groups.get_mut(&signature_digest.hex) {
            group.occurrences.push(occurrence);
        } else {
            groups.insert(
                signature_digest.hex.clone(),
                NormalizedIntegralGroup {
                    signature_digest,
                    signature,
                    occurrences: vec![occurrence],
                },
            );
        }
    }
    Ok(groups.into_values().collect())
}

fn validate_evaluations(
    space: &SpaceRequirement,
    evaluations: &BTreeSet<BasisEvaluationRequirement>,
) -> Result<(), RequirementInferenceError> {
    for evaluation in evaluations {
        let supported = match evaluation.derivative {
            DerivativeEvaluation::Value | DerivativeEvaluation::TimeDerivative => true,
            DerivativeEvaluation::Gradient | DerivativeEvaluation::SymmetricGradient => {
                matches!(space.space.family, SpaceFamily::H1 | SpaceFamily::Dg)
            }
            DerivativeEvaluation::Divergence => matches!(
                space.space.family,
                SpaceFamily::H1 | SpaceFamily::HDiv | SpaceFamily::Dg
            ),
            DerivativeEvaluation::Curl => matches!(
                space.space.family,
                SpaceFamily::H1 | SpaceFamily::HCurl | SpaceFamily::Dg
            ),
            DerivativeEvaluation::RotatedGradient => {
                matches!(space.space.family, SpaceFamily::H1 | SpaceFamily::Dg)
            }
        };
        if !supported {
            return Err(RequirementInferenceError::IncompatibleDifferentialSpace {
                symbol: space.symbol,
                family: space.space.family.clone(),
                operator: differential_operator(evaluation.derivative),
            });
        }
        if evaluation.trace_mapping == Some(TraceMapping::Normal)
            && matches!(space.space.family, SpaceFamily::HCurl | SpaceFamily::L2)
        {
            return Err(RequirementInferenceError::IncompatibleNormalTrace {
                symbol: space.symbol,
                family: space.space.family.clone(),
            });
        }
    }
    Ok(())
}

fn complete_trace_mappings(
    inputs: &mut BTreeMap<SymbolId, BTreeSet<BasisEvaluationRequirement>>,
    spaces: &[SpaceRequirement],
) {
    for (symbol, evaluations) in inputs {
        let natural = spaces
            .iter()
            .find(|space| space.symbol == *symbol)
            .map_or(TraceMapping::Value, |space| {
                trace_mapping(&space.space.family)
            });
        *evaluations = evaluations
            .iter()
            .map(|evaluation| BasisEvaluationRequirement {
                derivative: evaluation.derivative,
                site: evaluation.site,
                trace_mapping: evaluation.trace_mapping.or_else(|| {
                    matches!(
                        evaluation.site,
                        EvaluationSite::ExteriorTrace
                            | EvaluationSite::MinusTrace
                            | EvaluationSite::PlusTrace
                    )
                    .then_some(natural)
                }),
            })
            .collect();
    }
}

fn differential_operator(evaluation: DerivativeEvaluation) -> DifferentialOperator {
    match evaluation {
        DerivativeEvaluation::Gradient => DifferentialOperator::Gradient,
        DerivativeEvaluation::Divergence => DifferentialOperator::Divergence,
        DerivativeEvaluation::Curl => DifferentialOperator::Curl,
        DerivativeEvaluation::RotatedGradient => DifferentialOperator::RotatedGradient,
        DerivativeEvaluation::SymmetricGradient => DifferentialOperator::SymmetricGradient,
        DerivativeEvaluation::TimeDerivative => DifferentialOperator::TimeDerivative,
        DerivativeEvaluation::Value => DifferentialOperator::Gradient,
    }
}

fn add_trace_requirements(
    space: &mut SpaceRequirement,
    evaluations: &BTreeSet<BasisEvaluationRequirement>,
) {
    let mut traces = space.traces.iter().copied().collect::<BTreeSet<_>>();
    for evaluation in evaluations {
        let mapping = evaluation
            .trace_mapping
            .unwrap_or_else(|| trace_mapping(&space.space.family));
        let requirement = match (evaluation.site, mapping) {
            (EvaluationSite::ExteriorTrace, TraceMapping::Value) => TraceRequirement::ExteriorValue,
            (EvaluationSite::MinusTrace, TraceMapping::Value) => TraceRequirement::MinusValue,
            (EvaluationSite::PlusTrace, TraceMapping::Value) => TraceRequirement::PlusValue,
            (EvaluationSite::ExteriorTrace, TraceMapping::Tangential) => {
                TraceRequirement::ExteriorTangential
            }
            (EvaluationSite::MinusTrace, TraceMapping::Tangential) => {
                TraceRequirement::MinusTangential
            }
            (EvaluationSite::PlusTrace, TraceMapping::Tangential) => {
                TraceRequirement::PlusTangential
            }
            (EvaluationSite::ExteriorTrace, TraceMapping::Normal) => {
                TraceRequirement::ExteriorNormal
            }
            (EvaluationSite::MinusTrace, TraceMapping::Normal) => TraceRequirement::MinusNormal,
            (EvaluationSite::PlusTrace, TraceMapping::Normal) => TraceRequirement::PlusNormal,
            (EvaluationSite::Cell | EvaluationSite::Point, _) => continue,
        };
        traces.insert(requirement);
    }
    if evaluations.iter().any(|evaluation| {
        matches!(
            evaluation.site,
            EvaluationSite::MinusTrace | EvaluationSite::PlusTrace
        )
    }) {
        let mut orientations = space.orientations.iter().copied().collect::<BTreeSet<_>>();
        orientations.insert(OrientationRequirement::TwoSidedFacet);
        space.orientations = orientations.into_iter().collect();
    }
    space.traces = traces.into_iter().collect();
}

fn collect_evaluations(
    expressions: &[SemanticExpr],
    id: ExprId,
    derivative: DerivativeEvaluation,
    site: EvaluationSite,
    trace_mapping_override: Option<TraceMapping>,
    bindings: &BindingCatalog<'_>,
    inputs: &mut BTreeMap<SymbolId, BTreeSet<BasisEvaluationRequirement>>,
) -> Result<(), RequirementInferenceError> {
    collect_evaluations_inner(
        expressions,
        id,
        derivative,
        site,
        trace_mapping_override,
        bindings,
        &mut BTreeSet::new(),
        inputs,
    )
}

#[allow(clippy::too_many_arguments)]
fn collect_evaluations_inner(
    expressions: &[SemanticExpr],
    id: ExprId,
    derivative: DerivativeEvaluation,
    site: EvaluationSite,
    trace_mapping_override: Option<TraceMapping>,
    bindings: &BindingCatalog<'_>,
    expanding: &mut BTreeSet<SymbolId>,
    inputs: &mut BTreeMap<SymbolId, BTreeSet<BasisEvaluationRequirement>>,
) -> Result<(), RequirementInferenceError> {
    let kind = &expression(expressions, id)?.kind;
    match kind {
        SemanticExprKind::Symbol { symbol } => {
            inputs
                .entry(*symbol)
                .or_default()
                .insert(BasisEvaluationRequirement {
                    derivative,
                    site,
                    trace_mapping: trace_mapping_override,
                });
            if let Some(definition) = bindings.get(*symbol).and_then(|binding| binding.definition)
                && expanding.insert(*symbol)
            {
                let definition = match definition {
                    BindingDefinition::Value(definition)
                    | BindingDefinition::Property(definition)
                    | BindingDefinition::ConstitutiveLaw(definition) => definition,
                };
                collect_evaluations_inner(
                    expressions,
                    definition,
                    derivative,
                    site,
                    trace_mapping_override,
                    bindings,
                    expanding,
                    inputs,
                )?;
                expanding.remove(symbol);
            }
        }
        SemanticExprKind::Unary { arg, .. }
        | SemanticExprKind::TensorTrace { value: arg, .. }
        | SemanticExprKind::Conjugate { value: arg } => collect_evaluations_inner(
            expressions,
            *arg,
            derivative,
            site,
            trace_mapping_override,
            bindings,
            expanding,
            inputs,
        )?,
        SemanticExprKind::Differential { operator, arg } => collect_evaluations_inner(
            expressions,
            *arg,
            derivative_evaluation(*operator),
            site,
            trace_mapping_override,
            bindings,
            expanding,
            inputs,
        )?,
        SemanticExprKind::FacetTrace { value, side } => collect_evaluations_inner(
            expressions,
            *value,
            derivative,
            site_for_trace(*side),
            trace_mapping_override,
            bindings,
            expanding,
            inputs,
        )?,
        SemanticExprKind::Jump { value } | SemanticExprKind::Average { value } => {
            collect_evaluations_inner(
                expressions,
                *value,
                derivative,
                EvaluationSite::MinusTrace,
                trace_mapping_override,
                bindings,
                expanding,
                inputs,
            )?;
            collect_evaluations_inner(
                expressions,
                *value,
                derivative,
                EvaluationSite::PlusTrace,
                trace_mapping_override,
                bindings,
                expanding,
                inputs,
            )?;
        }
        SemanticExprKind::NormalComponent { value, side } => collect_evaluations_inner(
            expressions,
            *value,
            derivative,
            site_for_trace(*side),
            Some(TraceMapping::Normal),
            bindings,
            expanding,
            inputs,
        )?,
        SemanticExprKind::Binary { lhs, rhs, .. }
        | SemanticExprKind::Contraction { lhs, rhs, .. } => {
            collect_evaluations_inner(
                expressions,
                *lhs,
                derivative,
                site,
                trace_mapping_override,
                bindings,
                expanding,
                inputs,
            )?;
            collect_evaluations_inner(
                expressions,
                *rhs,
                derivative,
                site,
                trace_mapping_override,
                bindings,
                expanding,
                inputs,
            )?;
        }
        SemanticExprKind::Call { args, .. }
        | SemanticExprKind::ProviderCall { args, .. }
        | SemanticExprKind::Vector { elements: args } => {
            for arg in args {
                collect_evaluations_inner(
                    expressions,
                    *arg,
                    derivative,
                    site,
                    trace_mapping_override,
                    bindings,
                    expanding,
                    inputs,
                )?;
            }
        }
        SemanticExprKind::Index { value, indices } => {
            collect_evaluations_inner(
                expressions,
                *value,
                derivative,
                site,
                trace_mapping_override,
                bindings,
                expanding,
                inputs,
            )?;
            for index in indices {
                collect_evaluations_inner(
                    expressions,
                    *index,
                    DerivativeEvaluation::Value,
                    site,
                    trace_mapping_override,
                    bindings,
                    expanding,
                    inputs,
                )?;
            }
        }
        SemanticExprKind::Number { .. } | SemanticExprKind::String { .. } => {}
    }
    Ok(())
}

fn infer_boundary_requirements(
    model: &SemanticModel,
    form: &VariationalForm,
    spaces: &[SpaceRequirement],
) -> Result<
    (
        Vec<EssentialConstraintRequirement>,
        Vec<BoundaryPartitionRequirement>,
    ),
    RequirementInferenceError,
> {
    let mut constraints = Vec::new();
    let mut partitions = Vec::new();
    for assumption in &form.receipt.assumptions {
        match assumption {
            FormAssumption::ExteriorRegionsPartitionBoundary {
                domain, regions, ..
            } => {
                model
                    .domains
                    .get(domain.index())
                    .ok_or(RequirementInferenceError::InvalidDomain(*domain))?;
                let mut regions = regions.clone();
                regions.sort_unstable();
                regions.dedup();
                for region in &regions {
                    validate_region(model, *region, RegionKind::ExteriorFacet, Some(*domain))?;
                }
                partitions.push(BoundaryPartitionRequirement {
                    domain: *domain,
                    exterior_regions: regions,
                });
            }
            FormAssumption::TestTraceVanishes {
                argument,
                region,
                condition,
            } => {
                let argument_space = spaces
                    .iter()
                    .find(|space| space.symbol == *argument)
                    .ok_or(RequirementInferenceError::InvalidSymbol(*argument))?;
                validate_region(
                    model,
                    *region,
                    RegionKind::ExteriorFacet,
                    Some(argument_space.domain),
                )?;
                let declaration = model
                    .declarations
                    .get(condition.index())
                    .filter(|declaration| declaration.id == *condition)
                    .ok_or(RequirementInferenceError::IncompatibleBoundaryData {
                        condition: *condition,
                        region: *region,
                    })?;
                let SemanticDeclarationKind::BoundaryCondition {
                    region: condition_region,
                    target,
                    condition: BoundaryConditionKind::Dirichlet,
                    ..
                } = &declaration.kind
                else {
                    return Err(RequirementInferenceError::IncompatibleBoundaryData {
                        condition: *condition,
                        region: *region,
                    });
                };
                let compatible_target = target.and_then(|symbol| model.symbols.get(symbol.index()));
                if *condition_region != *region
                    || compatible_target.is_none_or(|target| {
                        target.domain != Some(argument_space.domain)
                            || target.space.as_ref() != Some(&argument_space.space)
                            || target.ty.shape
                                != SemanticShape::Numeric(argument_space.value_shape.clone())
                    })
                {
                    return Err(RequirementInferenceError::IncompatibleBoundaryData {
                        condition: *condition,
                        region: *region,
                    });
                }
                constraints.push(EssentialConstraintRequirement {
                    argument: *argument,
                    region: *region,
                    condition: *condition,
                });
            }
        }
    }
    constraints.sort_by_key(|item| (item.argument, item.region, item.condition));
    partitions.sort_by_key(|item| item.domain);
    Ok((constraints, partitions))
}

fn measure_domain(
    model: &SemanticModel,
    form: &VariationalForm,
    integral_index: usize,
    measure: &SemanticMeasure,
) -> Result<DomainId, RequirementInferenceError> {
    if let SemanticMeasure::Cell { domain } = measure {
        model
            .domains
            .get(domain.index())
            .ok_or(RequirementInferenceError::InvalidDomain(*domain))?;
        return Ok(*domain);
    }
    let (region_id, expected_kind) = match measure {
        SemanticMeasure::ExteriorFacet { region } => (*region, RegionKind::ExteriorFacet),
        SemanticMeasure::InteriorFacet { region } => (*region, RegionKind::InteriorFacet),
        SemanticMeasure::Interface { region } => (*region, RegionKind::Interface),
        SemanticMeasure::Point { region } => (*region, RegionKind::Point),
        SemanticMeasure::Cell { .. } => unreachable!(),
    };
    let (actual_kind, region_domain) = resolve_region(model, region_id)?;
    if actual_kind != expected_kind {
        return Err(RequirementInferenceError::RegionKindMismatch {
            region: region_id,
            expected: expected_kind,
            actual: actual_kind,
        });
    }
    let mut domains = BTreeSet::new();
    if let Some(domain) = region_domain {
        domains.insert(domain);
    }
    let mut symbols = BTreeSet::new();
    collect_symbols(
        &form.expressions,
        form.integrals[integral_index].integrand,
        &mut symbols,
    )?;
    for symbol in symbols {
        if let Some(domain) = form
            .arguments
            .iter()
            .find(|binding| binding.symbol == symbol)
            .map(|binding| binding.domain)
            .or_else(|| {
                form.captures
                    .iter()
                    .find(|binding| binding.symbol == symbol)
                    .and_then(|binding| binding.domain)
            })
        {
            domains.insert(domain);
        }
    }
    match domains.len() {
        0 => Err(RequirementInferenceError::MissingMeasureDomain {
            integral: integral_index,
        }),
        1 => Ok(*domains.iter().next().expect("one domain")),
        _ => Err(RequirementInferenceError::IncompatibleMeasureDomains {
            integral: integral_index,
            domains: domains.into_iter().collect(),
        }),
    }
}

fn validate_region(
    model: &SemanticModel,
    region: RegionId,
    expected: RegionKind,
    domain: Option<DomainId>,
) -> Result<(), RequirementInferenceError> {
    let (actual_kind, actual_domain) = resolve_region(model, region)?;
    if actual_kind != expected {
        return Err(RequirementInferenceError::RegionKindMismatch {
            region,
            expected,
            actual: actual_kind,
        });
    }
    if let (Some(expected_domain), Some(actual_domain)) = (domain, actual_domain)
        && expected_domain != actual_domain
    {
        return Err(RequirementInferenceError::IncompatibleMeasureDomains {
            integral: 0,
            domains: sorted_domains([expected_domain, actual_domain]),
        });
    }
    Ok(())
}

/// Resolve a region's kind and domain, whether it is a real declared region or a GX-A6
/// compiler-synthesized implicit natural-boundary region (see `RegionId::generated_for`); a
/// generated id is always `ExteriorFacet` with its domain encoded directly in the id, so it
/// never needs an arena lookup.
fn resolve_region(
    model: &SemanticModel,
    region: RegionId,
) -> Result<(RegionKind, Option<DomainId>), RequirementInferenceError> {
    if let Some(domain) = region.generated_domain() {
        return Ok((RegionKind::ExteriorFacet, Some(domain)));
    }
    let region = model
        .regions
        .get(region.index())
        .filter(|candidate| candidate.id == region)
        .ok_or(RequirementInferenceError::InvalidRegion(region))?;
    Ok((region.kind.clone(), region.domain))
}

fn geometry_requirements(
    measure: &SemanticMeasure,
    inputs: &[InputPreprocessingRequirement],
    spaces: &[SpaceRequirement],
) -> Vec<GeometryPreprocessingRequirement> {
    let mut geometry = BTreeSet::new();
    match measure {
        SemanticMeasure::Cell { .. } => {
            geometry.insert(GeometryPreprocessingRequirement::JacobianDeterminant);
        }
        SemanticMeasure::ExteriorFacet { .. }
        | SemanticMeasure::InteriorFacet { .. }
        | SemanticMeasure::Interface { .. } => {
            geometry.insert(GeometryPreprocessingRequirement::FacetJacobian);
        }
        SemanticMeasure::Point { .. } => {
            geometry.insert(GeometryPreprocessingRequirement::PointLocation);
        }
    }
    for input in inputs {
        let pullback = spaces
            .iter()
            .find(|space| space.symbol == input.symbol)
            .map(|space| space.pullback);
        if input.evaluations.iter().any(|evaluation| {
            !matches!(
                evaluation.derivative,
                DerivativeEvaluation::Value | DerivativeEvaluation::TimeDerivative
            )
        }) || matches!(
            pullback,
            Some(PullbackRequirement::CovariantPiola | PullbackRequirement::ContravariantPiola)
        ) {
            geometry.insert(GeometryPreprocessingRequirement::Jacobian);
            geometry.insert(GeometryPreprocessingRequirement::InverseJacobian);
        }
        if matches!(
            pullback,
            Some(PullbackRequirement::ContravariantPiola | PullbackRequirement::L2Density)
        ) {
            geometry.insert(GeometryPreprocessingRequirement::JacobianDeterminant);
        }
        if input
            .evaluations
            .iter()
            .any(|evaluation| evaluation.trace_mapping == Some(TraceMapping::Normal))
        {
            geometry.insert(GeometryPreprocessingRequirement::FacetNormal);
        }
    }
    geometry.into_iter().collect()
}

#[derive(Clone, Copy)]
struct PolynomialDegree {
    degree: u16,
    coefficient_dependent: bool,
}

fn polynomial_degree(
    expressions: &[SemanticExpr],
    id: ExprId,
    bindings: &BindingCatalog,
) -> Result<PolynomialDegree, RequirementInferenceError> {
    polynomial_degree_inner(expressions, id, bindings, &mut BTreeSet::new())
}

fn polynomial_degree_inner(
    expressions: &[SemanticExpr],
    id: ExprId,
    bindings: &BindingCatalog,
    expanding: &mut BTreeSet<SymbolId>,
) -> Result<PolynomialDegree, RequirementInferenceError> {
    let expression = expression(expressions, id)?;
    let combine_max = |left: PolynomialDegree, right: PolynomialDegree| PolynomialDegree {
        degree: left.degree.max(right.degree),
        coefficient_dependent: left.coefficient_dependent || right.coefficient_dependent,
    };
    let combine_product = |left: PolynomialDegree, right: PolynomialDegree| PolynomialDegree {
        degree: left.degree.saturating_add(right.degree),
        coefficient_dependent: left.coefficient_dependent || right.coefficient_dependent,
    };
    Ok(match &expression.kind {
        SemanticExprKind::Number { .. } => PolynomialDegree {
            degree: 0,
            coefficient_dependent: false,
        },
        SemanticExprKind::String { .. } => PolynomialDegree {
            degree: 0,
            coefficient_dependent: true,
        },
        SemanticExprKind::Symbol { symbol } => {
            let binding = bindings
                .get(*symbol)
                .ok_or(RequirementInferenceError::InvalidSymbol(*symbol))?;
            if let Some(definition) = binding.definition
                && expanding.insert(*symbol)
            {
                let definition = match definition {
                    BindingDefinition::Value(definition)
                    | BindingDefinition::Property(definition)
                    | BindingDefinition::ConstitutiveLaw(definition) => definition,
                };
                let degree = polynomial_degree_inner(expressions, definition, bindings, expanding)?;
                expanding.remove(symbol);
                degree
            } else {
                PolynomialDegree {
                    degree: binding
                        .space
                        .as_ref()
                        .map_or(0, |space| u16::from(space.order)),
                    coefficient_dependent: binding.space.is_none()
                        && !matches!(binding.role, BindingRole::Parameter | BindingRole::Constant),
                }
            }
        }
        SemanticExprKind::Unary { arg, .. }
        | SemanticExprKind::TensorTrace { value: arg, .. }
        | SemanticExprKind::FacetTrace { value: arg, .. }
        | SemanticExprKind::Jump { value: arg }
        | SemanticExprKind::Average { value: arg }
        | SemanticExprKind::Conjugate { value: arg }
        | SemanticExprKind::NormalComponent { value: arg, .. }
        | SemanticExprKind::Index { value: arg, .. } => {
            polynomial_degree_inner(expressions, *arg, bindings, expanding)?
        }
        SemanticExprKind::Differential { operator, arg } => {
            let mut degree = polynomial_degree_inner(expressions, *arg, bindings, expanding)?;
            if *operator != DifferentialOperator::TimeDerivative {
                degree.degree = degree.degree.saturating_sub(1);
            }
            degree
        }
        SemanticExprKind::Binary { op, lhs, rhs } => {
            let left = polynomial_degree_inner(expressions, *lhs, bindings, expanding)?;
            let right = polynomial_degree_inner(expressions, *rhs, bindings, expanding)?;
            match op {
                BinaryOp::Add | BinaryOp::Sub => combine_max(left, right),
                BinaryOp::Mul => combine_product(left, right),
                BinaryOp::Div if right.degree == 0 && !right.coefficient_dependent => left,
                BinaryOp::Pow => PolynomialDegree {
                    degree: left.degree.max(right.degree),
                    coefficient_dependent: true,
                },
                BinaryOp::Div
                | BinaryOp::Eq
                | BinaryOp::Lt
                | BinaryOp::Le
                | BinaryOp::Gt
                | BinaryOp::Ge => PolynomialDegree {
                    degree: left.degree.max(right.degree),
                    coefficient_dependent: true,
                },
            }
        }
        SemanticExprKind::Contraction { lhs, rhs, .. } => combine_product(
            polynomial_degree_inner(expressions, *lhs, bindings, expanding)?,
            polynomial_degree_inner(expressions, *rhs, bindings, expanding)?,
        ),
        SemanticExprKind::Call { args, .. } | SemanticExprKind::ProviderCall { args, .. } => {
            args.iter().try_fold(
                PolynomialDegree {
                    degree: 0,
                    coefficient_dependent: true,
                },
                |degree, arg| {
                    Ok::<_, RequirementInferenceError>(combine_max(
                        degree,
                        polynomial_degree_inner(expressions, *arg, bindings, expanding)?,
                    ))
                },
            )?
        }
        SemanticExprKind::Vector { elements } => elements.iter().try_fold(
            PolynomialDegree {
                degree: 0,
                coefficient_dependent: false,
            },
            |degree, element| {
                Ok::<_, RequirementInferenceError>(combine_max(
                    degree,
                    polynomial_degree_inner(expressions, *element, bindings, expanding)?,
                ))
            },
        )?,
    })
}

#[derive(Serialize)]
struct CanonicalRequirementsPayload<'a> {
    schema: &'static str,
    model: &'a str,
    form: &'a str,
    spaces: Vec<CanonicalSpace>,
    groups: Vec<(String, usize)>,
    essential_constraints: Vec<(String, String, String)>,
    boundary_partitions: Vec<(String, Vec<String>)>,
}

#[derive(Serialize)]
struct CanonicalSpace {
    binding: String,
    role: SpaceBindingRole,
    domain: String,
    shape: ValueShape,
    space: SpaceSpec,
    pullback: PullbackRequirement,
    orientations: Vec<OrientationRequirement>,
    traces: Vec<TraceRequirement>,
}

fn canonical_spaces(spaces: &[SpaceRequirement], bindings: &BindingCatalog) -> Vec<CanonicalSpace> {
    let mut canonical = spaces
        .iter()
        .map(|space| CanonicalSpace {
            binding: bindings.key(space.symbol),
            role: space.role.clone(),
            domain: bindings.domain_key(space.domain),
            shape: space.value_shape.clone(),
            space: space.space.clone(),
            pullback: space.pullback,
            orientations: space.orientations.clone(),
            traces: space.traces.clone(),
        })
        .collect::<Vec<_>>();
    canonical.sort_by(|left, right| left.binding.cmp(&right.binding));
    canonical
}

fn canonical_groups(groups: &[NormalizedIntegralGroup]) -> Vec<(String, usize)> {
    let mut groups = groups
        .iter()
        .map(|group| (group.signature_digest.hex.clone(), group.occurrences.len()))
        .collect::<Vec<_>>();
    groups.sort();
    groups
}

fn canonical_constraints(
    constraints: &[EssentialConstraintRequirement],
    bindings: &BindingCatalog,
) -> Vec<(String, String, String)> {
    let mut constraints = constraints
        .iter()
        .map(|constraint| {
            (
                bindings.key(constraint.argument),
                bindings.region_key(constraint.region),
                bindings.declaration_key(constraint.condition),
            )
        })
        .collect::<Vec<_>>();
    constraints.sort();
    constraints
}

fn canonical_partitions(
    partitions: &[BoundaryPartitionRequirement],
    bindings: &BindingCatalog,
) -> Vec<(String, Vec<String>)> {
    let mut partitions = partitions
        .iter()
        .map(|partition| {
            let mut regions = partition
                .exterior_regions
                .iter()
                .map(|region| bindings.region_key(*region))
                .collect::<Vec<_>>();
            regions.sort();
            (bindings.domain_key(partition.domain), regions)
        })
        .collect::<Vec<_>>();
    partitions.sort();
    partitions
}

#[derive(Serialize)]
struct CanonicalKernelSignature<'a> {
    measure_kind: &'static str,
    domain: String,
    region: Option<String>,
    output_type: SemanticType,
    inputs: Vec<CanonicalInput<'a>>,
    geometry: &'a [GeometryPreprocessingRequirement],
    quadrature: &'a QuadratureIntent,
    integrand_digest: &'a Digest,
}

#[derive(Serialize)]
struct CanonicalInput<'a> {
    binding: String,
    source: CanonicalInputSource,
    evaluations: &'a [BasisEvaluationRequirement],
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum CanonicalInputSource {
    Basis,
    ExternalValue,
    ModelDefinedValue { definition_digest: Digest },
    ModelDefinedProperty { definition_digest: Digest },
    ModelDefinedConstitutive { definition_digest: Digest },
}

fn canonical_signature_digest(
    signature: &KernelSignature,
    bindings: &BindingCatalog,
) -> Result<Digest, RequirementInferenceError> {
    let (measure_kind, region) = match signature.measure.measure {
        SemanticMeasure::Cell { .. } => ("cell", None),
        SemanticMeasure::ExteriorFacet { region } => {
            ("exterior_facet", Some(bindings.region_key(region)))
        }
        SemanticMeasure::InteriorFacet { region } => {
            ("interior_facet", Some(bindings.region_key(region)))
        }
        SemanticMeasure::Interface { region } => ("interface", Some(bindings.region_key(region))),
        SemanticMeasure::Point { region } => ("point", Some(bindings.region_key(region))),
    };
    let mut inputs = signature
        .inputs
        .iter()
        .map(|input| {
            let definition_digest = |definition| {
                normalize_expression(bindings.expressions, definition, bindings)
                    .map(|expression| span_independent_digest(&expression))
            };
            let source = match input.source {
                InputSourceRequirement::Basis => CanonicalInputSource::Basis,
                InputSourceRequirement::ExternalValue => CanonicalInputSource::ExternalValue,
                InputSourceRequirement::ModelDefinedValue { definition } => {
                    CanonicalInputSource::ModelDefinedValue {
                        definition_digest: definition_digest(definition)?,
                    }
                }
                InputSourceRequirement::ModelDefinedProperty { definition } => {
                    CanonicalInputSource::ModelDefinedProperty {
                        definition_digest: definition_digest(definition)?,
                    }
                }
                InputSourceRequirement::ModelDefinedConstitutive { definition } => {
                    CanonicalInputSource::ModelDefinedConstitutive {
                        definition_digest: definition_digest(definition)?,
                    }
                }
            };
            Ok(CanonicalInput {
                binding: bindings.key(input.symbol),
                source,
                evaluations: input.evaluations.as_slice(),
            })
        })
        .collect::<Result<Vec<_>, RequirementInferenceError>>()?;
    inputs.sort_by(|left, right| left.binding.cmp(&right.binding));
    let mut output_type = signature.output_type.clone();
    output_type.frame = crate::semantic::Frame::Neutral;
    Ok(span_independent_digest(&CanonicalKernelSignature {
        measure_kind,
        domain: bindings.domain_key(signature.measure.domain),
        region,
        output_type,
        inputs,
        geometry: &signature.geometry,
        quadrature: &signature.quadrature,
        integrand_digest: &signature.integrand_digest,
    }))
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum NormalizedExpression {
    Number {
        bits: u64,
        unit: Option<String>,
    },
    String {
        value: String,
    },
    Symbol {
        binding: String,
    },
    Unary {
        op: UnaryOp,
        arg: Box<Self>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Self>,
        rhs: Box<Self>,
    },
    Call {
        function: String,
        args: Vec<Self>,
    },
    /// Normalized the same shape as `Call`, keyed on the declared provider's name rather than
    /// its arena-local `ProviderId` so grouping stays name-free.
    ProviderCall {
        provider: String,
        args: Vec<Self>,
    },
    Differential {
        operator: DifferentialOperator,
        arg: Box<Self>,
    },
    Contraction {
        lhs: Box<Self>,
        rhs: Box<Self>,
        axes: Vec<AxisContraction>,
        conjugate_lhs: bool,
    },
    TensorTrace {
        value: Box<Self>,
        axes: AxisContraction,
    },
    FacetTrace {
        value: Box<Self>,
        side: TraceSide,
    },
    Jump {
        value: Box<Self>,
    },
    Average {
        value: Box<Self>,
    },
    Conjugate {
        value: Box<Self>,
    },
    NormalComponent {
        value: Box<Self>,
        side: TraceSide,
    },
    Index {
        value: Box<Self>,
        indices: Vec<Self>,
    },
    Vector {
        elements: Vec<Self>,
    },
}

fn normalize_expression(
    expressions: &[SemanticExpr],
    id: ExprId,
    bindings: &BindingCatalog,
) -> Result<NormalizedExpression, RequirementInferenceError> {
    let normalize = |id| normalize_expression(expressions, id, bindings);
    Ok(match &expression(expressions, id)?.kind {
        SemanticExprKind::Number { value, unit } => NormalizedExpression::Number {
            bits: value.to_bits(),
            unit: unit.as_ref().map(ToString::to_string),
        },
        SemanticExprKind::String { value } => NormalizedExpression::String {
            value: value.clone(),
        },
        SemanticExprKind::Symbol { symbol } => NormalizedExpression::Symbol {
            binding: bindings.key(*symbol),
        },
        SemanticExprKind::Unary { op, arg } => NormalizedExpression::Unary {
            op: *op,
            arg: Box::new(normalize(*arg)?),
        },
        SemanticExprKind::Binary { op, lhs, rhs } => NormalizedExpression::Binary {
            op: *op,
            lhs: Box::new(normalize(*lhs)?),
            rhs: Box::new(normalize(*rhs)?),
        },
        SemanticExprKind::Call { function, args } => NormalizedExpression::Call {
            function: function.clone(),
            args: args
                .iter()
                .map(|arg| normalize(*arg))
                .collect::<Result<_, _>>()?,
        },
        SemanticExprKind::ProviderCall { provider, args } => NormalizedExpression::ProviderCall {
            provider: bindings.provider_key(*provider),
            args: args
                .iter()
                .map(|arg| normalize(*arg))
                .collect::<Result<_, _>>()?,
        },
        SemanticExprKind::Differential { operator, arg } => NormalizedExpression::Differential {
            operator: *operator,
            arg: Box::new(normalize(*arg)?),
        },
        SemanticExprKind::Contraction {
            lhs,
            rhs,
            axes,
            conjugate_lhs,
        } => NormalizedExpression::Contraction {
            lhs: Box::new(normalize(*lhs)?),
            rhs: Box::new(normalize(*rhs)?),
            axes: axes.clone(),
            conjugate_lhs: *conjugate_lhs,
        },
        SemanticExprKind::TensorTrace { value, axes } => NormalizedExpression::TensorTrace {
            value: Box::new(normalize(*value)?),
            axes: *axes,
        },
        SemanticExprKind::FacetTrace { value, side } => NormalizedExpression::FacetTrace {
            value: Box::new(normalize(*value)?),
            side: *side,
        },
        SemanticExprKind::Jump { value } => NormalizedExpression::Jump {
            value: Box::new(normalize(*value)?),
        },
        SemanticExprKind::Average { value } => NormalizedExpression::Average {
            value: Box::new(normalize(*value)?),
        },
        SemanticExprKind::Conjugate { value } => NormalizedExpression::Conjugate {
            value: Box::new(normalize(*value)?),
        },
        SemanticExprKind::NormalComponent { value, side } => {
            NormalizedExpression::NormalComponent {
                value: Box::new(normalize(*value)?),
                side: *side,
            }
        }
        SemanticExprKind::Index { value, indices } => NormalizedExpression::Index {
            value: Box::new(normalize(*value)?),
            indices: indices
                .iter()
                .map(|index| normalize(*index))
                .collect::<Result<_, _>>()?,
        },
        SemanticExprKind::Vector { elements } => NormalizedExpression::Vector {
            elements: elements
                .iter()
                .map(|element| normalize(*element))
                .collect::<Result<_, _>>()?,
        },
    })
}

#[derive(Clone, Copy)]
enum BindingRole {
    PhysicalField,
    Parameter,
    Constant,
    Property,
    ConstitutiveLaw,
    Other,
}

#[derive(Clone, Copy)]
enum BindingDefinition {
    Value(ExprId),
    Property(ExprId),
    ConstitutiveLaw(ExprId),
}

struct BindingInfo<'a> {
    key: String,
    domain: Option<DomainId>,
    space: Option<&'a SpaceSpec>,
    role: BindingRole,
    definition: Option<BindingDefinition>,
}

struct BindingCatalog<'a> {
    bindings: BTreeMap<SymbolId, BindingInfo<'a>>,
    domains: BTreeMap<DomainId, String>,
    regions: BTreeMap<RegionId, String>,
    declarations: BTreeMap<DeclarationId, String>,
    providers: BTreeMap<ProviderId, String>,
    expressions: &'a [SemanticExpr],
}

impl<'a> BindingCatalog<'a> {
    fn new(model: &'a SemanticModel, form: &'a VariationalForm) -> Self {
        let names = model
            .symbols
            .iter()
            .map(|symbol| (symbol.id, symbol.name.clone()))
            .collect::<BTreeMap<_, _>>();
        let definitions = model
            .declarations
            .iter()
            .filter_map(|declaration| {
                let symbol = declaration.symbol?;
                let definition = match declaration.kind {
                    SemanticDeclarationKind::Value { value: Some(value) } => {
                        BindingDefinition::Value(value)
                    }
                    SemanticDeclarationKind::Property { value } => {
                        BindingDefinition::Property(value)
                    }
                    SemanticDeclarationKind::ConstitutiveLaw { value } => {
                        BindingDefinition::ConstitutiveLaw(value)
                    }
                    _ => return None,
                };
                Some((symbol, definition))
            })
            .collect::<BTreeMap<_, _>>();
        let mut bindings = model
            .symbols
            .iter()
            .map(|symbol| {
                let role = match symbol.ty.role {
                    SemanticRole::PhysicalField(_) => BindingRole::PhysicalField,
                    SemanticRole::Parameter => BindingRole::Parameter,
                    SemanticRole::Constant => BindingRole::Constant,
                    SemanticRole::Property => BindingRole::Property,
                    SemanticRole::ConstitutiveLaw => BindingRole::ConstitutiveLaw,
                    _ => BindingRole::Other,
                };
                (
                    symbol.id,
                    BindingInfo {
                        key: symbol.name.clone(),
                        domain: symbol.domain,
                        space: symbol.space.as_ref(),
                        role,
                        definition: definitions.get(&symbol.id).copied(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        for (index, argument) in form.arguments.iter().enumerate() {
            bindings.insert(
                argument.symbol,
                BindingInfo {
                    key: names.get(&argument.symbol).cloned().unwrap_or_else(|| {
                        format!("generated_{:?}_{index}", argument.role).to_lowercase()
                    }),
                    domain: Some(argument.domain),
                    space: Some(&argument.space),
                    role: BindingRole::PhysicalField,
                    definition: None,
                },
            );
        }
        Self {
            bindings,
            domains: model
                .domains
                .iter()
                .map(|domain| (domain.id, domain.name.clone()))
                .collect(),
            regions: model
                .regions
                .iter()
                .map(|region| (region.id, format!("{:?}:{}", region.kind, region.name)))
                .collect(),
            declarations: model
                .declarations
                .iter()
                .map(|declaration| (declaration.id, declaration.name.clone()))
                .collect(),
            providers: model
                .providers
                .iter()
                .map(|provider| (provider.id, provider.name.clone()))
                .collect(),
            expressions: &form.expressions,
        }
    }

    fn get(&self, symbol: SymbolId) -> Option<&BindingInfo<'a>> {
        self.bindings.get(&symbol)
    }

    /// Name-free normalization still needs a stable, semantic (not arena-index) key for a
    /// provider call; this mirrors `key()`'s symbol-name resolution.
    fn provider_key(&self, provider: ProviderId) -> String {
        self.providers
            .get(&provider)
            .cloned()
            .unwrap_or_else(|| format!("undeclared_provider_{}", provider.0))
    }

    fn input_source(
        &self,
        symbol: SymbolId,
    ) -> Result<InputSourceRequirement, RequirementInferenceError> {
        let binding = self
            .get(symbol)
            .ok_or(RequirementInferenceError::InvalidSymbol(symbol))?;
        Ok(match binding.definition {
            Some(BindingDefinition::Value(definition)) => {
                InputSourceRequirement::ModelDefinedValue { definition }
            }
            Some(BindingDefinition::Property(definition)) => {
                InputSourceRequirement::ModelDefinedProperty { definition }
            }
            Some(BindingDefinition::ConstitutiveLaw(definition)) => {
                InputSourceRequirement::ModelDefinedConstitutive { definition }
            }
            None if matches!(binding.role, BindingRole::PhysicalField) => {
                InputSourceRequirement::Basis
            }
            None => InputSourceRequirement::ExternalValue,
        })
    }

    fn key(&self, symbol: SymbolId) -> String {
        self.bindings
            .get(&symbol)
            .map(|binding| binding.key.clone())
            .unwrap_or_else(|| format!("unbound_{}", symbol.0))
    }

    fn domain_key(&self, domain: DomainId) -> String {
        self.domains
            .get(&domain)
            .cloned()
            .unwrap_or_else(|| format!("domain_{}", domain.0))
    }

    fn region_key(&self, region: RegionId) -> String {
        self.regions
            .get(&region)
            .cloned()
            .unwrap_or_else(|| format!("region_{}", region.0))
    }

    fn declaration_key(&self, declaration: DeclarationId) -> String {
        self.declarations
            .get(&declaration)
            .cloned()
            .unwrap_or_else(|| format!("declaration_{}", declaration.0))
    }
}

fn expression(
    expressions: &[SemanticExpr],
    id: ExprId,
) -> Result<&SemanticExpr, RequirementInferenceError> {
    expressions
        .get(id.index())
        .filter(|expression| expression.id == id)
        .ok_or(RequirementInferenceError::InvalidExpression(id))
}

fn collect_symbols(
    expressions: &[SemanticExpr],
    id: ExprId,
    symbols: &mut BTreeSet<SymbolId>,
) -> Result<(), RequirementInferenceError> {
    match &expression(expressions, id)?.kind {
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
            collect_symbols(expressions, *arg, symbols)?
        }
        SemanticExprKind::Binary { lhs, rhs, .. }
        | SemanticExprKind::Contraction { lhs, rhs, .. } => {
            collect_symbols(expressions, *lhs, symbols)?;
            collect_symbols(expressions, *rhs, symbols)?;
        }
        SemanticExprKind::Call { args, .. }
        | SemanticExprKind::ProviderCall { args, .. }
        | SemanticExprKind::Vector { elements: args } => {
            for arg in args {
                collect_symbols(expressions, *arg, symbols)?;
            }
        }
        SemanticExprKind::Index { value, indices } => {
            collect_symbols(expressions, *value, symbols)?;
            for index in indices {
                collect_symbols(expressions, *index, symbols)?;
            }
        }
        SemanticExprKind::Number { .. } | SemanticExprKind::String { .. } => {}
    }
    Ok(())
}

fn derivative_evaluation(operator: DifferentialOperator) -> DerivativeEvaluation {
    match operator {
        DifferentialOperator::Gradient => DerivativeEvaluation::Gradient,
        DifferentialOperator::Divergence => DerivativeEvaluation::Divergence,
        DifferentialOperator::Curl => DerivativeEvaluation::Curl,
        DifferentialOperator::RotatedGradient => DerivativeEvaluation::RotatedGradient,
        DifferentialOperator::SymmetricGradient => DerivativeEvaluation::SymmetricGradient,
        DifferentialOperator::TimeDerivative => DerivativeEvaluation::TimeDerivative,
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

fn trace_mapping(family: &SpaceFamily) -> TraceMapping {
    match family {
        SpaceFamily::HCurl => TraceMapping::Tangential,
        SpaceFamily::HDiv => TraceMapping::Normal,
        SpaceFamily::H1 | SpaceFamily::L2 | SpaceFamily::Dg => TraceMapping::Value,
    }
}

fn element_family(family: &SpaceFamily) -> ElementFamilyRequirement {
    match family {
        SpaceFamily::H1 => ElementFamilyRequirement::H1,
        SpaceFamily::L2 => ElementFamilyRequirement::L2,
        SpaceFamily::HCurl => ElementFamilyRequirement::Hcurl,
        SpaceFamily::HDiv => ElementFamilyRequirement::Hdiv,
        SpaceFamily::Dg => ElementFamilyRequirement::DiscontinuousGalerkin,
    }
}

fn sorted_domains<const N: usize>(domains: [DomainId; N]) -> Vec<DomainId> {
    let mut domains = domains.into_iter().collect::<Vec<_>>();
    domains.sort_unstable();
    domains.dedup();
    domains
}
