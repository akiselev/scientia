//! Typed variational-form artifacts projected from the canonical semantic arena.

use crate::id::{Digest, span_independent_digest};
use crate::scientific::{
    BinaryOp, BoundaryConditionKind, Continuity, FieldRole, SpaceFamily, SpaceSpec, UnaryOp,
    ValueShape,
};
use crate::semantic::{
    AxisContraction, DeclarationId, DifferentialOperator, DomainId, ExprId, Frame, RegionId,
    RegionKind, SemanticDeclarationKind, SemanticExpr, SemanticExprKind, SemanticMeasure,
    SemanticModel, SemanticModule, SemanticProvider, SemanticRegion, SemanticRole, SemanticShape,
    SemanticType, SymbolId, TraceSide, semantic_arena_digest,
};
use crate::source::SourceSpan;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

// "/5" (GX-A3) adds `providers`, so FC4 can see provider `differentiability` when deciding
// which `ModelDefinedProperty` inputs to inline for chain-rule tangents.
// "/6" (GX-facet) adds `FormCapture::definition`: a boundary condition whose authored value is a
// bare provider call (no wrapping symbol) synthesizes a compiler symbol and capture for it, so
// FC4 can bind it as an External input exactly like a cell-measure `ModelDefinedProperty`.
pub const VARIATIONAL_FORM_SCHEMA: &str = "scientia-variational-form/6";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormArity {
    pub test: u16,
    pub trial: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormArgumentRole {
    Test,
    Trial,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormArgument {
    pub symbol: SymbolId,
    pub role: FormArgumentRole,
    pub ty: SemanticType,
    pub domain: DomainId,
    pub space: SpaceSpec,
    pub source_span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormCaptureRole {
    PhysicalField(FieldRole),
    Parameter,
    Constant,
    Source,
    Property,
    ConstitutiveLaw,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormCapture {
    pub symbol: SymbolId,
    pub role: FormCaptureRole,
    pub ty: SemanticType,
    pub domain: Option<DomainId>,
    pub space: Option<SpaceSpec>,
    pub source_span: SourceSpan,
    /// GX-facet: the original defining expression, populated only when `symbol` is a
    /// compiler-synthesized id (see [`SymbolId::is_generated`]) that names no real
    /// `model.symbols` entry -- for example a boundary condition's bare provider-call value.
    /// `None` for every capture that names a real, declared symbol; those already resolve their
    /// definition through `model.declarations`.
    pub definition: Option<ExprId>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VariationalIntegral {
    pub measure: SemanticMeasure,
    pub side: FormSide,
    pub integrand: ExprId,
    pub source_span: SourceSpan,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormSide {
    Cell,
    Exterior,
    Interior,
    Interface,
    Point,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormTransformation {
    Authored,
    ResidualizeEquation {
        equation: DeclarationId,
    },
    MultiplyByTest {
        argument: SymbolId,
    },
    IntegrateByParts {
        source: ExprId,
        operator: DifferentialOperator,
    },
    SubstituteBoundaryCondition {
        declaration: DeclarationId,
    },
    EliminateEssentialBoundaryTerm {
        declaration: DeclarationId,
    },
    /// Batch P: the boundary term on `region` was substituted by the natural zero flux datum
    /// because no boundary condition targets the test field there; see
    /// [`BoundaryTermDisposition::NaturallyClosed`].
    SubstituteNaturalClosure {
        region: RegionId,
    },
}

/// Complex-valued forms never gain an implicit conjugation during FC2 derivation. Authored or
/// later compiler passes must carry each conjugation explicitly in the semantic expression arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormComplexConvention {
    ExplicitConjugationOnly,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormAssumption {
    ExteriorRegionsPartitionBoundary {
        domain: DomainId,
        regions: Vec<RegionId>,
        /// GX-A6 natural-boundary convention: `true` when the model declared no exterior
        /// region for `domain` and a whole-boundary region
        /// (`boundary("<Domain>.boundary")`, see `RegionId::generated_for`) was synthesized
        /// instead. A `BoundaryPartitionRequirement` is still produced for the synthesized
        /// region, so a downstream realization must still discharge it.
        implicit_natural_boundary: bool,
    },
    TestTraceVanishes {
        argument: SymbolId,
        region: RegionId,
        condition: DeclarationId,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryTermDisposition {
    /// Batch P (`sinbad/ARCHITECTURE.md` §4.4 `ImplicitNatural`): the region carries no
    /// boundary condition for the test field, so the flux datum is the natural zero and the
    /// boundary term is substituted by zero -- no integral is emitted. Keeping a state-computed
    /// facet integral here would cancel the integration by parts and enforce nothing; a case
    /// that wants a nonzero flux must bind a declared `neumann` value. `flux` is the signed
    /// strong normal-flux expression (the boundary datum's model-side counterpart), retained so
    /// a later `boundary_flux_strong` observable or SC-W2 port can name it.
    NaturallyClosed {
        flux: ExprId,
    },
    Substituted {
        declaration: DeclarationId,
        integral_index: usize,
    },
    EliminatedByEssentialCondition {
        declaration: DeclarationId,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryTermReceipt {
    pub region: RegionId,
    pub source: ExprId,
    pub integrand: ExprId,
    pub disposition: BoundaryTermDisposition,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormReceipt {
    pub source_declaration: DeclarationId,
    /// Physical field whose space supplied the generated test argument. Authored forms have no
    /// unique source field and therefore record `None`.
    pub test_space_source: Option<SymbolId>,
    pub source_span: SourceSpan,
    pub complex_convention: FormComplexConvention,
    pub transformations: Vec<FormTransformation>,
    pub assumptions: Vec<FormAssumption>,
    pub boundary_terms: Vec<BoundaryTermReceipt>,
}

/// A typed form retains canonical semantic expressions and resolved identities. Display names are
/// intentionally absent from arguments, captures, measures, and integrands.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VariationalForm {
    pub schema: String,
    pub model: String,
    pub name: String,
    pub source_semantic_digest: Digest,
    pub artifact_digest: Digest,
    pub declaration: DeclarationId,
    pub arity: FormArity,
    pub arguments: Vec<FormArgument>,
    pub captures: Vec<FormCapture>,
    pub expressions: Vec<SemanticExpr>,
    pub integrals: Vec<VariationalIntegral>,
    /// The source model's declared providers (GX-A3): carried through so FC4's tensor factoring
    /// can see a `ModelDefinedProperty` input's provider `differentiability`/`locality` without
    /// widening `factor_operator`'s own signature.
    pub providers: Vec<SemanticProvider>,
    pub receipt: FormReceipt,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum FormCompileError {
    #[error("semantic module has no model named `{0}`")]
    MissingModel(String),
    #[error("model `{model}` has no form named `{form}`")]
    MissingForm { model: String, form: String },
    #[error("model `{model}` declares form `{form}` more than once")]
    DuplicateForm { model: String, form: String },
    #[error("form `{form}` contains no integrals")]
    EmptyForm { form: String },
    #[error("form `{form}` references symbol {symbol} with invalid role {role:?}")]
    InvalidReferenceRole {
        form: String,
        symbol: SymbolId,
        role: SemanticRole,
    },
    #[error("form argument {symbol} has no resolved domain")]
    ArgumentWithoutDomain { symbol: SymbolId },
    #[error("form argument {symbol} has no function-space declaration")]
    ArgumentWithoutSpace { symbol: SymbolId },
    #[error("semantic expression id {0} is outside the canonical arena")]
    InvalidExpression(ExprId),
    #[error("semantic symbol id {0} is outside the canonical arena")]
    InvalidSymbol(SymbolId),
    #[error("model `{model}` has no equation named `{equation}`")]
    MissingEquation { model: String, equation: String },
    #[error("model `{model}` declares equation `{equation}` more than once")]
    DuplicateEquation { model: String, equation: String },
    #[error("equation `{equation}` has no integration domain")]
    EquationWithoutDomain { equation: String },
    #[error("equation `{equation}` has no unambiguous physical field for its test space")]
    AmbiguousTestField { equation: String },
    #[error("field {symbol} cannot supply a test space for equation `{equation}`")]
    InvalidTestField { equation: String, symbol: SymbolId },
    #[error("{code}: equation `{equation}` requires unsupported formulation capability: {detail}")]
    UnsupportedStrongForm {
        code: &'static str,
        equation: String,
        detail: String,
    },
    #[error("{code}: form `{form}` has invalid side semantics: {detail}")]
    InvalidSideSemantics {
        code: &'static str,
        form: String,
        detail: String,
    },
    #[error(
        "FORM_AMBIGUOUS_NEUMANN_FLUX: equation `{equation}` has multiple integrated flux terms for field {field} on region {region}"
    )]
    AmbiguousNeumannFlux {
        equation: String,
        field: SymbolId,
        region: RegionId,
    },
    #[error("FORM_INVALID_DERIVED_DIFFERENTIAL: cannot apply {operator:?} to shape {shape:?}")]
    InvalidDerivedDifferential {
        operator: DifferentialOperator,
        shape: SemanticShape,
    },
}

/// Compile one authored form exclusively from typed semantic declarations and identities.
pub fn compile_variational_form(
    module: &SemanticModule,
    model_name: &str,
    form_name: &str,
) -> Result<VariationalForm, FormCompileError> {
    let model = module
        .models
        .iter()
        .find(|model| model.name == model_name)
        .ok_or_else(|| FormCompileError::MissingModel(model_name.to_owned()))?;
    let mut matches = model.declarations.iter().filter(|declaration| {
        declaration.name == form_name
            && matches!(declaration.kind, SemanticDeclarationKind::Form { .. })
    });
    let declaration = matches
        .next()
        .ok_or_else(|| FormCompileError::MissingForm {
            model: model.name.clone(),
            form: form_name.to_owned(),
        })?;
    if matches.next().is_some() {
        return Err(FormCompileError::DuplicateForm {
            model: model.name.clone(),
            form: form_name.to_owned(),
        });
    }
    let SemanticDeclarationKind::Form { integrals } = &declaration.kind else {
        unreachable!("form declaration was selected above")
    };
    if integrals.is_empty() {
        return Err(FormCompileError::EmptyForm {
            form: form_name.to_owned(),
        });
    }

    let mut referenced = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for integral in integrals {
        collect_symbols(model, integral.integrand, &mut visited, &mut referenced)?;
    }

    let mut arguments = vec![];
    let mut captures = vec![];
    for symbol_id in referenced {
        let symbol = model
            .symbols
            .get(symbol_id.index())
            .ok_or(FormCompileError::InvalidSymbol(symbol_id))?;
        match &symbol.ty.role {
            SemanticRole::PhysicalField(FieldRole::Test | FieldRole::Trial) => {
                let role = match &symbol.ty.role {
                    SemanticRole::PhysicalField(FieldRole::Test) => FormArgumentRole::Test,
                    SemanticRole::PhysicalField(FieldRole::Trial) => FormArgumentRole::Trial,
                    _ => unreachable!(),
                };
                arguments.push(FormArgument {
                    symbol: symbol.id,
                    role,
                    ty: symbol.ty.clone(),
                    domain: symbol
                        .domain
                        .ok_or(FormCompileError::ArgumentWithoutDomain { symbol: symbol.id })?,
                    space: symbol
                        .space
                        .clone()
                        .ok_or(FormCompileError::ArgumentWithoutSpace { symbol: symbol.id })?,
                    source_span: symbol.span,
                });
            }
            role => {
                let role =
                    capture_role(role).ok_or_else(|| FormCompileError::InvalidReferenceRole {
                        form: form_name.to_owned(),
                        symbol: symbol.id,
                        role: role.clone(),
                    })?;
                captures.push(FormCapture {
                    symbol: symbol.id,
                    role,
                    ty: symbol.ty.clone(),
                    domain: symbol.domain,
                    space: symbol.space.clone(),
                    source_span: symbol.span,
                    definition: None,
                });
            }
        }
    }

    let source_semantic_digest = Digest {
        algorithm: "blake3".into(),
        hex: semantic_arena_digest(module),
    };
    let receipt = FormReceipt {
        source_declaration: declaration.id,
        test_space_source: None,
        source_span: declaration.span,
        complex_convention: FormComplexConvention::ExplicitConjugationOnly,
        transformations: vec![FormTransformation::Authored],
        assumptions: vec![],
        boundary_terms: vec![],
    };
    let integrals = integrals
        .iter()
        .map(|integral| VariationalIntegral {
            measure: integral.measure.clone(),
            side: side_for_measure(&integral.measure),
            integrand: integral.integrand,
            source_span: integral.span,
        })
        .collect::<Vec<_>>();
    validate_form_sides(form_name, &model.expressions, &integrals)?;
    let arity = FormArity {
        test: arguments
            .iter()
            .filter(|argument| argument.role == FormArgumentRole::Test)
            .count() as u16,
        trial: arguments
            .iter()
            .filter(|argument| argument.role == FormArgumentRole::Trial)
            .count() as u16,
    };
    let artifact_digest = span_independent_digest(&FormDigestPayload {
        schema: VARIATIONAL_FORM_SCHEMA,
        model: &model.name,
        name: form_name,
        source_semantic_digest: &source_semantic_digest,
        declaration: declaration.id,
        arity,
        arguments: &arguments,
        captures: &captures,
        expressions: &model.expressions,
        integrals: &integrals,
        providers: &model.providers,
        receipt: &receipt,
    });
    Ok(VariationalForm {
        schema: VARIATIONAL_FORM_SCHEMA.into(),
        model: model.name.clone(),
        name: form_name.to_owned(),
        source_semantic_digest,
        artifact_digest,
        declaration: declaration.id,
        arity,
        arguments,
        captures,
        expressions: model.expressions.to_vec(),
        integrals,
        providers: model.providers.clone(),
        receipt,
    })
}

/// Derive a residual form from one strong equation. The test space is selected only when the
/// equation result shape and domain identify one physical unknown/state field unambiguously.
pub fn derive_variational_form(
    module: &SemanticModule,
    model_name: &str,
    equation_name: &str,
) -> Result<VariationalForm, FormCompileError> {
    let model = module
        .models
        .iter()
        .find(|model| model.name == model_name)
        .ok_or_else(|| FormCompileError::MissingModel(model_name.to_owned()))?;
    let equation = select_equation(model, equation_name)?;
    let SemanticDeclarationKind::Equation { lhs, .. } = equation.kind else {
        unreachable!("equation declaration was selected above")
    };
    let domain = equation
        .domain
        .ok_or_else(|| FormCompileError::EquationWithoutDomain {
            equation: equation_name.to_owned(),
        })?;
    let direct = transitive_equation_fields(model, equation)?;
    let result_shape = &model
        .expressions
        .get(lhs.index())
        .ok_or(FormCompileError::InvalidExpression(lhs))?
        .ty
        .shape;
    let mut candidates = model
        .symbols
        .iter()
        .filter(|symbol| {
            symbol.domain == Some(domain)
                && symbol.space.is_some()
                && matches!(
                    symbol.ty.role,
                    SemanticRole::PhysicalField(FieldRole::Unknown | FieldRole::State)
                )
        })
        .filter(|symbol| {
            matches!(result_shape, SemanticShape::Deferred) || symbol.ty.shape == *result_shape
        })
        .collect::<Vec<_>>();
    if candidates.len() != 1 {
        let mut referenced = candidates
            .iter()
            .copied()
            .filter(|symbol| direct.contains(&symbol.id))
            .collect::<Vec<_>>();
        if referenced.len() == 1 {
            candidates = std::mem::take(&mut referenced);
        } else if !referenced.is_empty() {
            let mut ranked = referenced
                .into_iter()
                .map(|symbol| Ok((equation_field_score(model, equation, symbol.id)?, symbol)))
                .collect::<Result<Vec<_>, FormCompileError>>()?;
            ranked.sort_by_key(|(score, symbol)| (*score, symbol.id));
            if let Some((best_score, best)) = ranked.pop()
                && ranked
                    .last()
                    .is_none_or(|(next_score, _)| *next_score != best_score)
            {
                candidates = vec![best];
            }
        }
    }
    let test_field = match candidates.as_slice() {
        [field] => field.id,
        _ => {
            return Err(FormCompileError::AmbiguousTestField {
                equation: equation_name.to_owned(),
            });
        }
    };
    derive_variational_form_for(module, model_name, equation_name, test_field)
}

/// Derive a residual form with an explicit physical field supplying the test space.
/// SC-W1 (`sinbad/ARCHITECTURE.md` §6): the output kernel of `output NAME: Kind on D = G;` is
/// the point function `G` at a quadrature point together with its tangent with respect to the
/// model's fields. It is obtained through the ordinary FC2--FC5 chain by deriving the synthetic
/// scalar functional `∫_D G · w` against a generated `L2(order=0)` test argument `w`: the
/// test-dual primal QFunction output is exactly `G` (the test basis is never in the point
/// function), and the JVP is `dG/dx · δx`. Nothing about `G` is rewritten; provider calls are
/// lifted into captures exactly as in a residual.
pub fn derive_output_form(
    module: &SemanticModule,
    model_name: &str,
    output_name: &str,
) -> Result<VariationalForm, FormCompileError> {
    let model = module
        .models
        .iter()
        .find(|model| model.name == model_name)
        .ok_or_else(|| FormCompileError::MissingModel(model_name.to_owned()))?;
    let output = model
        .declarations
        .iter()
        .find(|declaration| {
            declaration.name == output_name
                && matches!(declaration.kind, SemanticDeclarationKind::Output { .. })
        })
        .ok_or_else(|| FormCompileError::MissingForm {
            model: model_name.to_owned(),
            form: format!("output {output_name}"),
        })?;
    let SemanticDeclarationKind::Output { value, domain } = output.kind else {
        unreachable!("output declaration was selected above")
    };
    let mut arena = FormArena::new(model.expressions.to_vec());
    let argument_symbol = SymbolId::generated_for(output.id);
    let mut argument_type = arena.expressions[value.index()].ty.clone();
    argument_type.role = SemanticRole::PhysicalField(FieldRole::Test);
    if matches!(argument_type.shape, SemanticShape::Deferred) {
        argument_type.shape = SemanticShape::Numeric(ValueShape::Scalar);
    }
    let argument_expr = arena.push(
        SemanticExprKind::Symbol {
            symbol: argument_symbol,
        },
        argument_type.clone(),
        output.span,
    );
    let mut lifted_captures: Vec<FormCapture> = vec![];
    let term = lift_provider_calls(&mut arena, value, &mut lifted_captures)?;
    arena.refine_deferred_shape(term, argument_type.shape.clone());
    let cell = arena.pair(term, argument_expr, output.span)?;
    let integrals = vec![VariationalIntegral {
        measure: SemanticMeasure::Cell { domain },
        side: FormSide::Cell,
        integrand: cell,
        source_span: output.span,
    }];

    let mut referenced = BTreeSet::new();
    let mut visited = BTreeSet::new();
    collect_symbol_ids(&arena.expressions, cell, &mut visited, &mut referenced)?;
    let mut fields = BTreeSet::new();
    collect_transitive_fields(
        model,
        value,
        &mut BTreeSet::new(),
        &mut BTreeSet::new(),
        &mut fields,
    )?;
    referenced.extend(fields);
    let mut captures = vec![];
    for symbol_id in referenced {
        if symbol_id == argument_symbol || symbol_id.is_generated() {
            continue;
        }
        let symbol = model
            .symbols
            .get(symbol_id.index())
            .ok_or(FormCompileError::InvalidSymbol(symbol_id))?;
        let role = capture_role(&symbol.ty.role).ok_or_else(|| {
            FormCompileError::InvalidReferenceRole {
                form: output_name.to_owned(),
                symbol: symbol.id,
                role: symbol.ty.role.clone(),
            }
        })?;
        captures.push(FormCapture {
            symbol: symbol.id,
            role,
            ty: symbol.ty.clone(),
            domain: symbol.domain,
            space: symbol.space.clone(),
            source_span: symbol.span,
            definition: None,
        });
    }
    captures.extend(lifted_captures);

    let source_semantic_digest = Digest {
        algorithm: "blake3".into(),
        hex: semantic_arena_digest(module),
    };
    let arguments = vec![FormArgument {
        symbol: argument_symbol,
        role: FormArgumentRole::Test,
        ty: argument_type,
        domain,
        space: SpaceSpec {
            family: SpaceFamily::L2,
            order: 0,
            continuity: Continuity::Discontinuous,
        },
        source_span: output.span,
    }];
    let receipt = FormReceipt {
        source_declaration: output.id,
        test_space_source: None,
        source_span: output.span,
        complex_convention: FormComplexConvention::ExplicitConjugationOnly,
        transformations: vec![FormTransformation::MultiplyByTest {
            argument: argument_symbol,
        }],
        assumptions: vec![],
        boundary_terms: vec![],
    };
    let name = format!("{output_name}::output");
    let arity = FormArity { test: 1, trial: 0 };
    validate_form_sides(&name, &arena.expressions, &integrals)?;
    let artifact_digest = span_independent_digest(&FormDigestPayload {
        schema: VARIATIONAL_FORM_SCHEMA,
        model: &model.name,
        name: &name,
        source_semantic_digest: &source_semantic_digest,
        declaration: output.id,
        arity,
        arguments: &arguments,
        captures: &captures,
        expressions: &arena.expressions,
        integrals: &integrals,
        providers: &model.providers,
        receipt: &receipt,
    });
    Ok(VariationalForm {
        schema: VARIATIONAL_FORM_SCHEMA.into(),
        model: model.name.clone(),
        name,
        source_semantic_digest,
        artifact_digest,
        declaration: output.id,
        arity,
        arguments,
        captures,
        expressions: arena.expressions,
        integrals,
        providers: model.providers.clone(),
        receipt,
    })
}

pub fn derive_variational_form_for(
    module: &SemanticModule,
    model_name: &str,
    equation_name: &str,
    test_field: SymbolId,
) -> Result<VariationalForm, FormCompileError> {
    let model = module
        .models
        .iter()
        .find(|model| model.name == model_name)
        .ok_or_else(|| FormCompileError::MissingModel(model_name.to_owned()))?;
    let equation = select_equation(model, equation_name)?;
    let domain = equation
        .domain
        .ok_or_else(|| FormCompileError::EquationWithoutDomain {
            equation: equation_name.to_owned(),
        })?;
    let field = model
        .symbols
        .get(test_field.index())
        .filter(|field| {
            field.id == test_field
                && field.domain == Some(domain)
                && field.space.is_some()
                && matches!(
                    field.ty.role,
                    SemanticRole::PhysicalField(FieldRole::Unknown | FieldRole::State)
                )
        })
        .ok_or_else(|| FormCompileError::InvalidTestField {
            equation: equation_name.to_owned(),
            symbol: test_field,
        })?;
    let SemanticDeclarationKind::Equation { lhs, rhs } = equation.kind else {
        unreachable!("equation declaration was selected above")
    };

    let mut arena = FormArena::new(model.expressions.to_vec());
    // Generated per declaration (not `model.symbols.len()`) so that distinct equations of the
    // same model never synthesize the same test-argument `SymbolId`; see
    // `SymbolId::generated_for` and `SymbolId::is_generated`.
    let argument_symbol = SymbolId::generated_for(equation.id);
    let mut argument_type = field.ty.clone();
    argument_type.role = SemanticRole::PhysicalField(FieldRole::Test);
    let argument_expr = arena.push(
        SemanticExprKind::Symbol {
            symbol: argument_symbol,
        },
        argument_type.clone(),
        equation.span,
    );
    let mut terms = vec![];
    flatten_residual_terms(&arena.expressions, lhs, 1, &mut terms)?;
    flatten_residual_terms(&arena.expressions, rhs, -1, &mut terms)?;

    let mut integrals = vec![];
    let mut transformations = vec![
        FormTransformation::ResidualizeEquation {
            equation: equation.id,
        },
        FormTransformation::MultiplyByTest {
            argument: argument_symbol,
        },
    ];
    let mut assumptions = vec![];
    let mut boundary_terms = vec![];
    let mut substituted_neumann_fluxes = BTreeSet::new();
    let mut boundary_captures: Vec<FormCapture> = vec![];
    // Batch P: every provider call inside a residual term becomes a compiler-synthesized
    // property capture (see `lift_provider_calls`), so `rho * convect(u, u)` or a bare
    // `reaction(c, phi)` source binds through the same `ModelDefinedProperty` path as a named
    // `property`, instead of reaching FC4 as an un-bindable raw call.
    for (term, _) in &mut terms {
        *term = lift_provider_calls(&mut arena, *term, &mut boundary_captures)?;
    }
    let spatial_dimension = model.domains[domain.index()].spatial_dimension;
    for (term, sign) in terms {
        match arena.expressions[term.index()].kind.clone() {
            SemanticExprKind::Differential {
                operator: DifferentialOperator::Divergence,
                arg: flux,
            } if field
                .space
                .as_ref()
                .is_some_and(test_space_supports_gradient) =>
            {
                arena.refine_deferred_shape(
                    flux,
                    divergence_operand_shape(&field.ty.shape, spatial_dimension),
                );
                transformations.push(FormTransformation::IntegrateByParts {
                    source: term,
                    operator: DifferentialOperator::Divergence,
                });
                let gradient = arena.differential(
                    DifferentialOperator::Gradient,
                    argument_expr,
                    domain,
                    spatial_dimension,
                    equation.span,
                )?;
                let cell = arena.contract(flux, gradient, true, equation.span)?;
                let cell = arena.with_sign(cell, -sign, equation.span);
                integrals.push(VariationalIntegral {
                    measure: SemanticMeasure::Cell { domain },
                    side: FormSide::Cell,
                    integrand: cell,
                    source_span: equation.span,
                });
                derive_boundary_terms(
                    model,
                    equation_name,
                    equation.id,
                    test_field,
                    term,
                    flux,
                    argument_expr,
                    sign,
                    &mut arena,
                    &mut integrals,
                    &mut transformations,
                    &mut assumptions,
                    &mut boundary_terms,
                    &mut substituted_neumann_fluxes,
                    &mut boundary_captures,
                )?;
            }
            SemanticExprKind::Differential {
                operator: DifferentialOperator::Gradient,
                arg: scalar,
            } if field
                .space
                .as_ref()
                .is_some_and(test_space_supports_divergence) =>
            {
                arena.refine_deferred_shape(scalar, SemanticShape::Numeric(ValueShape::Scalar));
                transformations.push(FormTransformation::IntegrateByParts {
                    source: term,
                    operator: DifferentialOperator::Gradient,
                });
                let divergence = arena.differential(
                    DifferentialOperator::Divergence,
                    argument_expr,
                    domain,
                    spatial_dimension,
                    equation.span,
                )?;
                let cell = arena.multiply(scalar, divergence, equation.span)?;
                let cell = arena.with_sign(cell, -sign, equation.span);
                integrals.push(VariationalIntegral {
                    measure: SemanticMeasure::Cell { domain },
                    side: FormSide::Cell,
                    integrand: cell,
                    source_span: equation.span,
                });
                derive_boundary_terms(
                    model,
                    equation_name,
                    equation.id,
                    test_field,
                    term,
                    scalar,
                    argument_expr,
                    sign,
                    &mut arena,
                    &mut integrals,
                    &mut transformations,
                    &mut assumptions,
                    &mut boundary_terms,
                    &mut substituted_neumann_fluxes,
                    &mut boundary_captures,
                )?;
            }
            SemanticExprKind::Differential {
                operator: DifferentialOperator::Curl,
                arg: flux,
            } if field.space.as_ref().is_some_and(test_space_supports_curl) => {
                arena.refine_deferred_shape(flux, field.ty.shape.clone());
                transformations.push(FormTransformation::IntegrateByParts {
                    source: term,
                    operator: DifferentialOperator::Curl,
                });
                let curl = arena.differential(
                    DifferentialOperator::Curl,
                    argument_expr,
                    domain,
                    spatial_dimension,
                    equation.span,
                )?;
                let cell = arena.pair(flux, curl, equation.span)?;
                let cell = arena.with_sign(cell, sign, equation.span);
                integrals.push(VariationalIntegral {
                    measure: SemanticMeasure::Cell { domain },
                    side: FormSide::Cell,
                    integrand: cell,
                    source_span: equation.span,
                });
                derive_boundary_terms(
                    model,
                    equation_name,
                    equation.id,
                    test_field,
                    term,
                    flux,
                    argument_expr,
                    sign,
                    &mut arena,
                    &mut integrals,
                    &mut transformations,
                    &mut assumptions,
                    &mut boundary_terms,
                    &mut substituted_neumann_fluxes,
                    &mut boundary_captures,
                )?;
            }
            _ => {
                arena.refine_deferred_shape(term, field.ty.shape.clone());
                let cell = arena.pair(term, argument_expr, equation.span)?;
                let cell = arena.with_sign(cell, sign, equation.span);
                integrals.push(VariationalIntegral {
                    measure: SemanticMeasure::Cell { domain },
                    side: FormSide::Cell,
                    integrand: cell,
                    source_span: equation.span,
                });
            }
        }
    }
    if integrals.is_empty() {
        return Err(FormCompileError::EmptyForm {
            form: equation_name.to_owned(),
        });
    }

    let mut referenced = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for integral in &integrals {
        collect_symbol_ids(
            &arena.expressions,
            integral.integrand,
            &mut visited,
            &mut referenced,
        )?;
    }
    referenced.extend(transitive_equation_fields(model, equation)?);
    let mut captures = vec![];
    for symbol_id in referenced {
        if symbol_id == argument_symbol {
            continue;
        }
        // GX-facet: a compiler-synthesized symbol (see `SymbolId::is_generated`) names no
        // `model.symbols` entry to look up here; `derive_boundary_terms` already recorded its
        // capture directly in `boundary_captures`, merged in below.
        if symbol_id.is_generated() {
            continue;
        }
        let symbol = model
            .symbols
            .get(symbol_id.index())
            .ok_or(FormCompileError::InvalidSymbol(symbol_id))?;
        let role = capture_role(&symbol.ty.role).ok_or_else(|| {
            FormCompileError::InvalidReferenceRole {
                form: equation_name.to_owned(),
                symbol: symbol.id,
                role: symbol.ty.role.clone(),
            }
        })?;
        captures.push(FormCapture {
            symbol: symbol.id,
            role,
            ty: symbol.ty.clone(),
            domain: symbol.domain,
            space: symbol.space.clone(),
            source_span: symbol.span,
            definition: None,
        });
    }
    captures.extend(boundary_captures);

    let source_semantic_digest = Digest {
        algorithm: "blake3".into(),
        hex: semantic_arena_digest(module),
    };
    let arguments = vec![FormArgument {
        symbol: argument_symbol,
        role: FormArgumentRole::Test,
        ty: argument_type,
        domain,
        space: field.space.clone().expect("validated above"),
        source_span: equation.span,
    }];
    let receipt = FormReceipt {
        source_declaration: equation.id,
        test_space_source: Some(test_field),
        source_span: equation.span,
        complex_convention: FormComplexConvention::ExplicitConjugationOnly,
        transformations,
        assumptions,
        boundary_terms,
    };
    let name = format!("{equation_name}::weak");
    let arity = FormArity { test: 1, trial: 0 };
    validate_form_sides(&name, &arena.expressions, &integrals)?;
    let artifact_digest = span_independent_digest(&FormDigestPayload {
        schema: VARIATIONAL_FORM_SCHEMA,
        model: &model.name,
        name: &name,
        source_semantic_digest: &source_semantic_digest,
        declaration: equation.id,
        arity,
        arguments: &arguments,
        captures: &captures,
        expressions: &arena.expressions,
        integrals: &integrals,
        providers: &model.providers,
        receipt: &receipt,
    });
    Ok(VariationalForm {
        schema: VARIATIONAL_FORM_SCHEMA.into(),
        model: model.name.clone(),
        name,
        source_semantic_digest,
        artifact_digest,
        declaration: equation.id,
        arity,
        arguments,
        captures,
        expressions: arena.expressions,
        integrals,
        providers: model.providers.clone(),
        receipt,
    })
}

fn select_equation<'a>(
    model: &'a SemanticModel,
    equation_name: &str,
) -> Result<&'a crate::semantic::SemanticDeclaration, FormCompileError> {
    let mut matches = model.declarations.iter().filter(|declaration| {
        declaration.name == equation_name
            && matches!(declaration.kind, SemanticDeclarationKind::Equation { .. })
    });
    let equation = matches
        .next()
        .ok_or_else(|| FormCompileError::MissingEquation {
            model: model.name.clone(),
            equation: equation_name.to_owned(),
        })?;
    if matches.next().is_some() {
        return Err(FormCompileError::DuplicateEquation {
            model: model.name.clone(),
            equation: equation_name.to_owned(),
        });
    }
    Ok(equation)
}

fn test_space_supports_gradient(space: &SpaceSpec) -> bool {
    matches!(space.family, SpaceFamily::H1 | SpaceFamily::Dg)
}

fn test_space_supports_divergence(space: &SpaceSpec) -> bool {
    matches!(
        space.family,
        SpaceFamily::H1 | SpaceFamily::HDiv | SpaceFamily::Dg
    )
}

fn test_space_supports_curl(space: &SpaceSpec) -> bool {
    matches!(
        space.family,
        SpaceFamily::H1 | SpaceFamily::HCurl | SpaceFamily::Dg
    )
}

fn divergence_operand_shape(result: &SemanticShape, spatial_dimension: u8) -> SemanticShape {
    match result {
        SemanticShape::Numeric(ValueShape::Scalar) => {
            SemanticShape::Numeric(ValueShape::Vector(spatial_dimension))
        }
        SemanticShape::Numeric(ValueShape::Vector(rows)) => {
            SemanticShape::Numeric(ValueShape::Tensor {
                rows: *rows,
                cols: spatial_dimension,
            })
        }
        _ => SemanticShape::Deferred,
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct EquationFieldScore {
    time_derivative_depth: usize,
    spatial_derivative_depth: usize,
    occurrences: usize,
}

fn equation_field_score(
    model: &SemanticModel,
    equation: &crate::semantic::SemanticDeclaration,
    target: SymbolId,
) -> Result<EquationFieldScore, FormCompileError> {
    let SemanticDeclarationKind::Equation { lhs, rhs } = equation.kind else {
        unreachable!("equation declaration was selected above")
    };
    let mut active_definitions = BTreeSet::new();
    let left = expression_field_score(model, lhs, target, 0, 0, &mut active_definitions)?;
    let right = expression_field_score(model, rhs, target, 0, 0, &mut active_definitions)?;
    Ok(EquationFieldScore {
        time_derivative_depth: left.time_derivative_depth.max(right.time_derivative_depth),
        spatial_derivative_depth: left
            .spatial_derivative_depth
            .max(right.spatial_derivative_depth),
        occurrences: left.occurrences + right.occurrences,
    })
}

fn expression_field_score(
    model: &SemanticModel,
    id: ExprId,
    target: SymbolId,
    time_depth: usize,
    spatial_depth: usize,
    active_definitions: &mut BTreeSet<SymbolId>,
) -> Result<EquationFieldScore, FormCompileError> {
    let expression = model
        .expressions
        .get(id.index())
        .ok_or(FormCompileError::InvalidExpression(id))?;
    if let SemanticExprKind::Symbol { symbol } = expression.kind {
        if symbol == target {
            return Ok(EquationFieldScore {
                time_derivative_depth: time_depth,
                spatial_derivative_depth: spatial_depth,
                occurrences: 1,
            });
        }
        if !active_definitions.insert(symbol) {
            return Ok(EquationFieldScore::default());
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
        let score = if let Some(definition) = definition {
            expression_field_score(
                model,
                definition,
                target,
                time_depth,
                spatial_depth,
                active_definitions,
            )?
        } else {
            EquationFieldScore::default()
        };
        active_definitions.remove(&symbol);
        return Ok(score);
    }
    let (time_depth, spatial_depth) = match expression.kind {
        SemanticExprKind::Differential {
            operator: DifferentialOperator::TimeDerivative,
            ..
        } => (time_depth + 1, spatial_depth),
        SemanticExprKind::Differential { .. } => (time_depth, spatial_depth + 1),
        _ => (time_depth, spatial_depth),
    };
    let mut score = EquationFieldScore::default();
    for child in expression_children(&expression.kind) {
        let child = expression_field_score(
            model,
            child,
            target,
            time_depth,
            spatial_depth,
            active_definitions,
        )?;
        score.time_derivative_depth = score.time_derivative_depth.max(child.time_derivative_depth);
        score.spatial_derivative_depth = score
            .spatial_derivative_depth
            .max(child.spatial_derivative_depth);
        score.occurrences += child.occurrences;
    }
    Ok(score)
}

fn transitive_equation_fields(
    model: &SemanticModel,
    equation: &crate::semantic::SemanticDeclaration,
) -> Result<BTreeSet<SymbolId>, FormCompileError> {
    let SemanticDeclarationKind::Equation { lhs, rhs } = &equation.kind else {
        unreachable!("equation declaration was selected above")
    };
    let mut fields = BTreeSet::new();
    let mut expressions = BTreeSet::new();
    let mut symbols = BTreeSet::new();
    collect_transitive_fields(model, *lhs, &mut expressions, &mut symbols, &mut fields)?;
    collect_transitive_fields(model, *rhs, &mut expressions, &mut symbols, &mut fields)?;
    Ok(fields)
}

fn collect_transitive_fields(
    model: &SemanticModel,
    id: ExprId,
    expressions: &mut BTreeSet<ExprId>,
    symbols: &mut BTreeSet<SymbolId>,
    fields: &mut BTreeSet<SymbolId>,
) -> Result<(), FormCompileError> {
    if !expressions.insert(id) {
        return Ok(());
    }
    let expression = model
        .expressions
        .get(id.index())
        .ok_or(FormCompileError::InvalidExpression(id))?;
    if let SemanticExprKind::Symbol { symbol } = expression.kind {
        let semantic_symbol = model
            .symbols
            .get(symbol.index())
            .ok_or(FormCompileError::InvalidSymbol(symbol))?;
        if matches!(semantic_symbol.ty.role, SemanticRole::PhysicalField(_)) {
            fields.insert(symbol);
        }
        if symbols.insert(symbol)
            && let Some(declaration) = model
                .declarations
                .iter()
                .find(|declaration| declaration.symbol == Some(symbol))
        {
            let value = match declaration.kind {
                SemanticDeclarationKind::Value { value } => value,
                SemanticDeclarationKind::Property { value }
                | SemanticDeclarationKind::ConstitutiveLaw { value } => Some(value),
                _ => None,
            };
            if let Some(value) = value {
                collect_transitive_fields(model, value, expressions, symbols, fields)?;
            }
        }
    }
    for child in expression_children(&expression.kind) {
        collect_transitive_fields(model, child, expressions, symbols, fields)?;
    }
    Ok(())
}

fn flatten_residual_terms(
    expressions: &[SemanticExpr],
    id: ExprId,
    sign: i8,
    terms: &mut Vec<(ExprId, i8)>,
) -> Result<(), FormCompileError> {
    let expression = expressions
        .get(id.index())
        .ok_or(FormCompileError::InvalidExpression(id))?;
    if matches!(
        expression.kind,
        SemanticExprKind::Number {
            value: 0.0,
            unit: None,
            ..
        }
    ) {
        return Ok(());
    }
    match expression.kind {
        SemanticExprKind::Binary {
            op: BinaryOp::Add,
            lhs,
            rhs,
        } => {
            flatten_residual_terms(expressions, lhs, sign, terms)?;
            flatten_residual_terms(expressions, rhs, sign, terms)?;
        }
        SemanticExprKind::Binary {
            op: BinaryOp::Sub,
            lhs,
            rhs,
        } => {
            flatten_residual_terms(expressions, lhs, sign, terms)?;
            flatten_residual_terms(expressions, rhs, -sign, terms)?;
        }
        SemanticExprKind::Unary {
            op: UnaryOp::Neg,
            arg,
        } => flatten_residual_terms(expressions, arg, -sign, terms)?,
        _ => terms.push((id, sign)),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn derive_boundary_terms(
    model: &SemanticModel,
    equation_name: &str,
    _equation: DeclarationId,
    test_field: SymbolId,
    source: ExprId,
    operand: ExprId,
    test: ExprId,
    sign: i8,
    arena: &mut FormArena,
    integrals: &mut Vec<VariationalIntegral>,
    transformations: &mut Vec<FormTransformation>,
    assumptions: &mut Vec<FormAssumption>,
    boundary_terms: &mut Vec<BoundaryTermReceipt>,
    substituted_neumann_fluxes: &mut BTreeSet<(RegionId, SymbolId)>,
    boundary_captures: &mut Vec<FormCapture>,
) -> Result<(), FormCompileError> {
    let argument = match arena.expressions[test.index()].kind {
        SemanticExprKind::Symbol { symbol } => symbol,
        _ => unreachable!("derived test argument is a symbol expression"),
    };
    let domain = model.symbols[test_field.index()]
        .domain
        .expect("test field domain validated before derivation");
    let mut regions: Vec<SemanticRegion> = model
        .regions
        .iter()
        .filter(|region| region.kind == RegionKind::ExteriorFacet && region.domain == Some(domain))
        .cloned()
        .collect();
    // GX-A6 natural-boundary convention: when the model declares no exterior region for this
    // domain, synthesize the implicit whole-boundary region `boundary("<Domain>.boundary")`
    // rather than refusing. Its boundary term is naturally closed (zero flux datum), exactly
    // like any other region with no matching boundary condition below;
    // `implicit_natural_boundary` still records that a case must discharge the partition.
    let implicit_natural_boundary = regions.is_empty();
    if implicit_natural_boundary {
        regions.push(SemanticRegion {
            id: RegionId::generated_for(domain),
            name: format!("{}.boundary", model.domains[domain.index()].name),
            kind: RegionKind::ExteriorFacet,
            domain: Some(domain),
            span: SourceSpan::default(),
        });
    }
    let partition_assumption = FormAssumption::ExteriorRegionsPartitionBoundary {
        domain,
        regions: regions.iter().map(|region| region.id).collect(),
        implicit_natural_boundary,
    };
    if !assumptions.contains(&partition_assumption) {
        assumptions.push(partition_assumption);
    }
    let operator = match arena.expressions[source.index()].kind {
        SemanticExprKind::Differential { operator, .. } => operator,
        _ => unreachable!("boundary derivation follows a differential operation"),
    };
    for region in regions {
        let test_trace = arena.trace(test, TraceSide::Exterior, region.span);
        // `flux` is the bare (test-free) boundary datum the integration by parts produced:
        // `q·n` for a divergence, the operand's trace for a gradient (the normal lands on the
        // vector test function) or a curl (the tangential trace pairs with the test).
        let (flux, formal) = match operator {
            DifferentialOperator::Divergence => {
                let normal = arena.normal_component(operand, TraceSide::Exterior, region.span)?;
                (normal, arena.pair(normal, test_trace, region.span)?)
            }
            DifferentialOperator::Gradient => {
                let value = arena.trace(operand, TraceSide::Exterior, region.span);
                let normal_test = arena.normal_component(test, TraceSide::Exterior, region.span)?;
                (value, arena.multiply(value, normal_test, region.span)?)
            }
            DifferentialOperator::Curl => {
                let flux_trace = arena.trace(operand, TraceSide::Exterior, region.span);
                (flux_trace, arena.pair(flux_trace, test_trace, region.span)?)
            }
            _ => unreachable!("only grad/div/curl boundary terms are derived"),
        };
        let formal = arena.with_sign(formal, sign, region.span);
        let condition = model.declarations.iter().find(|declaration| {
            matches!(
                declaration.kind,
                SemanticDeclarationKind::BoundaryCondition {
                    region: condition_region,
                    target: Some(target),
                    ..
                } if condition_region == region.id && target == test_field
            )
        });
        match condition.map(|condition| (&condition.kind, condition.id)) {
            Some((
                SemanticDeclarationKind::BoundaryCondition {
                    condition: BoundaryConditionKind::Dirichlet,
                    ..
                },
                declaration,
            )) => {
                transformations
                    .push(FormTransformation::EliminateEssentialBoundaryTerm { declaration });
                let assumption = FormAssumption::TestTraceVanishes {
                    argument,
                    region: region.id,
                    condition: declaration,
                };
                if !assumptions.contains(&assumption) {
                    assumptions.push(assumption);
                }
                boundary_terms.push(BoundaryTermReceipt {
                    region: region.id,
                    source,
                    integrand: formal,
                    disposition: BoundaryTermDisposition::EliminatedByEssentialCondition {
                        declaration,
                    },
                });
            }
            Some((
                SemanticDeclarationKind::BoundaryCondition {
                    condition: BoundaryConditionKind::Neumann,
                    value,
                    ..
                },
                declaration,
            )) => {
                if !substituted_neumann_fluxes.insert((region.id, test_field)) {
                    return Err(FormCompileError::AmbiguousNeumannFlux {
                        equation: equation_name.to_owned(),
                        field: test_field,
                        region: region.id,
                    });
                }
                let boundary_test = match operator {
                    DifferentialOperator::Divergence | DifferentialOperator::Curl => test_trace,
                    DifferentialOperator::Gradient => {
                        arena.normal_component(test, TraceSide::Exterior, region.span)?
                    }
                    _ => unreachable!("only grad/div/curl boundary terms are derived"),
                };
                let value = lift_provider_calls(arena, *value, boundary_captures)?;
                let substituted = arena.pair(value, boundary_test, region.span)?;
                let substituted = arena.with_sign(substituted, sign, region.span);
                let integral_index = integrals.len();
                integrals.push(VariationalIntegral {
                    measure: SemanticMeasure::ExteriorFacet { region: region.id },
                    side: FormSide::Exterior,
                    integrand: substituted,
                    source_span: region.span,
                });
                transformations
                    .push(FormTransformation::SubstituteBoundaryCondition { declaration });
                boundary_terms.push(BoundaryTermReceipt {
                    region: region.id,
                    source,
                    integrand: formal,
                    disposition: BoundaryTermDisposition::Substituted {
                        declaration,
                        integral_index,
                    },
                });
            }
            Some((
                SemanticDeclarationKind::BoundaryCondition {
                    condition: BoundaryConditionKind::Robin,
                    ..
                },
                _,
            )) => {
                return Err(FormCompileError::UnsupportedStrongForm {
                    code: "FORM_UNSUPPORTED_ROBIN_TRANSFORMATION",
                    equation: equation_name.to_owned(),
                    detail: format!(
                        "Robin condition on region {} needs an explicit flux law",
                        region.id
                    ),
                });
            }
            _ => {
                // Natural closure: the flux datum is zero, so the substituted integral
                // vanishes and nothing is emitted (see `BoundaryTermDisposition::NaturallyClosed`).
                let flux = arena.with_sign(flux, sign, region.span);
                transformations
                    .push(FormTransformation::SubstituteNaturalClosure { region: region.id });
                boundary_terms.push(BoundaryTermReceipt {
                    region: region.id,
                    source,
                    integrand: formal,
                    disposition: BoundaryTermDisposition::NaturallyClosed { flux },
                });
            }
        }
    }
    Ok(())
}

/// Batch P (generalizing GX-facet's bare-boundary-value rule): lift every `ProviderCall` inside
/// `id` into a compiler-synthesized property symbol whose `FormCapture.definition` is the
/// original call node, exactly what `property k = thermal_conductivity(T);` records for a named
/// property. FC3 then classifies the symbol as a `ModelDefinedProperty` input and FC4 binds it
/// as an External input with GX-A3's chain-rule tangent when the provider is differentiable and
/// scalar; a vector-valued call (`convect(u, u)`) stays a frozen coefficient, named truthfully
/// in the derivative receipt. Ancestors of a lifted call are rebuilt as new arena nodes; no
/// pre-existing node is mutated, so every `definition` id still addresses the model arena.
fn lift_provider_calls(
    arena: &mut FormArena,
    id: ExprId,
    captures: &mut Vec<FormCapture>,
) -> Result<ExprId, FormCompileError> {
    let expression = arena
        .expressions
        .get(id.index())
        .ok_or(FormCompileError::InvalidExpression(id))?
        .clone();
    if let SemanticExprKind::ProviderCall { .. } = expression.kind {
        let symbol = SymbolId::generated_for_expression(id);
        let mut ty = expression.ty.clone();
        ty.role = SemanticRole::Property;
        if !captures.iter().any(|capture| capture.symbol == symbol) {
            captures.push(FormCapture {
                symbol,
                role: FormCaptureRole::Property,
                ty: ty.clone(),
                domain: None,
                space: None,
                source_span: expression.span,
                definition: Some(id),
            });
        }
        return Ok(arena.push(SemanticExprKind::Symbol { symbol }, ty, expression.span));
    }
    let children = expression_children(&expression.kind);
    let mut lifted = Vec::with_capacity(children.len());
    for child in &children {
        lifted.push(lift_provider_calls(arena, *child, captures)?);
    }
    if lifted == children {
        return Ok(id);
    }
    let kind = replace_children(&expression.kind, &lifted);
    Ok(arena.push(kind, expression.ty, expression.span))
}

/// Rebuild `kind` with its children (in [`expression_children`] order) replaced by `children`.
fn replace_children(kind: &SemanticExprKind, children: &[ExprId]) -> SemanticExprKind {
    match kind {
        SemanticExprKind::Unary { op, .. } => SemanticExprKind::Unary {
            op: *op,
            arg: children[0],
        },
        SemanticExprKind::Differential { operator, .. } => SemanticExprKind::Differential {
            operator: *operator,
            arg: children[0],
        },
        SemanticExprKind::TensorTrace { axes, .. } => SemanticExprKind::TensorTrace {
            value: children[0],
            axes: *axes,
        },
        SemanticExprKind::FacetTrace { side, .. } => SemanticExprKind::FacetTrace {
            value: children[0],
            side: *side,
        },
        SemanticExprKind::Jump { .. } => SemanticExprKind::Jump { value: children[0] },
        SemanticExprKind::Average { .. } => SemanticExprKind::Average { value: children[0] },
        SemanticExprKind::Conjugate { .. } => SemanticExprKind::Conjugate { value: children[0] },
        SemanticExprKind::NormalComponent { side, .. } => SemanticExprKind::NormalComponent {
            value: children[0],
            side: *side,
        },
        SemanticExprKind::Binary { op, .. } => SemanticExprKind::Binary {
            op: *op,
            lhs: children[0],
            rhs: children[1],
        },
        SemanticExprKind::Contraction {
            axes,
            conjugate_lhs,
            ..
        } => SemanticExprKind::Contraction {
            lhs: children[0],
            rhs: children[1],
            axes: axes.clone(),
            conjugate_lhs: *conjugate_lhs,
        },
        SemanticExprKind::Call { function, .. } => SemanticExprKind::Call {
            function: function.clone(),
            args: children.to_vec(),
        },
        SemanticExprKind::ProviderCall { provider, .. } => SemanticExprKind::ProviderCall {
            provider: *provider,
            args: children.to_vec(),
        },
        SemanticExprKind::Vector { .. } => SemanticExprKind::Vector {
            elements: children.to_vec(),
        },
        SemanticExprKind::Index { .. } => SemanticExprKind::Index {
            value: children[0],
            indices: children[1..].to_vec(),
        },
        SemanticExprKind::Number { .. }
        | SemanticExprKind::String { .. }
        | SemanticExprKind::Symbol { .. } => kind.clone(),
    }
}

struct FormArena {
    expressions: Vec<SemanticExpr>,
}

impl FormArena {
    fn new(expressions: Vec<SemanticExpr>) -> Self {
        Self { expressions }
    }

    fn push(&mut self, kind: SemanticExprKind, ty: SemanticType, span: SourceSpan) -> ExprId {
        let id = ExprId(self.expressions.len() as u32);
        self.expressions.push(SemanticExpr { id, kind, ty, span });
        id
    }

    fn refine_deferred_shape(&mut self, id: ExprId, shape: SemanticShape) {
        let expression = &mut self.expressions[id.index()];
        if matches!(expression.ty.shape, SemanticShape::Deferred)
            && !matches!(shape, SemanticShape::Deferred)
        {
            expression.ty.axes = axes_for_shape(&shape);
            expression.ty.shape = shape;
        }
    }

    fn with_sign(&mut self, value: ExprId, sign: i8, span: SourceSpan) -> ExprId {
        if sign >= 0 {
            value
        } else {
            let ty = self.expressions[value.index()].ty.clone();
            self.push(
                SemanticExprKind::Unary {
                    op: UnaryOp::Neg,
                    arg: value,
                },
                ty,
                span,
            )
        }
    }

    fn differential(
        &mut self,
        operator: DifferentialOperator,
        arg: ExprId,
        domain: DomainId,
        spatial_dimension: u8,
        span: SourceSpan,
    ) -> Result<ExprId, FormCompileError> {
        let source = self
            .expressions
            .get(arg.index())
            .ok_or(FormCompileError::InvalidExpression(arg))?;
        let shape = match (&source.ty.shape, operator) {
            (SemanticShape::Numeric(ValueShape::Scalar), DifferentialOperator::Gradient) => {
                if spatial_dimension == 1 {
                    SemanticShape::Numeric(ValueShape::Scalar)
                } else {
                    SemanticShape::Numeric(ValueShape::Vector(spatial_dimension))
                }
            }
            (SemanticShape::Numeric(ValueShape::Vector(_)), DifferentialOperator::Divergence) => {
                SemanticShape::Numeric(ValueShape::Scalar)
            }
            (
                SemanticShape::Numeric(ValueShape::Tensor { rows, .. })
                | SemanticShape::Numeric(ValueShape::SymmetricTensor(rows)),
                DifferentialOperator::Divergence,
            ) => SemanticShape::Numeric(ValueShape::Vector(*rows)),
            (SemanticShape::Numeric(ValueShape::Vector(rows)), DifferentialOperator::Gradient) => {
                SemanticShape::Numeric(ValueShape::Tensor {
                    rows: *rows,
                    cols: spatial_dimension,
                })
            }
            (SemanticShape::Numeric(ValueShape::Vector(2)), DifferentialOperator::Curl)
                if spatial_dimension == 2 =>
            {
                SemanticShape::Numeric(ValueShape::Scalar)
            }
            (SemanticShape::Numeric(ValueShape::Vector(3)), DifferentialOperator::Curl)
                if spatial_dimension == 3 =>
            {
                SemanticShape::Numeric(ValueShape::Vector(3))
            }
            (SemanticShape::Numeric(ValueShape::Scalar), DifferentialOperator::Curl)
                if spatial_dimension == 2 =>
            {
                SemanticShape::Numeric(ValueShape::Vector(2))
            }
            (SemanticShape::Deferred, _) => SemanticShape::Deferred,
            _ => {
                return Err(FormCompileError::InvalidDerivedDifferential {
                    operator,
                    shape: source.ty.shape.clone(),
                });
            }
        };
        let ty = SemanticType {
            axes: axes_for_shape(&shape),
            shape,
            dimension: source.ty.dimension.and_then(|dimension| {
                dimension
                    .checked_quotient(quantitas::Dimension::LENGTH)
                    .ok()
            }),
            quantity_kind: None,
            frame: Frame::Domain(domain),
            role: SemanticRole::Intrinsic,
        };
        Ok(self.push(SemanticExprKind::Differential { operator, arg }, ty, span))
    }

    fn multiply(
        &mut self,
        lhs: ExprId,
        rhs: ExprId,
        span: SourceSpan,
    ) -> Result<ExprId, FormCompileError> {
        let left = self
            .expressions
            .get(lhs.index())
            .ok_or(FormCompileError::InvalidExpression(lhs))?;
        let right = self
            .expressions
            .get(rhs.index())
            .ok_or(FormCompileError::InvalidExpression(rhs))?;
        let shape = match (&left.ty.shape, &right.ty.shape) {
            (SemanticShape::Numeric(ValueShape::Scalar), other)
            | (other, SemanticShape::Numeric(ValueShape::Scalar)) => other.clone(),
            (left, right) if left == right => left.clone(),
            (SemanticShape::Deferred, _) | (_, SemanticShape::Deferred) => SemanticShape::Deferred,
            _ => SemanticShape::Deferred,
        };
        let ty = product_type(&left.ty, &right.ty, shape);
        Ok(self.push(
            SemanticExprKind::Binary {
                op: BinaryOp::Mul,
                lhs,
                rhs,
            },
            ty,
            span,
        ))
    }

    fn pair(
        &mut self,
        lhs: ExprId,
        rhs: ExprId,
        span: SourceSpan,
    ) -> Result<ExprId, FormCompileError> {
        let left = self
            .expressions
            .get(lhs.index())
            .ok_or(FormCompileError::InvalidExpression(lhs))?;
        let right = self
            .expressions
            .get(rhs.index())
            .ok_or(FormCompileError::InvalidExpression(rhs))?;
        if matches!(left.ty.shape, SemanticShape::Numeric(ValueShape::Scalar))
            || matches!(right.ty.shape, SemanticShape::Numeric(ValueShape::Scalar))
            || (left.ty.axes.is_empty() && right.ty.axes.is_empty())
        {
            return self.multiply(lhs, rhs, span);
        }
        self.contract(lhs, rhs, true, span)
    }

    fn contract(
        &mut self,
        lhs: ExprId,
        rhs: ExprId,
        all_axes: bool,
        span: SourceSpan,
    ) -> Result<ExprId, FormCompileError> {
        let left = self
            .expressions
            .get(lhs.index())
            .ok_or(FormCompileError::InvalidExpression(lhs))?;
        let right = self
            .expressions
            .get(rhs.index())
            .ok_or(FormCompileError::InvalidExpression(rhs))?;
        let count = if all_axes {
            match (left.ty.axes.len(), right.ty.axes.len()) {
                (0, right) => right,
                (left, 0) => left,
                (left, right) => left.min(right),
            }
        } else {
            1
        };
        let ty = product_type(
            &left.ty,
            &right.ty,
            SemanticShape::Numeric(ValueShape::Scalar),
        );
        Ok(self.push(
            SemanticExprKind::Contraction {
                lhs,
                rhs,
                axes: (0..count)
                    .map(|axis| AxisContraction {
                        lhs: axis as u8,
                        rhs: axis as u8,
                    })
                    .collect(),
                conjugate_lhs: false,
            },
            ty,
            span,
        ))
    }

    fn trace(&mut self, value: ExprId, side: TraceSide, span: SourceSpan) -> ExprId {
        let mut ty = self.expressions[value.index()].ty.clone();
        ty.frame = Frame::Neutral;
        ty.role = SemanticRole::Intrinsic;
        self.push(SemanticExprKind::FacetTrace { value, side }, ty, span)
    }

    fn normal_component(
        &mut self,
        value: ExprId,
        side: TraceSide,
        span: SourceSpan,
    ) -> Result<ExprId, FormCompileError> {
        let source = self
            .expressions
            .get(value.index())
            .ok_or(FormCompileError::InvalidExpression(value))?;
        let shape = match source.ty.shape {
            SemanticShape::Numeric(ValueShape::Vector(_)) => {
                SemanticShape::Numeric(ValueShape::Scalar)
            }
            SemanticShape::Numeric(ValueShape::Tensor { rows, .. })
            | SemanticShape::Numeric(ValueShape::SymmetricTensor(rows)) => {
                SemanticShape::Numeric(ValueShape::Vector(rows))
            }
            SemanticShape::Deferred => SemanticShape::Deferred,
            _ => SemanticShape::Deferred,
        };
        let mut ty = source.ty.clone();
        ty.shape = shape;
        ty.axes = axes_for_shape(&ty.shape);
        ty.frame = Frame::Neutral;
        ty.role = SemanticRole::Intrinsic;
        Ok(self.push(SemanticExprKind::NormalComponent { value, side }, ty, span))
    }
}

fn product_type(left: &SemanticType, right: &SemanticType, shape: SemanticShape) -> SemanticType {
    SemanticType {
        axes: axes_for_shape(&shape),
        shape,
        dimension: match (left.dimension, right.dimension) {
            (Some(left), Some(right)) => left.checked_product(right).ok(),
            _ => None,
        },
        quantity_kind: None,
        frame: match (&left.frame, &right.frame) {
            (Frame::Neutral, frame) | (frame, Frame::Neutral) => frame.clone(),
            (frame, _) => frame.clone(),
        },
        role: SemanticRole::Intrinsic,
    }
}

fn axes_for_shape(shape: &SemanticShape) -> Vec<crate::semantic::Axis> {
    match shape {
        SemanticShape::Numeric(ValueShape::Scalar) => vec![],
        SemanticShape::Numeric(ValueShape::Vector(extent)) => vec![crate::semantic::Axis {
            position: 0,
            extent: *extent,
        }],
        SemanticShape::Numeric(ValueShape::Tensor { rows, cols }) => vec![
            crate::semantic::Axis {
                position: 0,
                extent: *rows,
            },
            crate::semantic::Axis {
                position: 1,
                extent: *cols,
            },
        ],
        SemanticShape::Numeric(ValueShape::SymmetricTensor(extent)) => vec![
            crate::semantic::Axis {
                position: 0,
                extent: *extent,
            },
            crate::semantic::Axis {
                position: 1,
                extent: *extent,
            },
        ],
        _ => vec![],
    }
}

#[derive(Serialize)]
struct FormDigestPayload<'a> {
    schema: &'static str,
    model: &'a str,
    name: &'a str,
    source_semantic_digest: &'a Digest,
    declaration: DeclarationId,
    arity: FormArity,
    arguments: &'a [FormArgument],
    captures: &'a [FormCapture],
    expressions: &'a [SemanticExpr],
    integrals: &'a [VariationalIntegral],
    providers: &'a [SemanticProvider],
    receipt: &'a FormReceipt,
}

fn side_for_measure(measure: &SemanticMeasure) -> FormSide {
    match measure {
        SemanticMeasure::Cell { .. } => FormSide::Cell,
        SemanticMeasure::ExteriorFacet { .. } => FormSide::Exterior,
        SemanticMeasure::InteriorFacet { .. } => FormSide::Interior,
        SemanticMeasure::Interface { .. } => FormSide::Interface,
        SemanticMeasure::Point { .. } => FormSide::Point,
    }
}

fn validate_form_sides(
    form: &str,
    expressions: &[SemanticExpr],
    integrals: &[VariationalIntegral],
) -> Result<(), FormCompileError> {
    for integral in integrals {
        let mut visited = BTreeSet::new();
        validate_expression_side(
            form,
            expressions,
            integral.integrand,
            integral.side,
            false,
            &mut visited,
        )?;
    }
    Ok(())
}

fn validate_expression_side(
    form: &str,
    expressions: &[SemanticExpr],
    id: ExprId,
    integral_side: FormSide,
    explicitly_restricted: bool,
    visited: &mut BTreeSet<(ExprId, bool)>,
) -> Result<(), FormCompileError> {
    if !visited.insert((id, explicitly_restricted)) {
        return Ok(());
    }
    let expression = expressions
        .get(id.index())
        .ok_or(FormCompileError::InvalidExpression(id))?;
    let invalid = |detail: String| FormCompileError::InvalidSideSemantics {
        code: "FORM_INVALID_SIDE",
        form: form.to_owned(),
        detail,
    };
    match &expression.kind {
        SemanticExprKind::FacetTrace { value, side }
        | SemanticExprKind::NormalComponent { value, side } => {
            match integral_side {
                FormSide::Exterior if *side != TraceSide::Exterior => {
                    return Err(invalid(format!(
                        "exterior-facet integral uses {side:?} trace"
                    )));
                }
                FormSide::Interior | FormSide::Interface
                    if !matches!(side, TraceSide::Minus | TraceSide::Plus) =>
                {
                    return Err(invalid(
                        "two-sided facet integral requires explicit minus/plus trace".into(),
                    ));
                }
                FormSide::Cell | FormSide::Point => {
                    return Err(invalid(
                        "facet operation used outside a facet integral".into(),
                    ));
                }
                _ => {}
            }
            validate_expression_side(form, expressions, *value, integral_side, true, visited)?;
        }
        SemanticExprKind::Jump { value } | SemanticExprKind::Average { value } => {
            if !matches!(integral_side, FormSide::Interior | FormSide::Interface) {
                return Err(invalid(
                    "jump/average requires an interior-facet or interface integral".into(),
                ));
            }
            validate_expression_side(form, expressions, *value, integral_side, true, visited)?;
        }
        SemanticExprKind::Unary { arg, .. }
        | SemanticExprKind::Differential { arg, .. }
        | SemanticExprKind::TensorTrace { value: arg, .. }
        | SemanticExprKind::Conjugate { value: arg } => {
            validate_expression_side(
                form,
                expressions,
                *arg,
                integral_side,
                explicitly_restricted,
                visited,
            )?;
        }
        SemanticExprKind::Binary { lhs, rhs, .. }
        | SemanticExprKind::Contraction { lhs, rhs, .. } => {
            validate_expression_side(
                form,
                expressions,
                *lhs,
                integral_side,
                explicitly_restricted,
                visited,
            )?;
            validate_expression_side(
                form,
                expressions,
                *rhs,
                integral_side,
                explicitly_restricted,
                visited,
            )?;
        }
        SemanticExprKind::Call { args, .. }
        | SemanticExprKind::ProviderCall { args, .. }
        | SemanticExprKind::Vector { elements: args } => {
            for arg in args {
                validate_expression_side(
                    form,
                    expressions,
                    *arg,
                    integral_side,
                    explicitly_restricted,
                    visited,
                )?;
            }
        }
        SemanticExprKind::Index { value, indices } => {
            validate_expression_side(
                form,
                expressions,
                *value,
                integral_side,
                explicitly_restricted,
                visited,
            )?;
            for index in indices {
                validate_expression_side(
                    form,
                    expressions,
                    *index,
                    integral_side,
                    explicitly_restricted,
                    visited,
                )?;
            }
        }
        SemanticExprKind::Symbol { .. }
            if matches!(integral_side, FormSide::Interior | FormSide::Interface)
                && !explicitly_restricted
                && matches!(expression.ty.frame, Frame::Domain(_)) =>
        {
            return Err(invalid(
                "two-sided facet field reference requires trace, jump, or average".into(),
            ));
        }
        SemanticExprKind::Number { .. }
        | SemanticExprKind::String { .. }
        | SemanticExprKind::Symbol { .. } => {}
    }
    Ok(())
}

fn expression_children(kind: &SemanticExprKind) -> Vec<ExprId> {
    match kind {
        SemanticExprKind::Unary { arg, .. }
        | SemanticExprKind::Differential { arg, .. }
        | SemanticExprKind::TensorTrace { value: arg, .. }
        | SemanticExprKind::FacetTrace { value: arg, .. }
        | SemanticExprKind::Jump { value: arg }
        | SemanticExprKind::Average { value: arg }
        | SemanticExprKind::Conjugate { value: arg }
        | SemanticExprKind::NormalComponent { value: arg, .. } => vec![*arg],
        SemanticExprKind::Binary { lhs, rhs, .. }
        | SemanticExprKind::Contraction { lhs, rhs, .. } => vec![*lhs, *rhs],
        SemanticExprKind::Call { args, .. }
        | SemanticExprKind::ProviderCall { args, .. }
        | SemanticExprKind::Vector { elements: args } => args.clone(),
        SemanticExprKind::Index { value, indices } => {
            let mut children = Vec::with_capacity(indices.len() + 1);
            children.push(*value);
            children.extend(indices.iter().copied());
            children
        }
        SemanticExprKind::Number { .. }
        | SemanticExprKind::String { .. }
        | SemanticExprKind::Symbol { .. } => vec![],
    }
}

fn collect_symbol_ids(
    expressions: &[SemanticExpr],
    id: ExprId,
    visited: &mut BTreeSet<ExprId>,
    symbols: &mut BTreeSet<SymbolId>,
) -> Result<(), FormCompileError> {
    if !visited.insert(id) {
        return Ok(());
    }
    let expression = expressions
        .get(id.index())
        .ok_or(FormCompileError::InvalidExpression(id))?;
    if let SemanticExprKind::Symbol { symbol } = expression.kind {
        symbols.insert(symbol);
    }
    for child in expression_children(&expression.kind) {
        collect_symbol_ids(expressions, child, visited, symbols)?;
    }
    Ok(())
}

fn collect_symbols(
    model: &SemanticModel,
    expression: ExprId,
    visited: &mut BTreeSet<ExprId>,
    symbols: &mut BTreeSet<SymbolId>,
) -> Result<(), FormCompileError> {
    if !visited.insert(expression) {
        return Ok(());
    }
    let expression = model
        .expressions
        .get(expression.index())
        .ok_or(FormCompileError::InvalidExpression(expression))?;
    match &expression.kind {
        SemanticExprKind::Symbol { symbol } => {
            symbols.insert(*symbol);
        }
        SemanticExprKind::Unary { arg, .. } => {
            collect_symbols(model, *arg, visited, symbols)?;
        }
        SemanticExprKind::Differential { arg, .. }
        | SemanticExprKind::TensorTrace { value: arg, .. }
        | SemanticExprKind::FacetTrace { value: arg, .. }
        | SemanticExprKind::Jump { value: arg }
        | SemanticExprKind::Average { value: arg }
        | SemanticExprKind::Conjugate { value: arg }
        | SemanticExprKind::NormalComponent { value: arg, .. } => {
            collect_symbols(model, *arg, visited, symbols)?;
        }
        SemanticExprKind::Binary { lhs, rhs, .. } => {
            collect_symbols(model, *lhs, visited, symbols)?;
            collect_symbols(model, *rhs, visited, symbols)?;
        }
        SemanticExprKind::Contraction { lhs, rhs, .. } => {
            collect_symbols(model, *lhs, visited, symbols)?;
            collect_symbols(model, *rhs, visited, symbols)?;
        }
        SemanticExprKind::Call { args, .. }
        | SemanticExprKind::ProviderCall { args, .. }
        | SemanticExprKind::Vector { elements: args } => {
            for argument in args {
                collect_symbols(model, *argument, visited, symbols)?;
            }
        }
        SemanticExprKind::Index { value, indices } => {
            collect_symbols(model, *value, visited, symbols)?;
            for index in indices {
                collect_symbols(model, *index, visited, symbols)?;
            }
        }
        SemanticExprKind::Number { .. } | SemanticExprKind::String { .. } => {}
    }
    Ok(())
}

fn capture_role(role: &SemanticRole) -> Option<FormCaptureRole> {
    match role {
        SemanticRole::PhysicalField(role) => Some(FormCaptureRole::PhysicalField(role.clone())),
        SemanticRole::Parameter => Some(FormCaptureRole::Parameter),
        SemanticRole::Constant => Some(FormCaptureRole::Constant),
        SemanticRole::Source => Some(FormCaptureRole::Source),
        SemanticRole::Property => Some(FormCaptureRole::Property),
        SemanticRole::ConstitutiveLaw => Some(FormCaptureRole::ConstitutiveLaw),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile_semantics;
    use quantitas::UnitRegistry;

    #[test]
    fn authored_form_uses_typed_arena_identities_and_roles() {
        let compilation = compile_semantics(
            r#"
module form.test;
model Poisson {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: trial scalar H1(order=1) on Omega;
  field v: test scalar H1(order=1) on Omega;
  parameter alpha: Diffusivity;
  form residual { cell(Omega): alpha * u * v; }
}
"#,
            &UnitRegistry::si_bootstrap(),
        )
        .unwrap();
        let form = compile_variational_form(&compilation.semantic, "Poisson", "residual").unwrap();
        assert_eq!(form.arguments.len(), 2);
        assert_eq!(form.arguments[0].role, FormArgumentRole::Trial);
        assert_eq!(form.arguments[1].role, FormArgumentRole::Test);
        assert_eq!(form.captures.len(), 1);
        assert_eq!(form.captures[0].role, FormCaptureRole::Parameter);
        assert_eq!(form.integrals.len(), 1);
        assert_eq!(form.receipt.transformations, [FormTransformation::Authored]);
        assert_eq!(form.artifact_digest.algorithm, "blake3");
    }
}
