//! Typed sibling method-family compilers.
//!
//! These artifacts start from the same [`SemanticModule`] as variational forms, but they do not
//! pass through `VariationalForm`, form requirements, or FEM operator factorization. Concrete
//! topology and global actions remain downstream in Finitum.

use crate::id::{Digest, span_independent_digest};
use crate::scientific::{FieldRole, SpaceFamily, ValueShape};
use crate::{
    DeclarationId, DifferentialOperator, DomainId, ExprId, RegionId, SemanticDeclaration,
    SemanticDeclarationKind, SemanticExpr, SemanticExprKind, SemanticModel, SemanticModule,
    SemanticRole, SemanticType, SymbolId, semantic_arena_digest,
};
use malleus::{
    AccessMode, BinaryOp as MalleusBinaryOp, IndexingMap, IterationDomain, KernelOperand,
    KernelRegion, NumericPolicy, OperandId, ScalarExpr, Statement, StructuredKernel,
    StructuredModule, validate_module,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::sync::Arc;
use thiserror::Error;

pub const METHOD_PROGRAM_SCHEMA: &str = "scientia-method-program/2";
pub const AFFINE_METHOD_KERNEL_SCHEMA: &str = "scientia-affine-method-kernel/1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MethodFamily {
    ConservationLawFiniteVolume,
    StructuredStencilFiniteDifference,
    NetworkDae,
    Particle,
    BoundaryIntegral,
}

impl MethodFamily {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ConservationLawFiniteVolume => "conservation_law_finite_volume",
            Self::StructuredStencilFiniteDifference => "structured_stencil_finite_difference",
            Self::NetworkDae => "network_dae",
            Self::Particle => "particle",
            Self::BoundaryIntegral => "boundary_integral",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AffineMethodKernelSpec {
    pub name: String,
    pub inputs: Vec<String>,
    pub coefficients: Vec<f64>,
    pub constant: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AffineMethodKernel {
    pub schema: String,
    pub artifact_digest: Digest,
    pub spec: AffineMethodKernelSpec,
    pub module: StructuredModule,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MethodReceipt {
    pub family: MethodFamily,
    pub source_semantic_digest: Digest,
    pub equations: Vec<DeclarationId>,
    pub state_symbols: Vec<SymbolId>,
    pub selection: MethodSelectionReceipt,
    pub selected_without_variational_form: bool,
    pub local_kernel_digest: Option<Digest>,
}

/// Family-specific structural matches that made a method program eligible for selection.
///
/// These identities make the receipt independently auditable without traversing the semantic
/// expression arena again.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
pub enum MethodSelectionReceipt {
    ConservationLawFiniteVolume {
        time_derivative: ExprId,
        flux_divergence: ExprId,
        flux: ExprId,
    },
    StructuredStencilFiniteDifference {
        spatial_differential: ExprId,
    },
    NetworkDae {
        differential_states: Vec<SymbolId>,
    },
    Particle {
        positions: SymbolId,
        velocities: SymbolId,
        pair_force: SymbolId,
    },
    BoundaryIntegral {
        boundary_operator: ExprId,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MethodStateBinding {
    pub symbol: SymbolId,
    pub ty: SemanticType,
    pub domain: DomainId,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConservationLawMethod {
    pub domain: DomainId,
    pub equation: DeclarationId,
    pub conserved: SymbolId,
    pub time_derivative: ExprId,
    pub flux_divergence: ExprId,
    pub flux: ExprId,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FiniteDifferenceMethod {
    pub domain: DomainId,
    pub equation: DeclarationId,
    pub state: SymbolId,
    pub spatial_differential: ExprId,
    pub offsets: Vec<isize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkDaeMethod {
    pub domain: DomainId,
    pub equations: Vec<DeclarationId>,
    pub states: Vec<SymbolId>,
    pub state_components: Vec<usize>,
    pub differential_states: Vec<SymbolId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParticleMethod {
    pub domain: DomainId,
    pub equations: Vec<DeclarationId>,
    pub positions: SymbolId,
    pub velocities: SymbolId,
    pub pair_force: SymbolId,
    pub position_components: usize,
    pub velocity_components: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundaryIntegralMethod {
    pub domain: DomainId,
    pub equation: DeclarationId,
    pub unknown: SymbolId,
    pub boundary_region: RegionId,
    pub boundary_operator: ExprId,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "family", rename_all = "snake_case")]
pub enum MethodProgramKind {
    ConservationLawFiniteVolume(ConservationLawMethod),
    StructuredStencilFiniteDifference(FiniteDifferenceMethod),
    NetworkDae(NetworkDaeMethod),
    Particle(ParticleMethod),
    BoundaryIntegral(BoundaryIntegralMethod),
}

impl MethodProgramKind {
    pub const fn family(&self) -> MethodFamily {
        match self {
            Self::ConservationLawFiniteVolume(_) => MethodFamily::ConservationLawFiniteVolume,
            Self::StructuredStencilFiniteDifference(_) => {
                MethodFamily::StructuredStencilFiniteDifference
            }
            Self::NetworkDae(_) => MethodFamily::NetworkDae,
            Self::Particle(_) => MethodFamily::Particle,
            Self::BoundaryIntegral(_) => MethodFamily::BoundaryIntegral,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MethodProgram {
    pub schema: String,
    pub model: String,
    pub source_semantic_digest: Digest,
    pub artifact_digest: Digest,
    pub kind: MethodProgramKind,
    pub state_bindings: Vec<MethodStateBinding>,
    pub expressions: Arc<[SemanticExpr]>,
    pub local_kernel: Option<AffineMethodKernel>,
    pub receipt: MethodReceipt,
}

impl MethodProgram {
    pub const fn family(&self) -> MethodFamily {
        self.kind.family()
    }
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum MethodCompileError {
    #[error("METHOD_MODEL_NOT_FOUND: model `{model}` is absent")]
    MissingModel { model: String },
    #[error("METHOD_DECLARATION_NOT_FOUND: {role} `{name}` is absent from model `{model}`")]
    MissingDeclaration {
        model: String,
        role: &'static str,
        name: String,
    },
    #[error("METHOD_DOMAIN_MISMATCH: {0}")]
    DomainMismatch(String),
    #[error("METHOD_UNSUPPORTED_EQUATION: {0}")]
    UnsupportedEquation(String),
    #[error("METHOD_INVALID_KERNEL: {0}")]
    InvalidKernel(String),
}

pub fn compile_conservation_law_method(
    module: &SemanticModule,
    model_name: &str,
    equation_name: &str,
    conserved_name: &str,
    numerical_flux: AffineMethodKernelSpec,
) -> Result<MethodProgram, MethodCompileError> {
    let model = find_model(module, model_name)?;
    let equation = find_declaration(model, equation_name, "equation", |role| {
        matches!(role, SemanticRole::Equation)
    })?;
    let conserved = find_physical_field(model, conserved_name)?;
    let domain = shared_domain(model, conserved, equation)?;
    let domain_data = &model.domains[domain.index()];
    if domain_data.spatial_dimension == 0 {
        return Err(MethodCompileError::DomainMismatch(format!(
            "finite-volume conservation requires a spatial domain; `{}` has dimension zero",
            domain_data.name
        )));
    }
    let symbol = &model.symbols[conserved.index()];
    if !matches!(
        symbol.space.as_ref().map(|space| &space.family),
        Some(SpaceFamily::Dg)
    ) {
        return Err(MethodCompileError::DomainMismatch(format!(
            "finite-volume conserved field `{conserved_name}` must use a discontinuous DG space"
        )));
    }
    let (lhs, rhs) = equation_sides(equation)?;
    let time_derivative = find_differential(
        model,
        [lhs, rhs],
        DifferentialOperator::TimeDerivative,
        Some(conserved),
    )
    .ok_or_else(|| {
        MethodCompileError::UnsupportedEquation(format!(
            "equation `{equation_name}` has no time derivative of `{conserved_name}`"
        ))
    })?;
    let flux_divergence =
        find_differential(model, [lhs, rhs], DifferentialOperator::Divergence, None).ok_or_else(
            || {
                MethodCompileError::UnsupportedEquation(format!(
                    "equation `{equation_name}` has no conservative divergence term"
                ))
            },
        )?;
    let flux = match model.expressions[flux_divergence.index()].kind {
        SemanticExprKind::Differential { arg, .. } => arg,
        _ => unreachable!("differential search returns a differential expression"),
    };
    let kernel = compile_affine_kernel(numerical_flux)?;
    finish(
        module,
        model,
        MethodProgramKind::ConservationLawFiniteVolume(ConservationLawMethod {
            domain,
            equation: equation.id,
            conserved,
            time_derivative,
            flux_divergence,
            flux,
        }),
        vec![equation.id],
        vec![conserved],
        Some(kernel),
    )
}

pub fn compile_finite_difference_method(
    module: &SemanticModule,
    model_name: &str,
    equation_name: &str,
    state_name: &str,
    offsets: Vec<isize>,
    stencil: AffineMethodKernelSpec,
) -> Result<MethodProgram, MethodCompileError> {
    let model = find_model(module, model_name)?;
    let equation = find_declaration(model, equation_name, "equation", |role| {
        matches!(role, SemanticRole::Equation)
    })?;
    let state = find_physical_field(model, state_name)?;
    let domain = shared_domain(model, state, equation)?;
    if model.domains[domain.index()].spatial_dimension == 0 {
        return Err(MethodCompileError::DomainMismatch(
            "finite differences require a positive-dimensional Cartesian-like domain".into(),
        ));
    }
    if offsets.is_empty() || offsets.len() != stencil.inputs.len() {
        return Err(MethodCompileError::InvalidKernel(
            "stencil offsets must be nonempty and match the affine kernel input count".into(),
        ));
    }
    if offsets.iter().copied().collect::<BTreeSet<_>>().len() != offsets.len() {
        return Err(MethodCompileError::InvalidKernel(
            "stencil offsets must be unique".into(),
        ));
    }
    let (lhs, rhs) = equation_sides(equation)?;
    let spatial_differential =
        find_spatial_differential_of(model, [lhs, rhs], state).ok_or_else(|| {
            MethodCompileError::UnsupportedEquation(format!(
                "equation `{equation_name}` has no spatial differential rooted in `{state_name}`"
            ))
        })?;
    let kernel = compile_affine_kernel(stencil)?;
    finish(
        module,
        model,
        MethodProgramKind::StructuredStencilFiniteDifference(FiniteDifferenceMethod {
            domain,
            equation: equation.id,
            state,
            spatial_differential,
            offsets,
        }),
        vec![equation.id],
        vec![state],
        Some(kernel),
    )
}

pub fn compile_network_dae_method(
    module: &SemanticModule,
    model_name: &str,
    equation_names: &[&str],
    state_names: &[&str],
) -> Result<MethodProgram, MethodCompileError> {
    let model = find_model(module, model_name)?;
    let equations = equation_names
        .iter()
        .map(|name| {
            find_declaration(model, name, "equation", |role| {
                matches!(role, SemanticRole::Equation)
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let states = state_names
        .iter()
        .map(|name| find_physical_field(model, name))
        .collect::<Result<Vec<_>, _>>()?;
    require_nonempty(&equations, "network DAE requires at least one equation")?;
    require_nonempty(&states, "network DAE requires at least one state field")?;
    let domain = common_state_domain(model, &states)?;
    if model.domains[domain.index()].spatial_dimension != 0 {
        return Err(MethodCompileError::DomainMismatch(
            "network DAE compilation requires a zero-dimensional lumped domain".into(),
        ));
    }
    for equation in &equations {
        if equation.domain != Some(domain) {
            return Err(MethodCompileError::DomainMismatch(format!(
                "network equation `{}` is not on the selected lumped domain",
                equation.name
            )));
        }
    }
    let differential_states = states
        .iter()
        .copied()
        .filter(|state| {
            equations.iter().any(|equation| {
                equation_sides(equation).is_ok_and(|(lhs, rhs)| {
                    find_differential(
                        model,
                        [lhs, rhs],
                        DifferentialOperator::TimeDerivative,
                        Some(*state),
                    )
                    .is_some()
                })
            })
        })
        .collect::<Vec<_>>();
    if differential_states.is_empty() {
        return Err(MethodCompileError::UnsupportedEquation(
            "network DAE has no time-derivative state; use an algebraic network compiler when that contract exists".into(),
        ));
    }
    finish(
        module,
        model,
        MethodProgramKind::NetworkDae(NetworkDaeMethod {
            domain,
            equations: equations.iter().map(|equation| equation.id).collect(),
            states: states.clone(),
            state_components: states
                .iter()
                .map(|state| symbol_components(model, *state))
                .collect::<Result<Vec<_>, _>>()?,
            differential_states,
        }),
        equations.iter().map(|equation| equation.id).collect(),
        states,
        None,
    )
}

pub fn compile_particle_method(
    module: &SemanticModule,
    model_name: &str,
    equation_names: &[&str],
    positions_name: &str,
    velocities_name: &str,
    pair_force_name: &str,
) -> Result<MethodProgram, MethodCompileError> {
    let model = find_model(module, model_name)?;
    let equations = equation_names
        .iter()
        .map(|name| {
            find_declaration(model, name, "equation", |role| {
                matches!(role, SemanticRole::Equation)
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    require_nonempty(
        &equations,
        "particle compilation requires evolution equations",
    )?;
    let positions = find_physical_field(model, positions_name)?;
    let velocities = find_physical_field(model, velocities_name)?;
    let pair_force = find_symbol(model, pair_force_name, "constitutive law", |role| {
        matches!(role, SemanticRole::ConstitutiveLaw)
    })?;
    let domain = common_state_domain(model, &[positions, velocities])?;
    let domain_data = &model.domains[domain.index()];
    if domain_data.spatial_dimension != 0
        || !matches!(&domain_data.coordinates, crate::scientific::CoordinateSystem::Custom(name) if name == "particle_space")
    {
        return Err(MethodCompileError::DomainMismatch(
            "particle compilation requires a zero-dimensional `particle_space` domain".into(),
        ));
    }
    for state in [positions, velocities] {
        let differentiated = equations.iter().any(|equation| {
            equation_sides(equation).is_ok_and(|(lhs, rhs)| {
                find_differential(
                    model,
                    [lhs, rhs],
                    DifferentialOperator::TimeDerivative,
                    Some(state),
                )
                .is_some()
            })
        });
        if !differentiated {
            return Err(MethodCompileError::UnsupportedEquation(format!(
                "particle state `{}` has no evolution equation",
                model.symbols[state.index()].name
            )));
        }
    }
    let position_components = symbol_components(model, positions)?;
    let velocity_components = symbol_components(model, velocities)?;
    if position_components != velocity_components {
        return Err(MethodCompileError::UnsupportedEquation(
            "particle position and velocity states must have the same component extent".into(),
        ));
    }
    finish(
        module,
        model,
        MethodProgramKind::Particle(ParticleMethod {
            domain,
            equations: equations.iter().map(|equation| equation.id).collect(),
            positions,
            velocities,
            pair_force,
            position_components,
            velocity_components,
        }),
        equations.iter().map(|equation| equation.id).collect(),
        vec![positions, velocities],
        None,
    )
}

pub fn compile_boundary_integral_method(
    module: &SemanticModule,
    model_name: &str,
    equation_name: &str,
    unknown_name: &str,
    boundary_condition_name: &str,
) -> Result<MethodProgram, MethodCompileError> {
    let model = find_model(module, model_name)?;
    let equation = find_declaration(model, equation_name, "equation", |role| {
        matches!(role, SemanticRole::Equation)
    })?;
    let unknown = find_physical_field(model, unknown_name)?;
    let boundary = find_declaration(
        model,
        boundary_condition_name,
        "boundary condition",
        |role| matches!(role, SemanticRole::BoundaryCondition),
    )?;
    let domain = shared_domain(model, unknown, equation)?;
    if model.domains[domain.index()].spatial_dimension < 2 {
        return Err(MethodCompileError::DomainMismatch(
            "boundary-integral compilation requires an ambient domain of dimension at least two"
                .into(),
        ));
    }
    let (boundary_region, boundary_operator, target) = match &boundary.kind {
        SemanticDeclarationKind::BoundaryCondition {
            region,
            target,
            value,
            ..
        } => (*region, *value, *target),
        _ => unreachable!("role-filtered boundary declaration"),
    };
    if target != Some(unknown) {
        return Err(MethodCompileError::UnsupportedEquation(format!(
            "boundary condition `{boundary_condition_name}` does not constrain `{unknown_name}`"
        )));
    }
    finish(
        module,
        model,
        MethodProgramKind::BoundaryIntegral(BoundaryIntegralMethod {
            domain,
            equation: equation.id,
            unknown,
            boundary_region,
            boundary_operator,
        }),
        vec![equation.id],
        vec![unknown],
        None,
    )
}

fn compile_affine_kernel(
    spec: AffineMethodKernelSpec,
) -> Result<AffineMethodKernel, MethodCompileError> {
    if spec.name.trim().is_empty()
        || spec.inputs.is_empty()
        || spec.inputs.len() != spec.coefficients.len()
        || !spec.constant.is_finite()
        || spec
            .coefficients
            .iter()
            .any(|coefficient| !coefficient.is_finite())
        || spec.inputs.iter().any(|input| input.trim().is_empty())
        || spec.inputs.iter().collect::<BTreeSet<_>>().len() != spec.inputs.len()
    {
        return Err(MethodCompileError::InvalidKernel(
            "affine kernel requires a nonempty unique input list, one finite coefficient per input, and a finite constant".into(),
        ));
    }
    let output = OperandId::new(spec.inputs.len());
    let mut expression = ScalarExpr::Constant(spec.constant);
    for (index, coefficient) in spec.coefficients.iter().copied().enumerate() {
        expression = ScalarExpr::binary(
            MalleusBinaryOp::Add,
            expression,
            ScalarExpr::binary(
                MalleusBinaryOp::Mul,
                ScalarExpr::Constant(coefficient),
                ScalarExpr::Load(OperandId::new(index)),
            ),
        );
    }
    let mut operands = spec
        .inputs
        .iter()
        .map(|name| KernelOperand::scalar(name, AccessMode::Read))
        .collect::<Vec<_>>();
    operands.push(KernelOperand::scalar("output", AccessMode::Write));
    let module = StructuredModule {
        name: spec.name.clone(),
        kernels: vec![StructuredKernel {
            name: spec.name.clone(),
            iteration_domain: IterationDomain::default(),
            iterators: Vec::new(),
            indexing_maps: (0..operands.len())
                .map(|index| IndexingMap::scalar(OperandId::new(index)))
                .collect(),
            operands,
            body: KernelRegion {
                statements: vec![Statement::Store {
                    operand: output,
                    value: expression,
                }],
            },
            numeric_policy: NumericPolicy::default(),
        }],
    };
    validate_module(module.clone())
        .map_err(|error| MethodCompileError::InvalidKernel(error.to_string()))?;
    let artifact_digest = span_independent_digest(&KernelDigestPayload {
        schema: AFFINE_METHOD_KERNEL_SCHEMA,
        spec: &spec,
    });
    Ok(AffineMethodKernel {
        schema: AFFINE_METHOD_KERNEL_SCHEMA.into(),
        artifact_digest,
        spec,
        module,
    })
}

fn finish(
    module: &SemanticModule,
    model: &SemanticModel,
    kind: MethodProgramKind,
    equations: Vec<DeclarationId>,
    state_symbols: Vec<SymbolId>,
    local_kernel: Option<AffineMethodKernel>,
) -> Result<MethodProgram, MethodCompileError> {
    let family = kind.family();
    let source_semantic_digest = Digest {
        algorithm: "blake3".into(),
        hex: semantic_arena_digest(module),
    };
    let state_bindings = state_symbols
        .iter()
        .map(|symbol| {
            let source = &model.symbols[symbol.index()];
            Ok(MethodStateBinding {
                symbol: *symbol,
                ty: source.ty.clone(),
                domain: source.domain.ok_or_else(|| {
                    MethodCompileError::DomainMismatch(format!(
                        "method state `{}` has no domain",
                        source.name
                    ))
                })?,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let receipt = MethodReceipt {
        family,
        source_semantic_digest: source_semantic_digest.clone(),
        equations,
        state_symbols: state_symbols.clone(),
        selection: selection_receipt(&kind),
        selected_without_variational_form: true,
        local_kernel_digest: local_kernel
            .as_ref()
            .map(|kernel| kernel.artifact_digest.clone()),
    };
    let artifact_digest = span_independent_digest(&ProgramDigestPayload {
        schema: METHOD_PROGRAM_SCHEMA,
        model: &model.name,
        source_semantic_digest: &source_semantic_digest,
        kind: &kind,
        state_bindings: &state_bindings,
        receipt: &receipt,
    });
    Ok(MethodProgram {
        schema: METHOD_PROGRAM_SCHEMA.into(),
        model: model.name.clone(),
        source_semantic_digest,
        artifact_digest,
        kind,
        state_bindings,
        expressions: Arc::clone(&model.expressions),
        local_kernel,
        receipt,
    })
}

fn selection_receipt(kind: &MethodProgramKind) -> MethodSelectionReceipt {
    match kind {
        MethodProgramKind::ConservationLawFiniteVolume(method) => {
            MethodSelectionReceipt::ConservationLawFiniteVolume {
                time_derivative: method.time_derivative,
                flux_divergence: method.flux_divergence,
                flux: method.flux,
            }
        }
        MethodProgramKind::StructuredStencilFiniteDifference(method) => {
            MethodSelectionReceipt::StructuredStencilFiniteDifference {
                spatial_differential: method.spatial_differential,
            }
        }
        MethodProgramKind::NetworkDae(method) => MethodSelectionReceipt::NetworkDae {
            differential_states: method.differential_states.clone(),
        },
        MethodProgramKind::Particle(method) => MethodSelectionReceipt::Particle {
            positions: method.positions,
            velocities: method.velocities,
            pair_force: method.pair_force,
        },
        MethodProgramKind::BoundaryIntegral(method) => MethodSelectionReceipt::BoundaryIntegral {
            boundary_operator: method.boundary_operator,
        },
    }
}

fn find_model<'a>(
    module: &'a SemanticModule,
    name: &str,
) -> Result<&'a SemanticModel, MethodCompileError> {
    module
        .models
        .iter()
        .find(|model| model.name == name)
        .ok_or_else(|| MethodCompileError::MissingModel { model: name.into() })
}

fn find_declaration<'a>(
    model: &'a SemanticModel,
    name: &str,
    role: &'static str,
    predicate: impl Fn(&SemanticRole) -> bool,
) -> Result<&'a SemanticDeclaration, MethodCompileError> {
    model
        .declarations
        .iter()
        .find(|declaration| declaration.name == name && predicate(&declaration.role))
        .ok_or_else(|| MethodCompileError::MissingDeclaration {
            model: model.name.clone(),
            role,
            name: name.into(),
        })
}

fn find_physical_field(model: &SemanticModel, name: &str) -> Result<SymbolId, MethodCompileError> {
    find_symbol(model, name, "physical field", |role| {
        matches!(
            role,
            SemanticRole::PhysicalField(FieldRole::State | FieldRole::Unknown | FieldRole::Trial)
        )
    })
}

fn find_symbol(
    model: &SemanticModel,
    name: &str,
    role: &'static str,
    predicate: impl Fn(&SemanticRole) -> bool,
) -> Result<SymbolId, MethodCompileError> {
    model
        .symbols
        .iter()
        .find(|symbol| symbol.name == name && predicate(&symbol.ty.role))
        .map(|symbol| symbol.id)
        .ok_or_else(|| MethodCompileError::MissingDeclaration {
            model: model.name.clone(),
            role,
            name: name.into(),
        })
}

fn shared_domain(
    model: &SemanticModel,
    state: SymbolId,
    equation: &SemanticDeclaration,
) -> Result<DomainId, MethodCompileError> {
    let state_domain = model.symbols[state.index()].domain;
    if state_domain.is_none() || state_domain != equation.domain {
        return Err(MethodCompileError::DomainMismatch(format!(
            "field `{}` and equation `{}` do not share one declared domain",
            model.symbols[state.index()].name,
            equation.name
        )));
    }
    Ok(state_domain.expect("checked present"))
}

fn common_state_domain(
    model: &SemanticModel,
    states: &[SymbolId],
) -> Result<DomainId, MethodCompileError> {
    let Some(first) = states
        .first()
        .and_then(|state| model.symbols[state.index()].domain)
    else {
        return Err(MethodCompileError::DomainMismatch(
            "selected states have no domain".into(),
        ));
    };
    if states
        .iter()
        .any(|state| model.symbols[state.index()].domain != Some(first))
    {
        return Err(MethodCompileError::DomainMismatch(
            "selected states span more than one domain".into(),
        ));
    }
    Ok(first)
}

fn symbol_components(model: &SemanticModel, symbol: SymbolId) -> Result<usize, MethodCompileError> {
    match &model.symbols[symbol.index()].ty.shape {
        crate::SemanticShape::Numeric(ValueShape::Scalar) => Ok(1),
        crate::SemanticShape::Numeric(ValueShape::Vector(extent)) => Ok(usize::from(*extent)),
        crate::SemanticShape::Numeric(ValueShape::Tensor { rows, cols }) => {
            Ok(usize::from(*rows) * usize::from(*cols))
        }
        crate::SemanticShape::Numeric(ValueShape::SymmetricTensor(extent)) => {
            Ok(usize::from(*extent) * usize::from(*extent))
        }
        _ => Err(MethodCompileError::UnsupportedEquation(format!(
            "method state `{}` does not have a concrete numeric shape",
            model.symbols[symbol.index()].name
        ))),
    }
}

fn equation_sides(equation: &SemanticDeclaration) -> Result<(ExprId, ExprId), MethodCompileError> {
    match equation.kind {
        SemanticDeclarationKind::Equation { lhs, rhs } => Ok((lhs, rhs)),
        _ => Err(MethodCompileError::UnsupportedEquation(format!(
            "`{}` is not an equation",
            equation.name
        ))),
    }
}

fn find_differential(
    model: &SemanticModel,
    roots: impl IntoIterator<Item = ExprId>,
    operator: DifferentialOperator,
    base: Option<SymbolId>,
) -> Option<ExprId> {
    let mut stack = roots.into_iter().collect::<Vec<_>>();
    let mut visited = BTreeSet::new();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        let expression = &model.expressions[id.index()];
        if let SemanticExprKind::Differential {
            operator: found,
            arg,
        } = expression.kind
            && found == operator
            && base.is_none_or(|symbol| expression_contains_symbol(model, arg, symbol))
        {
            return Some(id);
        }
        children(&expression.kind, &mut stack);
    }
    None
}

fn find_spatial_differential_of(
    model: &SemanticModel,
    roots: impl IntoIterator<Item = ExprId>,
    state: SymbolId,
) -> Option<ExprId> {
    let mut stack = roots.into_iter().collect::<Vec<_>>();
    let mut visited = BTreeSet::new();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        let expression = &model.expressions[id.index()];
        if let SemanticExprKind::Differential { operator, arg } = expression.kind
            && operator != DifferentialOperator::TimeDerivative
            && expression_contains_symbol(model, arg, state)
        {
            return Some(id);
        }
        children(&expression.kind, &mut stack);
    }
    None
}

fn expression_contains_symbol(model: &SemanticModel, root: ExprId, symbol: SymbolId) -> bool {
    let mut stack = vec![root];
    let mut visited = BTreeSet::new();
    while let Some(id) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }
        let expression = &model.expressions[id.index()];
        if matches!(expression.kind, SemanticExprKind::Symbol { symbol: found } if found == symbol)
        {
            return true;
        }
        children(&expression.kind, &mut stack);
    }
    false
}

fn children(kind: &SemanticExprKind, output: &mut Vec<ExprId>) {
    match kind {
        SemanticExprKind::Unary { arg, .. }
        | SemanticExprKind::Differential { arg, .. }
        | SemanticExprKind::TensorTrace { value: arg, .. }
        | SemanticExprKind::FacetTrace { value: arg, .. }
        | SemanticExprKind::Jump { value: arg }
        | SemanticExprKind::Average { value: arg }
        | SemanticExprKind::Conjugate { value: arg }
        | SemanticExprKind::NormalComponent { value: arg, .. } => output.push(*arg),
        SemanticExprKind::Binary { lhs, rhs, .. }
        | SemanticExprKind::Contraction { lhs, rhs, .. } => output.extend([*lhs, *rhs]),
        SemanticExprKind::Call { args, .. } | SemanticExprKind::Vector { elements: args } => {
            output.extend(args.iter().copied())
        }
        SemanticExprKind::Index { value, indices } => {
            output.push(*value);
            output.extend(indices.iter().copied());
        }
        SemanticExprKind::Number { .. }
        | SemanticExprKind::String { .. }
        | SemanticExprKind::Symbol { .. } => {}
    }
}

fn require_nonempty<T>(values: &[T], message: &str) -> Result<(), MethodCompileError> {
    if values.is_empty() {
        Err(MethodCompileError::UnsupportedEquation(message.into()))
    } else {
        Ok(())
    }
}

#[derive(Serialize)]
struct KernelDigestPayload<'a> {
    schema: &'static str,
    spec: &'a AffineMethodKernelSpec,
}

#[derive(Serialize)]
struct ProgramDigestPayload<'a> {
    schema: &'static str,
    model: &'a str,
    source_semantic_digest: &'a Digest,
    kind: &'a MethodProgramKind,
    state_bindings: &'a [MethodStateBinding],
    receipt: &'a MethodReceipt,
}
