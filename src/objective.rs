//! SV1-A: the `.res` producer of [`DerivativeRequest`]s (GX decision 8: objectives are consumed,
//! not invented). `derive_derivative_request` reads an `objective` (or `observable`) declaration
//! and the case-bindable slots a caller names as active or frozen, and returns the frozen
//! `scientia-derivative-request/1` record together with typed links into the semantic arena
//! (`SymbolId`/`ExprId`/`DeclarationId`) and the binding-slot manifest, so Sinbad can join the
//! request to the compiled case without re-deriving anything by name.
//!
//! The `/1` record types are not changed (Sinbad constructs them today); the links live beside
//! the record in [`LinkedDerivativeRequest`], the `scientia-derivative-request/2` artifact.

use crate::binding_slots::{BindingSlot, SlotKind, SlotStatus, derive_binding_slots};
use crate::derivative::{
    ActiveSet, Control, DERIVATIVE_REQUEST_SCHEMA, DerivativeConvention, DerivativeDependence,
    DerivativeLevel, DerivativeProductSpec, DerivativeRefusal, DerivativeRequest,
    DerivativeStateConvention, DesignVariable, DifferentiabilityDisposition, Objective,
    ObjectiveSense, ObservableFunctional, ScalarConvention,
};
use crate::id::{Digest, span_independent_digest};
use crate::scientific::format_expression;
use crate::semantic::{
    DeclarationId, ExprId, ProviderId, SemanticCompilation, SemanticDeclarationKind,
    SemanticExprKind, SemanticModel, SymbolId, semantic_arena_digest,
};
use crate::tensor::collect_direct_symbols;
use quantitas::{Dimension, QuantityKindId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const LINKED_DERIVATIVE_REQUEST_SCHEMA: &str = "scientia-derivative-request/2";

/// What a caller (Sinbad, from a compiled case) asks for. Slots are named by their C2.1 ids
/// (`provider/diffusivity`, `source/f`, `input/u_obs`, `boundary/walls/u`, ...).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivativeRequestSpec {
    pub model: String,
    /// An `objective` or `observable` declaration name; an observable is requested with the
    /// `Measure` sense.
    pub objective: String,
    /// Slot ids differentiated with respect to.
    pub active: Vec<String>,
    /// Slot ids named as held fixed (recorded so the active/frozen partition is explicit).
    #[serde(default)]
    pub frozen: Vec<String>,
    pub product: DerivativeProductSpec,
    pub state: DerivativeStateConvention,
    #[serde(default = "default_level")]
    pub level: DerivativeLevel,
}

fn default_level() -> DerivativeLevel {
    DerivativeLevel::Discrete
}

/// The objective's identity in the semantic arena.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObjectiveLink {
    pub name: String,
    /// The `observable/<name>` slot an objective is also evaluated through.
    pub slot: String,
    pub declaration: DeclarationId,
    pub symbol: Option<SymbolId>,
    pub expression: ExprId,
    pub sense: ObjectiveSense,
    /// The arena type of the functional's expression; `None` when the arena defers it.
    pub dimension: Option<Dimension>,
    pub quantity_kind: Option<QuantityKindId>,
    /// Every model symbol the functional reads, closed over property, constitutive, and
    /// model-defined value definitions; sorted.
    pub depends_on: Vec<SymbolId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveInputRole {
    /// A provider, valueless parameter, or `input value` slot: a `/1` `DesignVariable`.
    DesignVariable,
    /// Distributed or boundary/initial data (valueless `source`, `input field`, boundary or
    /// initial value): a `/1` `Control`.
    Control,
}

/// One named slot of the request, active or frozen, joined to the arena and the manifest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActiveInputLink {
    /// The `/1` design-variable or control name: the slot id.
    pub name: String,
    pub role: ActiveInputRole,
    pub active: bool,
    pub kind: SlotKind,
    pub symbol: Option<SymbolId>,
    pub declaration: Option<DeclarationId>,
    pub provider: Option<ProviderId>,
    pub expression: Option<ExprId>,
    pub dimension: Option<Dimension>,
    pub quantity_kind: Option<QuantityKindId>,
    /// `dim(objective) / dim(input)`: the unit of one gradient component; `None` when either
    /// side is deferred.
    pub gradient_dimension: Option<Dimension>,
}

/// `scientia-derivative-request/2`: the `/1` record plus its arena and manifest links.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkedDerivativeRequest {
    pub schema: String,
    pub model: String,
    pub parent_semantic_digest: Digest,
    pub request: DerivativeRequest,
    pub objective: ObjectiveLink,
    /// Sorted by `name`; exactly the union of the request's active and frozen sets.
    pub inputs: Vec<ActiveInputLink>,
    pub identity: Digest,
}

#[derive(Serialize)]
struct LinkedIdentity<'a> {
    schema: &'a str,
    model: &'a str,
    parent_semantic_digest: &'a Digest,
    request: &'a DerivativeRequest,
    objective: &'a ObjectiveLink,
    inputs: &'a [ActiveInputLink],
}

impl LinkedDerivativeRequest {
    pub fn validate(&self) -> Result<(), DerivativeRefusal> {
        self.request.validate()?;
        if self.schema != LINKED_DERIVATIVE_REQUEST_SCHEMA {
            return Err(refusal(
                "DERIVATIVE_SCHEMA_MISMATCH",
                format!(
                    "expected {LINKED_DERIVATIVE_REQUEST_SCHEMA}, found {}",
                    self.schema
                ),
            ));
        }
        if self.request.parent_semantic_digest != self.parent_semantic_digest.hex {
            return Err(refusal(
                "DERIVATIVE_PARENT_MISMATCH",
                "the linked request and its /1 record name different parent semantic digests",
            ));
        }
        if self
            .inputs
            .windows(2)
            .any(|pair| pair[0].name >= pair[1].name)
        {
            return Err(refusal(
                "DERIVATIVE_NONCANONICAL_INPUTS",
                "linked inputs must be uniquely sorted by slot id",
            ));
        }
        let named = self
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect::<BTreeSet<_>>();
        let partitioned = self
            .request
            .active_set
            .active
            .iter()
            .chain(&self.request.active_set.frozen)
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if named != partitioned
            || self
                .inputs
                .iter()
                .any(|input| input.active != self.request.active_set.active.contains(&input.name))
        {
            return Err(refusal(
                "DERIVATIVE_LINK_PARTITION",
                "linked inputs must be exactly the /1 active and frozen sets",
            ));
        }
        if self.identity != self.expected_identity() {
            return Err(refusal(
                "DERIVATIVE_IDENTITY_MISMATCH",
                "linked derivative request identity does not match its contents",
            ));
        }
        Ok(())
    }

    fn expected_identity(&self) -> Digest {
        span_independent_digest(&LinkedIdentity {
            schema: &self.schema,
            model: &self.model,
            parent_semantic_digest: &self.parent_semantic_digest,
            request: &self.request,
            objective: &self.objective,
            inputs: &self.inputs,
        })
    }
}

/// Derive the request `spec` asks for from `compilation` (the FC1 boundary) and the model's
/// binding-slot manifest. Refusals are typed `DERIVATIVE_*` codes; nothing is guessed: a slot
/// the model defines itself, a domain, a region, an observable, or an undeclared provider
/// cannot be an active input.
pub fn derive_derivative_request(
    compilation: &SemanticCompilation,
    spec: &DerivativeRequestSpec,
) -> Result<LinkedDerivativeRequest, DerivativeRefusal> {
    let model = compilation
        .semantic
        .models
        .iter()
        .find(|model| model.name == spec.model)
        .ok_or_else(|| {
            refusal(
                "DERIVATIVE_UNKNOWN_MODEL",
                format!("module has no model named `{}`", spec.model),
            )
        })?;
    let source_model = compilation
        .source
        .models
        .iter()
        .find(|model| model.name == spec.model);
    let manifest = derive_binding_slots(compilation)
        .into_iter()
        .find(|manifest| manifest.model == spec.model)
        .expect("every elaborated model has a slot manifest");
    let parent_semantic_digest = Digest {
        algorithm: "blake3".into(),
        hex: semantic_arena_digest(&compilation.semantic),
    };

    // ---- objective
    let declaration = model
        .declarations
        .iter()
        .find(|declaration| {
            declaration.name == spec.objective
                && matches!(
                    declaration.kind,
                    SemanticDeclarationKind::Objective { .. }
                        | SemanticDeclarationKind::Observable { .. }
                )
        })
        .ok_or_else(|| {
            refusal(
                "DERIVATIVE_UNKNOWN_OBJECTIVE",
                format!(
                    "model `{}` declares no objective or observable named `{}`",
                    spec.model, spec.objective
                ),
            )
        })?;
    let (expression, sense) = match declaration.kind {
        SemanticDeclarationKind::Objective { value, sense } => (value, sense),
        SemanticDeclarationKind::Observable { value } => (value, ObjectiveSense::Measure),
        _ => unreachable!("filtered above"),
    };
    let ty = &model.expressions[expression.index()].ty;
    if !matches!(
        ty.shape,
        crate::semantic::SemanticShape::Numeric(crate::scientific::ValueShape::Scalar)
            | crate::semantic::SemanticShape::Deferred
    ) {
        return Err(refusal(
            "DERIVATIVE_OBJECTIVE_NOT_SCALAR",
            format!(
                "objective `{}` has shape {:?}; a derivative request needs a scalar functional",
                spec.objective, ty.shape
            ),
        ));
    }
    let mut depends_on = BTreeSet::new();
    close_dependencies(model, expression, &mut depends_on).map_err(|error| {
        refusal(
            "DERIVATIVE_OBJECTIVE_INVALID",
            format!("objective `{}`: {error}", spec.objective),
        )
    })?;
    let authored = source_model.and_then(|source| {
        source
            .objectives
            .iter()
            .find(|objective| objective.name == spec.objective)
            .map(|objective| format_expression(&objective.value))
            .or_else(|| {
                source
                    .observables
                    .iter()
                    .find(|observable| observable.name == spec.objective)
                    .map(|observable| format_expression(&observable.value))
            })
    });
    let Some(objective_dimension) = resolved_dimension(ty.dimension, ty.quantity_kind.as_ref())
    else {
        return Err(refusal(
            "DERIVATIVE_OBJECTIVE_DIMENSION_UNKNOWN",
            format!(
                "objective `{}` declares a quantity kind the registry cannot resolve",
                spec.objective
            ),
        ));
    };

    let objective = ObjectiveLink {
        name: spec.objective.clone(),
        slot: format!("observable/{}", spec.objective),
        declaration: declaration.id,
        symbol: declaration.symbol,
        expression,
        sense,
        dimension: Some(objective_dimension),
        quantity_kind: ty.quantity_kind.clone(),
        depends_on: depends_on.into_iter().collect(),
    };

    // ---- inputs
    if spec.active.is_empty() {
        return Err(refusal(
            "DERIVATIVE_NO_ACTIVE_INPUT",
            "a derivative request needs at least one active slot",
        ));
    }
    let mut inputs: Vec<ActiveInputLink> = Vec::new();
    let mut design_variables = Vec::new();
    let mut controls = Vec::new();
    for (ids, active) in [(&spec.active, true), (&spec.frozen, false)] {
        for id in ids {
            if inputs.iter().any(|input| &input.name == id) {
                return Err(refusal(
                    "DERIVATIVE_DUPLICATE_INPUT",
                    format!("slot `{id}` is named more than once"),
                ));
            }
            let slot = manifest
                .slots
                .iter()
                .find(|slot| &slot.id == id)
                .ok_or_else(|| {
                    refusal(
                        "DERIVATIVE_UNKNOWN_SLOT",
                        format!("model `{}` has no binding slot `{id}`", spec.model),
                    )
                })?;
            let link = link_slot(model, slot, active, Some(objective_dimension))?;
            match link.role {
                ActiveInputRole::DesignVariable => {
                    let Some(dimension) = link.dimension else {
                        return Err(refusal(
                            "DERIVATIVE_SLOT_DIMENSION_UNKNOWN",
                            format!(
                                "slot `{id}` has no resolved dimension; declare its quantity kind"
                            ),
                        ));
                    };
                    design_variables.push(DesignVariable {
                        name: id.clone(),
                        dimension,
                        parameter_owner: format!("case slot {id}"),
                        admissible_set: admissible_set(slot),
                    });
                }
                ActiveInputRole::Control => {
                    let Some(dimension) = link.dimension else {
                        return Err(refusal(
                            "DERIVATIVE_SLOT_DIMENSION_UNKNOWN",
                            format!(
                                "slot `{id}` has no resolved dimension; declare its quantity kind"
                            ),
                        ));
                    };
                    controls.push(Control {
                        name: id.clone(),
                        dimension,
                        support: control_support(model, slot),
                    });
                }
            }
            inputs.push(link);
        }
    }
    inputs.sort_by(|left, right| left.name.cmp(&right.name));

    let (dependence, evaluation_state) = match spec.state {
        DerivativeStateConvention::FixedState => {
            (DerivativeDependence::Partial, "fixed primal state")
        }
        DerivativeStateConvention::ConvergedState => (
            DerivativeDependence::Total,
            "converged discrete primal solution",
        ),
        DerivativeStateConvention::AcceptedTrajectory => {
            (DerivativeDependence::Total, "accepted discrete trajectory")
        }
    };
    let request = DerivativeRequest {
        schema: DERIVATIVE_REQUEST_SCHEMA.into(),
        parent_semantic_digest: parent_semantic_digest.hex.clone(),
        objective: Objective {
            name: spec.objective.clone(),
            functional: ObservableFunctional {
                name: format!("{}::{}", spec.model, spec.objective),
                semantic_expression: authored.unwrap_or_else(|| format!("expr {expression}")),
                dimension: objective_dimension,
            },
            sense,
        },
        design_variables,
        controls,
        product: spec.product,
        active_set: ActiveSet {
            active: spec.active.clone(),
            frozen: spec.frozen.clone(),
        },
        evaluation_state: evaluation_state.into(),
        convention: DerivativeConvention {
            dependence,
            level: spec.level,
            scalar: ScalarConvention::Real,
            state: spec.state,
            disposition: DifferentiabilityDisposition::Smooth,
            event_or_refusal_basis: None,
        },
        shape: None,
        identity: Digest::blake3(b"unset"),
    }
    .finish()?;

    let mut linked = LinkedDerivativeRequest {
        schema: LINKED_DERIVATIVE_REQUEST_SCHEMA.into(),
        model: spec.model.clone(),
        parent_semantic_digest,
        request,
        objective,
        inputs,
        identity: Digest::blake3(b"unset"),
    };
    linked.identity = linked.expected_identity();
    linked.validate()?;
    Ok(linked)
}

fn link_slot(
    model: &SemanticModel,
    slot: &BindingSlot,
    active: bool,
    objective_dimension: Option<Dimension>,
) -> Result<ActiveInputLink, DerivativeRefusal> {
    let role = match (&slot.kind, slot.status) {
        (SlotKind::Domain { .. } | SlotKind::Region { .. } | SlotKind::Observable, _) => {
            return Err(refusal(
                "DERIVATIVE_SLOT_NOT_DIFFERENTIABLE",
                format!(
                    "slot `{}` is geometry, topology, or an observable; shape derivatives are \
                     SV1-G and observables are not inputs",
                    slot.id
                ),
            ));
        }
        (SlotKind::Provider { provider: None }, _) | (_, SlotStatus::Unbound) => {
            return Err(refusal(
                "DERIVATIVE_UNBOUND_SLOT",
                format!(
                    "slot `{}` calls an undeclared provider; declare its signature before \
                     differentiating with respect to it",
                    slot.id
                ),
            ));
        }
        (_, SlotStatus::ModelDefined) => {
            let mut through = BTreeSet::new();
            if let Some(expression) = slot.expression {
                let _ = collect_provider_slots(model, expression, &mut through);
            }
            let hint = if through.is_empty() {
                String::new()
            } else {
                format!(
                    "; its definition calls {}",
                    through
                        .iter()
                        .map(|name| format!("`provider/{name}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            };
            return Err(refusal(
                "DERIVATIVE_MODEL_DEFINED_SLOT",
                format!(
                    "slot `{}` is defined by the model, not bound by the case; name the case \
                     slot(s) it depends on{hint}",
                    slot.id
                ),
            ));
        }
        (SlotKind::Provider { .. } | SlotKind::Parameter, SlotStatus::Required) => {
            ActiveInputRole::DesignVariable
        }
        (
            SlotKind::ExternalValue
            | SlotKind::BoundaryValue { .. }
            | SlotKind::InitialValue { .. },
            SlotStatus::Required,
        ) => ActiveInputRole::Control,
    };
    let provider = match slot.kind {
        SlotKind::Provider { provider } => provider,
        _ => None,
    };
    let dimension = resolved_dimension(slot.dimension, slot.quantity_kind.as_ref());
    let gradient_dimension = match (objective_dimension, dimension) {
        (Some(objective), Some(input)) => objective.checked_quotient(input).ok(),
        _ => None,
    };
    Ok(ActiveInputLink {
        name: slot.id.clone(),
        role,
        active,
        kind: slot.kind.clone(),
        symbol: slot.symbol,
        declaration: slot.declaration,
        provider,
        expression: slot.expression,
        dimension,
        quantity_kind: slot.quantity_kind.clone(),
        gradient_dimension,
    })
}

/// A symbol that declares no quantity kind at all is the language's dimensionless scalar
/// (elaboration already types its arithmetic that way); a declared kind the registry could not
/// resolve stays deferred and is refused rather than guessed.
fn resolved_dimension(
    dimension: Option<Dimension>,
    quantity_kind: Option<&QuantityKindId>,
) -> Option<Dimension> {
    dimension.or_else(|| quantity_kind.is_none().then_some(Dimension::DIMENSIONLESS))
}

fn admissible_set(slot: &BindingSlot) -> String {
    match &slot.kind {
        SlotKind::Provider { .. } => match slot.differentiability.as_ref() {
            Some(contract) => format!("case-bound provider model; {contract:?} tangent"),
            None => "case-bound provider model".into(),
        },
        _ => "case-bound value; unconstrained".into(),
    }
}

fn control_support(model: &SemanticModel, slot: &BindingSlot) -> String {
    match &slot.kind {
        SlotKind::BoundaryValue { region, .. } => {
            format!("region {}", model.regions[region.index()].name)
        }
        SlotKind::InitialValue { .. } => "initial state".into(),
        _ => slot
            .symbol
            .and_then(|symbol| model.symbols[symbol.index()].domain)
            .map(|domain| format!("domain {}", model.domains[domain.index()].name))
            .unwrap_or_else(|| "model".into()),
    }
}

/// Close `expression`'s direct symbols over property, constitutive, and model-defined value
/// definitions, so the objective's dependency set names the fields and case-bound symbols it
/// really reads.
pub(crate) fn close_dependencies(
    model: &SemanticModel,
    expression: ExprId,
    out: &mut BTreeSet<SymbolId>,
) -> Result<(), crate::tensor::TensorCompileError> {
    let mut frontier = BTreeSet::new();
    collect_direct_symbols(&model.expressions, expression, &mut frontier)?;
    while let Some(symbol) = frontier.pop_first() {
        if !out.insert(symbol) {
            continue;
        }
        let definition = model.declarations.iter().find_map(|declaration| {
            (declaration.symbol == Some(symbol))
                .then_some(match declaration.kind {
                    SemanticDeclarationKind::Property { value }
                    | SemanticDeclarationKind::ConstitutiveLaw { value }
                    | SemanticDeclarationKind::Value { value: Some(value) } => Some(value),
                    _ => None,
                })
                .flatten()
        });
        if let Some(definition) = definition {
            let mut next = BTreeSet::new();
            collect_direct_symbols(&model.expressions, definition, &mut next)?;
            frontier.extend(next.into_iter().filter(|next| !out.contains(next)));
        }
    }
    Ok(())
}

/// Provider names called (directly or through definitions) by `expression`.
pub(crate) fn collect_provider_slots(
    model: &SemanticModel,
    expression: ExprId,
    out: &mut BTreeSet<String>,
) -> Result<(), crate::tensor::TensorCompileError> {
    let mut symbols = BTreeSet::new();
    close_dependencies(model, expression, &mut symbols)?;
    let mut stack = vec![expression];
    for symbol in &symbols {
        stack.extend(model.declarations.iter().filter_map(|declaration| {
            (declaration.symbol == Some(*symbol))
                .then_some(match declaration.kind {
                    SemanticDeclarationKind::Property { value }
                    | SemanticDeclarationKind::ConstitutiveLaw { value }
                    | SemanticDeclarationKind::Value { value: Some(value) } => Some(value),
                    _ => None,
                })
                .flatten()
        }));
    }
    let mut seen = BTreeSet::new();
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let Some(expr) = model.expressions.get(id.index()) else {
            continue;
        };
        match &expr.kind {
            SemanticExprKind::ProviderCall { provider, args, .. } => {
                out.insert(model.providers[provider.index()].name.clone());
                stack.extend(args.iter().copied());
            }
            SemanticExprKind::Call { function, args } => {
                if !crate::semantic::is_intrinsic_call(function)
                    && !model.providers.iter().any(|p| &p.name == function)
                {
                    out.insert(function.clone());
                }
                stack.extend(args.iter().copied());
            }
            SemanticExprKind::Unary { arg, .. }
            | SemanticExprKind::Differential { arg, .. }
            | SemanticExprKind::TensorTrace { value: arg, .. }
            | SemanticExprKind::FacetTrace { value: arg, .. }
            | SemanticExprKind::Jump { value: arg }
            | SemanticExprKind::Average { value: arg }
            | SemanticExprKind::Conjugate { value: arg }
            | SemanticExprKind::NormalComponent { value: arg, .. } => stack.push(*arg),
            SemanticExprKind::Binary { lhs, rhs, .. }
            | SemanticExprKind::Contraction { lhs, rhs, .. } => {
                stack.push(*lhs);
                stack.push(*rhs);
            }
            SemanticExprKind::Vector { elements } => stack.extend(elements.iter().copied()),
            SemanticExprKind::Index { value, indices } => {
                stack.push(*value);
                stack.extend(indices.iter().copied());
            }
            SemanticExprKind::Number { .. }
            | SemanticExprKind::String { .. }
            | SemanticExprKind::Symbol { .. } => {}
        }
    }
    Ok(())
}

fn refusal(code: &str, message: impl Into<String>) -> DerivativeRefusal {
    DerivativeRefusal {
        code: code.into(),
        message: message.into(),
    }
}
