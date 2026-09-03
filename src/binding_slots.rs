//! `scientia-binding-slots/1`: a mesh-free, form-free inventory of every named datum a case
//! (Sinbad's `sinbad-case/1`) must eventually bind to run a model (GX-A1, contract C2).
//!
//! `derive_binding_slots` reads only the elaborated [`SemanticModel`] arena -- domains,
//! regions, providers, and declarations -- so it is derivable for every model that elaborates,
//! independent of whether any of its equations derive a form (FC2+).

use crate::id::{Digest, span_independent_digest};
use crate::scientific::{
    BoundaryConditionKind, CoordinateSystem, DerivativeContract, PropertyLocality, ValueShape,
};
use crate::semantic::{
    DeclarationId, DomainId, ExprId, ProviderId, RegionId, RegionKind, SemanticCompilation,
    SemanticDeclarationKind, SemanticExprKind, SemanticModel, SemanticRole, SemanticShape,
    SymbolId, is_intrinsic_call, semantic_arena_digest,
};
use crate::source::SourceSpan;
use quantitas::{Dimension, QuantityKindId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const BINDING_SLOTS_SCHEMA: &str = "scientia-binding-slots/1";

/// One model's complete binding-slot inventory.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BindingSlotManifest {
    pub schema: String,
    pub model: String,
    pub parent_semantic_digest: Digest,
    /// Sorted by [`BindingSlot::id`].
    pub slots: Vec<BindingSlot>,
    /// Span-independent digest of `slots`; stable across declaration reordering and
    /// presentation changes, like every other FC-boundary artifact digest.
    pub identity: Digest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BindingSlot {
    /// Canonical slot path (contract C2.1), e.g. `provider/thermal_conductivity` or
    /// `boundary/walls/T`. Unique per model and stable across declaration reordering.
    pub id: String,
    pub kind: SlotKind,
    pub symbol: Option<SymbolId>,
    pub declaration: Option<DeclarationId>,
    /// The defining expression, when the model authors one directly for this slot: a
    /// property's value, an observable/invariant's value, or a boundary/initial condition's
    /// value. Populated whenever such an expression exists, regardless of `status` -- a
    /// `Required` boundary slot still names the authored value expression (e.g. a call to an
    /// unbound provider) that a case must eventually discharge.
    pub expression: Option<ExprId>,
    pub quantity_kind: Option<QuantityKindId>,
    pub dimension: Option<Dimension>,
    pub shape: Option<ValueShape>,
    /// Provider inputs (name, quantity kind, dimension, shape); empty for every non-`Provider`
    /// slot kind.
    pub inputs: Vec<SlotInput>,
    pub differentiability: Option<DerivativeContract>,
    pub locality: Option<PropertyLocality>,
    pub status: SlotStatus,
    pub span: SourceSpan,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SlotInput {
    pub name: String,
    /// `None` is the `selector` pseudo-kind.
    pub quantity_kind: Option<QuantityKindId>,
    pub dimension: Option<Dimension>,
    pub shape: Option<ValueShape>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SlotKind {
    /// `provider: None` when the call site names no declared provider (`Unbound`).
    Provider {
        provider: Option<ProviderId>,
    },
    /// `source Q: Kind;` / a valueless parameter.
    ExternalValue,
    /// A declared `parameter`, or a model-computed `property` (status distinguishes the two;
    /// the frozen C2 `SlotKind` enum has no dedicated property variant).
    Parameter,
    BoundaryValue {
        region: RegionId,
        target: SymbolId,
        condition: BoundaryConditionKind,
    },
    InitialValue {
        target: SymbolId,
    },
    Region {
        region: RegionId,
        region_kind: RegionKind,
    },
    Domain {
        domain: DomainId,
        dimension: u8,
        coordinates: CoordinateSystem,
    },
    /// An `observable` or an `invariant` (folded into one id namespace; C2.1 lists only
    /// `observable/<name>`).
    Observable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotStatus {
    /// Data is required from the case (declared provider, external value, region, domain,
    /// parameter, boundary/initial value).
    Required,
    /// Provider call with no declared signature; refused at compile until declared.
    Unbound,
    /// Fully defined by the model (property/observable/invariant = expression over fields or
    /// other slots); no data needed, but still listed so consumers can inspect and
    /// override-refuse.
    ModelDefined,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum BindingSlotError {
    #[error("SLOT_DUPLICATE_ID: duplicate slot id `{0}`")]
    DuplicateId(String),
}

/// Derive one [`BindingSlotManifest`] per model in `compilation`. Slot derivation reads only
/// domains, regions, providers, and declarations -- never a derived [`crate::VariationalForm`]
/// or [`crate::FormRequirements`] -- so it succeeds for every model that elaborates.
pub fn derive_binding_slots(compilation: &SemanticCompilation) -> Vec<BindingSlotManifest> {
    let parent_semantic_digest = Digest {
        algorithm: "blake3".into(),
        hex: semantic_arena_digest(&compilation.semantic),
    };
    compilation
        .semantic
        .models
        .iter()
        .map(|model| derive_model_slots(model, parent_semantic_digest.clone()))
        .collect()
}

/// Check the C2 invariant that every slot id in a manifest is unique. `derive_binding_slots`
/// already de-duplicates by construction (first declaration wins, in declaration order), so
/// this should never fail for a manifest it produced; it exists for consumers (or a manifest
/// reconstructed some other way) to verify the invariant explicitly.
pub fn validate_binding_slot_manifest(
    manifest: &BindingSlotManifest,
) -> Result<(), BindingSlotError> {
    let mut seen = BTreeSet::new();
    for slot in &manifest.slots {
        if !seen.insert(slot.id.as_str()) {
            return Err(BindingSlotError::DuplicateId(slot.id.clone()));
        }
    }
    Ok(())
}

fn derive_model_slots(
    model: &SemanticModel,
    parent_semantic_digest: Digest,
) -> BindingSlotManifest {
    let mut slots: BTreeMap<String, BindingSlot> = BTreeMap::new();
    let insert = |slots: &mut BTreeMap<String, BindingSlot>, slot: BindingSlot| {
        // First declaration (in source/elaboration order) wins; this is what keeps `id`
        // unique by construction even if a model declares two boundary conditions for the
        // same (region, target field) pair.
        slots.entry(slot.id.clone()).or_insert(slot);
    };

    for domain in &model.domains {
        insert(
            &mut slots,
            BindingSlot {
                id: format!("domain/{}", domain.name),
                kind: SlotKind::Domain {
                    domain: domain.id,
                    dimension: domain.spatial_dimension,
                    coordinates: domain.coordinates.clone(),
                },
                symbol: None,
                declaration: None,
                expression: None,
                quantity_kind: None,
                dimension: None,
                shape: None,
                inputs: vec![],
                differentiability: None,
                locality: None,
                status: SlotStatus::Required,
                span: domain.span,
            },
        );
    }

    for region in &model.regions {
        insert(
            &mut slots,
            BindingSlot {
                id: format!("region/{}", region.name),
                kind: SlotKind::Region {
                    region: region.id,
                    region_kind: region.kind.clone(),
                },
                symbol: None,
                declaration: None,
                expression: None,
                quantity_kind: None,
                dimension: None,
                shape: None,
                inputs: vec![],
                differentiability: None,
                locality: None,
                status: SlotStatus::Required,
                span: region.span,
            },
        );
    }

    for provider in &model.providers {
        let inputs = provider
            .inputs
            .iter()
            .map(|input| SlotInput {
                name: input.name.clone(),
                quantity_kind: input.quantity_kind.clone(),
                dimension: input.dimension,
                shape: Some(input.shape.clone()),
            })
            .collect();
        insert(
            &mut slots,
            BindingSlot {
                id: format!("provider/{}", provider.name),
                kind: SlotKind::Provider {
                    provider: Some(provider.id),
                },
                symbol: None,
                declaration: None,
                expression: None,
                quantity_kind: Some(provider.output.quantity_kind.clone()),
                dimension: provider.output.dimension,
                shape: Some(provider.output.shape.clone()),
                inputs,
                differentiability: Some(provider.differentiability.clone()),
                locality: Some(provider.locality.clone()),
                status: SlotStatus::Required,
                span: provider.span,
            },
        );
    }

    // Every call to a non-intrinsic, non-declared function is an unbound provider slot
    // (contract C1.2's `RESOLVE_UNDECLARED_PROVIDER` advisory), independent of whether a
    // typed form derives from any equation that references it.
    let declared_provider_names = model
        .providers
        .iter()
        .map(|provider| provider.name.as_str())
        .collect::<BTreeSet<_>>();
    for expression in model.expressions.iter() {
        if let SemanticExprKind::Call { function, .. } = &expression.kind
            && !is_intrinsic_call(function)
            && !declared_provider_names.contains(function.as_str())
        {
            insert(
                &mut slots,
                BindingSlot {
                    id: format!("provider/{function}"),
                    kind: SlotKind::Provider { provider: None },
                    symbol: None,
                    declaration: None,
                    expression: None,
                    quantity_kind: None,
                    dimension: None,
                    shape: None,
                    inputs: vec![],
                    differentiability: None,
                    locality: None,
                    status: SlotStatus::Unbound,
                    span: expression.span,
                },
            );
        }
    }

    for declaration in &model.declarations {
        match (&declaration.role, &declaration.kind) {
            (SemanticRole::Property, SemanticDeclarationKind::Property { value }) => {
                let ty = &model.expressions[value.index()].ty;
                insert(
                    &mut slots,
                    BindingSlot {
                        id: format!("property/{}", declaration.name),
                        kind: SlotKind::Parameter,
                        symbol: declaration.symbol,
                        declaration: Some(declaration.id),
                        expression: Some(*value),
                        quantity_kind: ty.quantity_kind.clone(),
                        dimension: ty.dimension,
                        shape: numeric_shape(&ty.shape),
                        inputs: vec![],
                        differentiability: None,
                        locality: None,
                        status: SlotStatus::ModelDefined,
                        span: declaration.span,
                    },
                );
            }
            (SemanticRole::Parameter, SemanticDeclarationKind::Value { .. }) => {
                let ty = declaration
                    .symbol
                    .map(|symbol| &model.symbols[symbol.index()].ty);
                insert(
                    &mut slots,
                    BindingSlot {
                        id: format!("parameter/{}", declaration.name),
                        kind: SlotKind::Parameter,
                        symbol: declaration.symbol,
                        declaration: Some(declaration.id),
                        expression: None,
                        quantity_kind: ty.and_then(|ty| ty.quantity_kind.clone()),
                        dimension: ty.and_then(|ty| ty.dimension),
                        shape: ty.and_then(|ty| numeric_shape(&ty.shape)),
                        inputs: vec![],
                        differentiability: None,
                        locality: None,
                        status: SlotStatus::Required,
                        span: declaration.span,
                    },
                );
            }
            (SemanticRole::Source, SemanticDeclarationKind::Value { value }) => {
                let ty = declaration
                    .symbol
                    .map(|symbol| &model.symbols[symbol.index()].ty);
                // A defined `source x = expr;` is `ModelDefined`, never `Required`: the model
                // authors the datum (08's Joule term), so a case must not be forced to shadow
                // it (`sinbad/ARCHITECTURE.md` §3.3). Only a valueless `source x: Kind;` is
                // external data.
                let status = if value.is_some() {
                    SlotStatus::ModelDefined
                } else {
                    SlotStatus::Required
                };
                insert(
                    &mut slots,
                    BindingSlot {
                        id: format!("source/{}", declaration.name),
                        kind: SlotKind::ExternalValue,
                        symbol: declaration.symbol,
                        declaration: Some(declaration.id),
                        expression: *value,
                        quantity_kind: ty.and_then(|ty| ty.quantity_kind.clone()),
                        dimension: ty.and_then(|ty| ty.dimension),
                        shape: ty.and_then(|ty| numeric_shape(&ty.shape)),
                        inputs: vec![],
                        differentiability: None,
                        locality: None,
                        status,
                        span: declaration.span,
                    },
                );
            }
            // `input field x: Kind on D;` / `input value x: Kind;` (§3.3): binding slots by
            // construction, always `Required`. Their slot kinds reuse the frozen C2 enum -- an
            // input field is externally supplied field data exactly like a valueless `source`
            // (`ExternalValue`), an input value is a valueless `parameter` -- and the `input/`
            // id prefix records the declaration form.
            (
                SemanticRole::Source | SemanticRole::Parameter,
                SemanticDeclarationKind::InputField { .. } | SemanticDeclarationKind::InputValue,
            ) => {
                let ty = declaration
                    .symbol
                    .map(|symbol| &model.symbols[symbol.index()].ty);
                let kind = if matches!(declaration.kind, SemanticDeclarationKind::InputField { .. })
                {
                    SlotKind::ExternalValue
                } else {
                    SlotKind::Parameter
                };
                insert(
                    &mut slots,
                    BindingSlot {
                        id: format!("input/{}", declaration.name),
                        kind,
                        symbol: declaration.symbol,
                        declaration: Some(declaration.id),
                        expression: None,
                        quantity_kind: ty.and_then(|ty| ty.quantity_kind.clone()),
                        dimension: ty.and_then(|ty| ty.dimension),
                        shape: ty.and_then(|ty| numeric_shape(&ty.shape)),
                        inputs: vec![],
                        differentiability: None,
                        locality: None,
                        status: SlotStatus::Required,
                        span: declaration.span,
                    },
                );
            }
            (
                SemanticRole::Observable | SemanticRole::Invariant,
                SemanticDeclarationKind::Observable { value }
                | SemanticDeclarationKind::Invariant { value },
            ) => {
                let ty = &model.expressions[value.index()].ty;
                insert(
                    &mut slots,
                    BindingSlot {
                        id: format!("observable/{}", declaration.name),
                        kind: SlotKind::Observable,
                        symbol: declaration.symbol,
                        declaration: Some(declaration.id),
                        expression: Some(*value),
                        quantity_kind: ty.quantity_kind.clone(),
                        dimension: ty.dimension,
                        shape: numeric_shape(&ty.shape),
                        inputs: vec![],
                        differentiability: None,
                        locality: None,
                        status: SlotStatus::ModelDefined,
                        span: declaration.span,
                    },
                );
            }
            (
                SemanticRole::InitialCondition,
                SemanticDeclarationKind::InitialCondition {
                    target: Some(target),
                    value,
                },
            ) => {
                let target_ty = &model.symbols[target.index()].ty;
                insert(
                    &mut slots,
                    BindingSlot {
                        id: format!("initial/{}", model.symbols[target.index()].name),
                        kind: SlotKind::InitialValue { target: *target },
                        symbol: Some(*target),
                        declaration: Some(declaration.id),
                        expression: Some(*value),
                        quantity_kind: target_ty.quantity_kind.clone(),
                        dimension: target_ty.dimension,
                        shape: numeric_shape(&target_ty.shape),
                        inputs: vec![],
                        differentiability: None,
                        locality: None,
                        status: SlotStatus::Required,
                        span: declaration.span,
                    },
                );
            }
            (
                SemanticRole::BoundaryCondition | SemanticRole::InterfaceCondition,
                SemanticDeclarationKind::BoundaryCondition {
                    region,
                    target: Some(target),
                    condition,
                    value,
                    ..
                },
            ) => {
                let target_ty = &model.symbols[target.index()].ty;
                let region_name = &model.regions[region.index()].name;
                insert(
                    &mut slots,
                    BindingSlot {
                        id: format!(
                            "boundary/{region_name}/{}",
                            model.symbols[target.index()].name
                        ),
                        kind: SlotKind::BoundaryValue {
                            region: *region,
                            target: *target,
                            condition: condition.clone(),
                        },
                        symbol: Some(*target),
                        declaration: Some(declaration.id),
                        expression: Some(*value),
                        quantity_kind: target_ty.quantity_kind.clone(),
                        dimension: target_ty.dimension,
                        shape: numeric_shape(&target_ty.shape),
                        inputs: vec![],
                        differentiability: None,
                        locality: None,
                        status: SlotStatus::Required,
                        span: declaration.span,
                    },
                );
            }
            _ => {}
        }
    }

    let slots = slots.into_values().collect::<Vec<_>>();
    let identity = span_independent_digest(&slots);
    BindingSlotManifest {
        schema: BINDING_SLOTS_SCHEMA.into(),
        model: model.name.clone(),
        parent_semantic_digest,
        slots,
        identity,
    }
}

fn numeric_shape(shape: &SemanticShape) -> Option<ValueShape> {
    match shape {
        SemanticShape::Numeric(value) => Some(value.clone()),
        _ => None,
    }
}
