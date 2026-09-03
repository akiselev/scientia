//! SC-W1: the `ScientificSystem` (`scientia-system/1`, `sinbad/ARCHITECTURE.md` §1–§3, §5):
//! instances of models, system domains, `bind` chains, the dense system-level id arena with
//! its origin map, and the instance-prefixed binding-slot manifest
//! (`scientia-binding-slots/2`). A bare `model` compiles as the implicit one-instance system
//! (§1.2): instance 0, empty slot prefix, identity domain and region maps, so every artifact
//! and slot id of a directly compiled model is unchanged.
//!
//! Per-model arenas and artifacts are never rewritten (§2.2): a system references each
//! instance's model by [`GlobalDeclId`] and semantic digest and keeps only maps and links.

use crate::binding_slots::{BindingSlot, SlotKind, SlotStatus, derive_binding_slots};
use crate::id::{Digest, span_independent_digest};
use crate::scientific::{
    CoordinateSystem, DeclKind, FieldRole, GlobalDeclId, ModuleClosure, ModuleDigest, ValueShape,
};
use crate::semantic::{
    DeclarationId, DomainId, ExprId, RegionId, RegionKind, Registries, SemanticCompilation,
    SemanticDeclarationKind, SemanticExprKind, SemanticModel, SemanticRole, SymbolId,
    compile_module_in_closure, semantic_arena_digest,
};
use crate::source::{SourceLocator, SourceSpan};
use crate::structural::scc::{Digraph, tarjan_scc};
use quantitas::{Dimension, QuantityKindId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use thiserror::Error;

pub const SYSTEM_SCHEMA: &str = "scientia-system/1";
pub const SYSTEM_SLOTS_SCHEMA: &str = "scientia-binding-slots/2";

macro_rules! sys_id {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(
            Clone,
            Copy,
            Debug,
            Default,
            PartialEq,
            Eq,
            PartialOrd,
            Ord,
            Hash,
            Serialize,
            Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub u32);

        impl $name {
            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

sys_id!(
    InstanceId,
    "Dense instance index in declaration order; the implicit root is 0."
);
sys_id!(
    SysVarId,
    "Dense system-level variable id (§2.3); Krasis `SemanticId` mirrors it."
);
sys_id!(SysResId, "Dense system-level residual id (§2.3).");
sys_id!(SysDomainId, "Dense system-level domain id.");
sys_id!(
    SysRegionId,
    "Dense system-level region id; Finitum keys region maps by it."
);
sys_id!(
    OutputId,
    "Dense id of one instance output across the system."
);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemDomain {
    pub id: SysDomainId,
    pub name: String,
    pub dimension: u8,
    pub coordinates: CoordinateSystem,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemRegion {
    pub id: SysRegionId,
    /// Display path: `<instance>.<region>`, or the bare region name on the implicit root.
    pub name: String,
    pub kind: RegionKind,
    /// `None` for a region the model declares without a domain.
    pub domain: Option<SysDomainId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceRecord {
    pub instance: InstanceId,
    /// Instance name; empty on the implicit root.
    pub name: String,
    /// Slot-id prefix (`<name>/`); empty on the implicit root so all existing ids stay valid.
    pub prefix: String,
    pub model: GlobalDeclId,
    pub model_name: String,
    /// `semantic_arena_digest` of the module the model was elaborated in.
    pub semantic_digest: Digest,
    pub domain_map: Vec<(DomainId, SysDomainId)>,
    pub region_map: Vec<(RegionId, SysRegionId)>,
    pub locator: SourceLocator,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SysVarKind {
    Field { role: FieldRole },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SysVar {
    pub id: SysVarId,
    pub owner: InstanceId,
    pub local: SymbolId,
    /// Display path `<instance>.<field>` (bare field name on the implicit root).
    pub name: String,
    pub kind: SysVarKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrientationBasis {
    /// The residual carries a `dt(state)` term; the row is oriented so accumulation is `+1`.
    Accumulation,
    /// No accumulation term and no `oriented by` (SC-W2); the authored sign is kept.
    Unoriented,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResidualOrigin {
    Equation {
        instance: InstanceId,
        declaration: DeclarationId,
        name: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SysRes {
    pub id: SysResId,
    pub origin: ResidualOrigin,
    /// §3.3 orientation normalization, before any gauge: `+1` keeps the authored residual,
    /// `-1` negates it.
    pub orientation: i8,
    pub orientation_basis: OrientationBasis,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SystemOutput {
    pub id: OutputId,
    pub instance: InstanceId,
    pub name: String,
    pub declaration: DeclarationId,
    pub expression: ExprId,
    pub domain: SysDomainId,
    pub dimension: Option<Dimension>,
    pub quantity_kind: Option<QuantityKindId>,
    pub shape: Option<ValueShape>,
    /// The producer's physical fields the output depends on (closed over property,
    /// constitutive, and value definitions); sorted.
    pub depends_on_fields: Vec<SymbolId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SlotBinding {
    /// Closed by case data (or still open); `status` on the local slot says which is required.
    Open,
    /// Closed by `bind`; index into `ScientificSystem::binds`.
    Bound { bind: usize },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SystemSlot {
    /// `<prefix><local id>`; the implicit root has no prefix.
    pub id: String,
    pub instance: InstanceId,
    pub local_id: String,
    pub slot: BindingSlot,
    pub binding: SlotBinding,
}

/// `scientia-binding-slots/2`: the per-instance manifests under their prefixes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SystemSlotManifest {
    pub schema: String,
    pub slots: Vec<SystemSlot>,
    pub identity: Digest,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BoundChain {
    /// Same system domain: the consumer's kernel takes the input as an external field and the
    /// producer's output kernel is evaluated at the same quadrature point (§6).
    Composed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemBind {
    pub consumer: InstanceId,
    /// The prefixed slot id closed by this bind.
    pub consumer_slot: String,
    pub consumer_symbol: SymbolId,
    pub producer: OutputId,
    pub chain: BoundChain,
    pub locator: SourceLocator,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencyEdge {
    pub producer: InstanceId,
    pub consumer: InstanceId,
    pub bind: usize,
}

/// Coupling structure over instances (§5): one edge per bind from producer to consumer, the
/// strongly connected components in a topological order of the condensation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemDependency {
    pub edges: Vec<DependencyEdge>,
    pub components: Vec<Vec<InstanceId>>,
    /// Every component is a single instance: the system is a DAG and admits a sequential
    /// schedule in `components` order.
    pub sequential: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariableOrigin {
    pub variable: SysVarId,
    pub instance: InstanceId,
    pub model: GlobalDeclId,
    pub symbol: SymbolId,
    pub locator: SourceLocator,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OriginMap {
    pub variables: Vec<VariableOrigin>,
    pub residuals: Vec<(SysResId, ResidualOrigin, SourceLocator)>,
    pub regions: Vec<(SysRegionId, InstanceId, RegionId)>,
    pub domains: Vec<(SysDomainId, Vec<(InstanceId, DomainId)>)>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScientificSystem {
    pub schema: String,
    pub root: GlobalDeclId,
    pub name: String,
    /// The implicit one-instance system of a bare `model`.
    pub implicit: bool,
    pub closure_identity: String,
    pub domains: Vec<SystemDomain>,
    pub regions: Vec<SystemRegion>,
    pub instances: Vec<InstanceRecord>,
    pub variables: Vec<SysVar>,
    pub residuals: Vec<SysRes>,
    pub outputs: Vec<SystemOutput>,
    pub slots: SystemSlotManifest,
    pub binds: Vec<SystemBind>,
    pub dependency: SystemDependency,
    pub origin_map: OriginMap,
    pub identity: Digest,
}

/// A system plus the elaborated modules its instances come from, keyed by module digest.
#[derive(Clone, Debug, PartialEq)]
pub struct SystemCompilation {
    pub system: ScientificSystem,
    pub modules: BTreeMap<ModuleDigest, SemanticCompilation>,
}

impl SystemCompilation {
    /// The elaborated model behind an instance.
    pub fn model(&self, instance: InstanceId) -> &SemanticModel {
        let record = &self.system.instances[instance.index()];
        self.modules[&record.model.module]
            .semantic
            .models
            .iter()
            .find(|model| model.name == record.model_name)
            .expect("instance records name an elaborated model")
    }

    pub fn compilation(&self, instance: InstanceId) -> &SemanticCompilation {
        let record = &self.system.instances[instance.index()];
        &self.modules[&record.model.module]
    }
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum SystemError {
    #[error("SYSTEM_UNKNOWN_SYSTEM: module `{module}` declares no system `{system}`")]
    UnknownSystem { module: String, system: String },
    #[error(
        "SYSTEM_UNKNOWN_MODEL: `{model}` names no local or imported model (instance `{instance}`)"
    )]
    UnknownModel { instance: String, model: String },
    #[error("SYSTEM_DUPLICATE_INSTANCE: instance `{0}` is declared twice")]
    DuplicateInstance(String),
    #[error("SYSTEM_ELABORATION: module `{module}` failed to elaborate: {detail}")]
    Elaboration { module: String, detail: String },
    #[error(
        "SYSTEM_UNKNOWN_PARAMETER: model `{model}` has no domain parameter `{parameter}` (instance `{instance}`)"
    )]
    UnknownParameter {
        instance: String,
        model: String,
        parameter: String,
    },
    #[error("SYSTEM_UNKNOWN_DOMAIN: system declares no domain `{domain}` (instance `{instance}`)")]
    UnknownDomain { instance: String, domain: String },
    #[error("SYSTEM_DOMAIN_UNMAPPED: instance `{instance}` does not map model domain `{domain}`")]
    DomainUnmapped { instance: String, domain: String },
    #[error(
        "SYSTEM_DOMAIN_MISMATCH: instance `{instance}` maps `{parameter}` (dimension {expected}) to `{domain}` (dimension {actual})"
    )]
    DomainMismatch {
        instance: String,
        parameter: String,
        expected: u8,
        domain: String,
        actual: u8,
    },
    #[error("SYSTEM_UNKNOWN_INSTANCE: no instance named `{0}`")]
    UnknownInstance(String),
    #[error("SYSTEM_UNKNOWN_INPUT: instance `{instance}` has no input `{member}`")]
    UnknownInput { instance: String, member: String },
    #[error("SYSTEM_UNKNOWN_OUTPUT: instance `{instance}` has no output `{member}`")]
    UnknownOutput { instance: String, member: String },
    #[error(
        "SYSTEM_PRIVATE_SYMBOL: `{instance}.{member}` is not an input or output of the instance's model"
    )]
    PrivateSymbol { instance: String, member: String },
    #[error("SYSTEM_DUPLICATE_BINDING: slot `{0}` has more than one producer")]
    DuplicateBinding(String),
    #[error("SYSTEM_BIND_KIND_MISMATCH: `{consumer}` <- `{producer}`: {detail}")]
    BindKindMismatch {
        consumer: String,
        producer: String,
        detail: String,
    },
    #[error(
        "SYSTEM_BIND_SUPPORT_MISMATCH: `{consumer}` is a spatially constant input but `{producer}` is a field output"
    )]
    BindSupportMismatch { consumer: String, producer: String },
    #[error(
        "SYSTEM_BIND_CROSS_DOMAIN: `{consumer}` lives on `{consumer_domain}` but `{producer}` on `{producer_domain}`; a cross-domain bind needs a declared relation (SC-W2)"
    )]
    BindCrossDomain {
        consumer: String,
        consumer_domain: String,
        producer: String,
        producer_domain: String,
    },
    #[error("SYSTEM_OPEN_INPUT: required slot `{0}` is closed neither by a bind nor by case data")]
    OpenInput(String),
}

/// Compile the system `system_name` declared in the closure's root module.
pub fn compile_system(
    closure: &ModuleClosure,
    registries: Registries<'_>,
    system_name: &str,
) -> Result<SystemCompilation, SystemError> {
    let root = closure.root_module();
    let declaration = root
        .module
        .systems
        .iter()
        .find(|system| system.name == system_name)
        .ok_or_else(|| SystemError::UnknownSystem {
            module: root.name.clone(),
            system: system_name.to_owned(),
        })?;
    let root_compilation =
        compile_module_in_closure(closure, &root.name, registries).map_err(|diagnostics| {
            SystemError::Elaboration {
                module: root.name.clone(),
                detail: format!("{diagnostics:?}"),
            }
        })?;
    let mut modules = BTreeMap::new();
    modules.insert(root.digest.clone(), root_compilation);
    let mut builder = Builder {
        closure,
        registries,
        modules,
        root: GlobalDeclId {
            module: root.digest.clone(),
            kind: DeclKind::System,
            name: system_name.to_owned(),
        },
        name: system_name.to_owned(),
        implicit: false,
        domains: vec![],
        regions: vec![],
        instances: vec![],
        variables: vec![],
        residuals: vec![],
        outputs: vec![],
        slots: vec![],
        binds: vec![],
        origin_regions: vec![],
        origin_domains: vec![],
    };
    for domain in &declaration.domains {
        let id = SysDomainId(builder.domains.len() as u32);
        builder.domains.push(SystemDomain {
            id,
            name: domain.name.clone(),
            dimension: domain.dimension,
            coordinates: domain.coordinates.clone(),
        });
        builder.origin_domains.push((id, vec![]));
    }
    let mut seen = BTreeSet::new();
    for instance in &declaration.instances {
        if !seen.insert(instance.name.clone()) {
            return Err(SystemError::DuplicateInstance(instance.name.clone()));
        }
        let model = builder.resolve_model(&root.digest, &instance.model, &instance.name)?;
        let arguments = instance
            .arguments
            .iter()
            .map(|argument| (argument.parameter.clone(), argument.value.clone()))
            .collect::<Vec<_>>();
        builder.add_instance(
            &instance.name,
            model,
            &arguments,
            SourceLocator {
                module: root.digest.clone(),
                span: instance.span,
            },
        )?;
    }
    for bind in &declaration.binds {
        builder.add_bind(
            &bind.consumer.instance,
            &bind.consumer.member,
            &bind.producer.instance,
            &bind.producer.member,
            SourceLocator {
                module: root.digest.clone(),
                span: bind.span,
            },
        )?;
    }
    builder.finish(declaration.span)
}

/// Compile the implicit one-instance system of `model_name`, declared in the closure's root
/// module (§1.2): instance 0, no slot prefix, identity domain and region maps.
pub fn compile_model_system(
    closure: &ModuleClosure,
    registries: Registries<'_>,
    model_name: &str,
) -> Result<SystemCompilation, SystemError> {
    let root = closure.root_module();
    let root_compilation =
        compile_module_in_closure(closure, &root.name, registries).map_err(|diagnostics| {
            SystemError::Elaboration {
                module: root.name.clone(),
                detail: format!("{diagnostics:?}"),
            }
        })?;
    let model = closure
        .declaration(&root.name, DeclKind::Model, model_name)
        .ok_or_else(|| SystemError::UnknownModel {
            instance: String::new(),
            model: model_name.to_owned(),
        })?;
    let semantic_model = root_compilation
        .semantic
        .models
        .iter()
        .find(|model| model.name == model_name)
        .ok_or_else(|| SystemError::UnknownModel {
            instance: String::new(),
            model: model_name.to_owned(),
        })?;
    let span = semantic_model.span;
    let domains = semantic_model
        .domains
        .iter()
        .map(|domain| SystemDomain {
            id: SysDomainId(domain.id.0),
            name: domain.name.clone(),
            dimension: domain.spatial_dimension,
            coordinates: domain.coordinates.clone(),
        })
        .collect::<Vec<_>>();
    let arguments = semantic_model
        .domains
        .iter()
        .map(|domain| (domain.name.clone(), domain.name.clone()))
        .collect::<Vec<_>>();
    let mut modules = BTreeMap::new();
    modules.insert(root.digest.clone(), root_compilation);
    let mut builder = Builder {
        closure,
        registries,
        modules,
        root: model.clone(),
        name: model_name.to_owned(),
        implicit: true,
        origin_domains: domains.iter().map(|domain| (domain.id, vec![])).collect(),
        domains,
        regions: vec![],
        instances: vec![],
        variables: vec![],
        residuals: vec![],
        outputs: vec![],
        slots: vec![],
        binds: vec![],
        origin_regions: vec![],
    };
    builder.add_instance(
        "",
        model,
        &arguments,
        SourceLocator {
            module: root.digest.clone(),
            span,
        },
    )?;
    builder.finish(span)
}

struct Builder<'a> {
    closure: &'a ModuleClosure,
    registries: Registries<'a>,
    modules: BTreeMap<ModuleDigest, SemanticCompilation>,
    root: GlobalDeclId,
    name: String,
    implicit: bool,
    domains: Vec<SystemDomain>,
    regions: Vec<SystemRegion>,
    instances: Vec<InstanceRecord>,
    variables: Vec<SysVar>,
    residuals: Vec<SysRes>,
    outputs: Vec<SystemOutput>,
    slots: Vec<SystemSlot>,
    binds: Vec<SystemBind>,
    origin_regions: Vec<(SysRegionId, InstanceId, RegionId)>,
    origin_domains: Vec<(SysDomainId, Vec<(InstanceId, DomainId)>)>,
}

impl Builder<'_> {
    /// A model reference in a system: a model of the root module, or an import binding
    /// (selective name or `alias.Model`) of kind `Model`.
    fn resolve_model(
        &mut self,
        root: &ModuleDigest,
        model: &str,
        instance: &str,
    ) -> Result<GlobalDeclId, SystemError> {
        let root_compilation = &self.modules[root];
        if let Some(id) = self
            .closure
            .declaration(&self.closure.root, DeclKind::Model, model)
        {
            return Ok(id);
        }
        let imported = root_compilation
            .semantic
            .imports
            .iter()
            .find(|import| import.name == model && import.target.kind == DeclKind::Model)
            .map(|import| import.target.clone())
            .ok_or_else(|| SystemError::UnknownModel {
                instance: instance.to_owned(),
                model: model.to_owned(),
            })?;
        self.ensure_module(&imported.module)?;
        Ok(imported)
    }

    fn ensure_module(&mut self, digest: &ModuleDigest) -> Result<(), SystemError> {
        if self.modules.contains_key(digest) {
            return Ok(());
        }
        let entry = self
            .closure
            .by_digest(digest)
            .expect("import targets are closure modules");
        let compilation = compile_module_in_closure(self.closure, &entry.name, self.registries)
            .map_err(|diagnostics| SystemError::Elaboration {
                module: entry.name.clone(),
                detail: format!("{diagnostics:?}"),
            })?;
        self.modules.insert(digest.clone(), compilation);
        Ok(())
    }

    fn add_instance(
        &mut self,
        name: &str,
        model_id: GlobalDeclId,
        arguments: &[(String, String)],
        locator: SourceLocator,
    ) -> Result<(), SystemError> {
        let instance = InstanceId(self.instances.len() as u32);
        let compilation = &self.modules[&model_id.module];
        let model = compilation
            .semantic
            .models
            .iter()
            .find(|model| model.name == model_id.name)
            .ok_or_else(|| SystemError::UnknownModel {
                instance: name.to_owned(),
                model: model_id.name.clone(),
            })?
            .clone();
        let semantic_digest = Digest {
            algorithm: "blake3".into(),
            hex: semantic_arena_digest(&compilation.semantic),
        };
        let prefix = if name.is_empty() {
            String::new()
        } else {
            format!("{name}/")
        };
        let display = |local: &str| {
            if name.is_empty() {
                local.to_owned()
            } else {
                format!("{name}.{local}")
            }
        };

        // Domain parameters.
        let mut domain_map = Vec::new();
        for (parameter, value) in arguments {
            let Some(domain) = model
                .domains
                .iter()
                .find(|domain| &domain.name == parameter)
            else {
                return Err(SystemError::UnknownParameter {
                    instance: name.to_owned(),
                    model: model.name.clone(),
                    parameter: parameter.clone(),
                });
            };
            let Some(system_domain) = self.domains.iter().find(|domain| &domain.name == value)
            else {
                return Err(SystemError::UnknownDomain {
                    instance: name.to_owned(),
                    domain: value.clone(),
                });
            };
            if system_domain.dimension != domain.spatial_dimension {
                return Err(SystemError::DomainMismatch {
                    instance: name.to_owned(),
                    parameter: parameter.clone(),
                    expected: domain.spatial_dimension,
                    domain: value.clone(),
                    actual: system_domain.dimension,
                });
            }
            domain_map.push((domain.id, system_domain.id));
            self.origin_domains[system_domain.id.index()]
                .1
                .push((instance, domain.id));
        }
        for domain in &model.domains {
            if !domain_map.iter().any(|(local, _)| *local == domain.id) {
                return Err(SystemError::DomainUnmapped {
                    instance: name.to_owned(),
                    domain: domain.name.clone(),
                });
            }
        }
        domain_map.sort();
        let map_domain = |local: DomainId| {
            domain_map
                .iter()
                .find(|(candidate, _)| *candidate == local)
                .map(|(_, system)| *system)
                .expect("every model domain is mapped")
        };

        // Regions: one system region per instance region.
        let mut region_map = Vec::new();
        for region in &model.regions {
            let id = SysRegionId(self.regions.len() as u32);
            self.regions.push(SystemRegion {
                id,
                name: display(&region.name),
                kind: region.kind.clone(),
                domain: region.domain.map(map_domain),
            });
            region_map.push((region.id, id));
            self.origin_regions.push((id, instance, region.id));
        }

        // Owned states and unknowns.
        for symbol in &model.symbols {
            let SemanticRole::PhysicalField(role @ (FieldRole::Unknown | FieldRole::State)) =
                &symbol.ty.role
            else {
                continue;
            };
            self.variables.push(SysVar {
                id: SysVarId(self.variables.len() as u32),
                owner: instance,
                local: symbol.id,
                name: display(&symbol.name),
                kind: SysVarKind::Field { role: role.clone() },
            });
        }

        // Residual rows.
        for declaration in &model.declarations {
            let SemanticDeclarationKind::Equation { lhs, rhs } = declaration.kind else {
                continue;
            };
            let (orientation, orientation_basis) = orient_equation(&model, lhs, rhs);
            self.residuals.push(SysRes {
                id: SysResId(self.residuals.len() as u32),
                origin: ResidualOrigin::Equation {
                    instance,
                    declaration: declaration.id,
                    name: declaration.name.clone(),
                },
                orientation,
                orientation_basis,
            });
        }

        // Outputs.
        for declaration in &model.declarations {
            let SemanticDeclarationKind::Output { value, domain } = declaration.kind else {
                continue;
            };
            let ty = &model.expressions[value.index()].ty;
            let mut dependencies = BTreeSet::new();
            let _ = crate::objective::close_dependencies(&model, value, &mut dependencies);
            let depends_on_fields = dependencies
                .into_iter()
                .filter(|symbol| {
                    matches!(
                        model.symbols[symbol.index()].ty.role,
                        SemanticRole::PhysicalField(_)
                    )
                })
                .collect();
            self.outputs.push(SystemOutput {
                id: OutputId(self.outputs.len() as u32),
                instance,
                name: declaration.name.clone(),
                declaration: declaration.id,
                expression: value,
                domain: map_domain(domain),
                dimension: ty.dimension,
                quantity_kind: ty.quantity_kind.clone(),
                shape: match &ty.shape {
                    crate::semantic::SemanticShape::Numeric(shape) => Some(shape.clone()),
                    _ => None,
                },
                depends_on_fields,
            });
        }

        // Slots, prefixed.
        let manifest = derive_binding_slots(compilation)
            .into_iter()
            .find(|manifest| manifest.model == model.name)
            .expect("every elaborated model has a manifest");
        for slot in manifest.slots {
            self.slots.push(SystemSlot {
                id: format!("{prefix}{}", slot.id),
                instance,
                local_id: slot.id.clone(),
                slot,
                binding: SlotBinding::Open,
            });
        }

        self.instances.push(InstanceRecord {
            instance,
            name: name.to_owned(),
            prefix,
            model: model_id,
            model_name: model.name.clone(),
            semantic_digest,
            domain_map,
            region_map,
            locator,
        });
        Ok(())
    }

    fn instance_named(&self, name: &str) -> Result<InstanceId, SystemError> {
        self.instances
            .iter()
            .find(|instance| instance.name == name)
            .map(|instance| instance.instance)
            .ok_or_else(|| SystemError::UnknownInstance(name.to_owned()))
    }

    fn model_of(&self, instance: InstanceId) -> &SemanticModel {
        let record = &self.instances[instance.index()];
        self.modules[&record.model.module]
            .semantic
            .models
            .iter()
            .find(|model| model.name == record.model_name)
            .expect("instance records name an elaborated model")
    }

    fn add_bind(
        &mut self,
        consumer_name: &str,
        input: &str,
        producer_name: &str,
        output: &str,
        locator: SourceLocator,
    ) -> Result<(), SystemError> {
        let consumer = self.instance_named(consumer_name)?;
        let producer = self.instance_named(producer_name)?;
        let consumer_prefix = self.instances[consumer.index()].prefix.clone();

        // Consumer input: an `input/`, valueless `source/`, or valueless `parameter/` slot.
        let candidates = [
            format!("{consumer_prefix}input/{input}"),
            format!("{consumer_prefix}source/{input}"),
            format!("{consumer_prefix}parameter/{input}"),
        ];
        let slot_index = self.slots.iter().position(|slot| {
            slot.instance == consumer
                && candidates.contains(&slot.id)
                && slot.slot.status == SlotStatus::Required
        });
        let Some(slot_index) = slot_index else {
            let model = self.model_of(consumer);
            return Err(if model.symbols.iter().any(|symbol| symbol.name == input) {
                SystemError::PrivateSymbol {
                    instance: consumer_name.to_owned(),
                    member: input.to_owned(),
                }
            } else {
                SystemError::UnknownInput {
                    instance: consumer_name.to_owned(),
                    member: input.to_owned(),
                }
            });
        };
        if matches!(self.slots[slot_index].binding, SlotBinding::Bound { .. }) {
            return Err(SystemError::DuplicateBinding(
                self.slots[slot_index].id.clone(),
            ));
        }

        // Producer output.
        let Some(output_record) = self
            .outputs
            .iter()
            .find(|candidate| candidate.instance == producer && candidate.name == output)
        else {
            let model = self.model_of(producer);
            return Err(
                if model.symbols.iter().any(|symbol| symbol.name == output) {
                    SystemError::PrivateSymbol {
                        instance: producer_name.to_owned(),
                        member: output.to_owned(),
                    }
                } else {
                    SystemError::UnknownOutput {
                        instance: producer_name.to_owned(),
                        member: output.to_owned(),
                    }
                },
            );
        };
        let output_id = output_record.id;
        let output_domain = output_record.domain;
        let output_dimension = output_record.dimension;
        let output_kind = output_record.quantity_kind.clone();
        let output_shape = output_record.shape.clone();

        let consumer_slot = &self.slots[slot_index];
        let consumer_id = consumer_slot.id.clone();
        let producer_path = format!("{producer_name}.{output}");
        let consumer_symbol = consumer_slot
            .slot
            .symbol
            .expect("input slots name their symbol");
        if let (Some(expected), Some(actual)) = (consumer_slot.slot.dimension, output_dimension)
            && expected != actual
        {
            return Err(SystemError::BindKindMismatch {
                consumer: consumer_id,
                producer: producer_path,
                detail: format!("dimension {expected} vs {actual}"),
            });
        }
        if let (Some(expected), Some(actual)) = (&consumer_slot.slot.quantity_kind, &output_kind)
            && expected != actual
        {
            return Err(SystemError::BindKindMismatch {
                consumer: consumer_id,
                producer: producer_path,
                detail: format!("quantity kind {} vs {}", expected.as_str(), actual.as_str()),
            });
        }
        if let (Some(expected), Some(actual)) = (&consumer_slot.slot.shape, &output_shape)
            && expected != actual
        {
            return Err(SystemError::BindKindMismatch {
                consumer: consumer_id,
                producer: producer_path,
                detail: format!("shape {expected:?} vs {actual:?}"),
            });
        }
        // Support: an input field lives on a domain; an input value / valueless parameter
        // does not and cannot take a field output.
        let consumer_model = self.model_of(consumer);
        let consumer_domain = consumer_model.symbols[consumer_symbol.index()].domain;
        let record = &self.instances[consumer.index()];
        let chain = match consumer_domain {
            None => {
                return Err(SystemError::BindSupportMismatch {
                    consumer: consumer_id,
                    producer: producer_path,
                });
            }
            Some(local) => {
                let mapped = record
                    .domain_map
                    .iter()
                    .find(|(candidate, _)| *candidate == local)
                    .map(|(_, system)| *system)
                    .expect("mapped");
                if mapped != output_domain {
                    return Err(SystemError::BindCrossDomain {
                        consumer: consumer_id,
                        consumer_domain: self.domains[mapped.index()].name.clone(),
                        producer: producer_path,
                        producer_domain: self.domains[output_domain.index()].name.clone(),
                    });
                }
                BoundChain::Composed
            }
        };
        let bind = self.binds.len();
        self.binds.push(SystemBind {
            consumer,
            consumer_slot: consumer_id,
            consumer_symbol,
            producer: output_id,
            chain,
            locator,
        });
        self.slots[slot_index].binding = SlotBinding::Bound { bind };
        Ok(())
    }

    fn finish(self, span: SourceSpan) -> Result<SystemCompilation, SystemError> {
        let Builder {
            closure,
            modules,
            root,
            name,
            implicit,
            domains,
            regions,
            instances,
            variables,
            residuals,
            outputs,
            slots,
            binds,
            origin_regions,
            origin_domains,
            ..
        } = self;
        let _ = span;

        // Dependency graph over instances.
        let mut graph = Digraph::new(instances.len());
        let mut edges = Vec::new();
        for (index, bind) in binds.iter().enumerate() {
            let producer = outputs[bind.producer.index()].instance;
            let consumer = bind.consumer;
            edges.push(DependencyEdge {
                producer,
                consumer,
                bind: index,
            });
            let _ = graph.add_edge(producer.index(), consumer.index());
        }
        graph.normalize();
        let sccs = tarjan_scc(&graph);
        // Topological order of the condensation: Kahn over components.
        let component_of =
            |node: usize| sccs.component_of(node).expect("every node has a component");
        let component_count = sccs.components().len();
        let mut indegree = vec![0usize; component_count];
        let mut successors = vec![BTreeSet::new(); component_count];
        for edge in &edges {
            let (from, to) = (
                component_of(edge.producer.index()),
                component_of(edge.consumer.index()),
            );
            if from != to && successors[from].insert(to) {
                indegree[to] += 1;
            }
        }
        let mut ready = (0..component_count)
            .filter(|component| indegree[*component] == 0)
            .collect::<BTreeSet<_>>();
        let mut order = Vec::new();
        while let Some(component) = ready.pop_first() {
            order.push(component);
            for next in &successors[component] {
                indegree[*next] -= 1;
                if indegree[*next] == 0 {
                    ready.insert(*next);
                }
            }
        }
        let components = order
            .into_iter()
            .map(|component| {
                let mut members = sccs.components()[component]
                    .iter()
                    .map(|node| InstanceId(*node as u32))
                    .collect::<Vec<_>>();
                members.sort();
                members
            })
            .collect::<Vec<_>>();
        let sequential = components.iter().all(|component| component.len() == 1);
        let dependency = SystemDependency {
            edges,
            components,
            sequential,
        };

        let origin_map = OriginMap {
            variables: variables
                .iter()
                .map(|variable| {
                    let record = &instances[variable.owner.index()];
                    let model = modules[&record.model.module]
                        .semantic
                        .models
                        .iter()
                        .find(|model| model.name == record.model_name)
                        .expect("elaborated");
                    VariableOrigin {
                        variable: variable.id,
                        instance: variable.owner,
                        model: record.model.clone(),
                        symbol: variable.local,
                        locator: SourceLocator {
                            module: record.model.module.clone(),
                            span: model.symbols[variable.local.index()].span,
                        },
                    }
                })
                .collect(),
            residuals: residuals
                .iter()
                .map(|residual| {
                    let ResidualOrigin::Equation {
                        instance,
                        declaration,
                        ..
                    } = &residual.origin;
                    let record = &instances[instance.index()];
                    let model = modules[&record.model.module]
                        .semantic
                        .models
                        .iter()
                        .find(|model| model.name == record.model_name)
                        .expect("elaborated");
                    (
                        residual.id,
                        residual.origin.clone(),
                        SourceLocator {
                            module: record.model.module.clone(),
                            span: model.declarations[declaration.index()].span,
                        },
                    )
                })
                .collect(),
            regions: origin_regions,
            domains: origin_domains,
        };
        let slots = SystemSlotManifest {
            schema: SYSTEM_SLOTS_SCHEMA.into(),
            identity: span_independent_digest(&slots),
            slots,
        };
        let mut system = ScientificSystem {
            schema: SYSTEM_SCHEMA.into(),
            root,
            name,
            implicit,
            closure_identity: closure.identity.clone(),
            domains,
            regions,
            instances,
            variables,
            residuals,
            outputs,
            slots,
            binds,
            dependency,
            origin_map,
            identity: Digest::blake3(b"unset"),
        };
        system.identity = system.expected_identity();
        Ok(SystemCompilation { system, modules })
    }
}

/// §3.3 orientation: a residual `lhs - rhs` whose top-level sum carries a `dt(...)` term is
/// oriented so that term is positive.
fn orient_equation(model: &SemanticModel, lhs: ExprId, rhs: ExprId) -> (i8, OrientationBasis) {
    fn find_dt(model: &SemanticModel, id: ExprId, sign: i8) -> Option<i8> {
        match &model.expressions[id.index()].kind {
            SemanticExprKind::Differential {
                operator: crate::semantic::DifferentialOperator::TimeDerivative,
                ..
            } => Some(sign),
            SemanticExprKind::Unary {
                op: crate::scientific::UnaryOp::Neg,
                arg,
            } => find_dt(model, *arg, -sign),
            SemanticExprKind::Binary { op, lhs, rhs } => match op {
                crate::scientific::BinaryOp::Add => {
                    find_dt(model, *lhs, sign).or_else(|| find_dt(model, *rhs, sign))
                }
                crate::scientific::BinaryOp::Sub => {
                    find_dt(model, *lhs, sign).or_else(|| find_dt(model, *rhs, -sign))
                }
                crate::scientific::BinaryOp::Mul => {
                    find_dt(model, *lhs, sign).or_else(|| find_dt(model, *rhs, sign))
                }
                _ => None,
            },
            _ => None,
        }
    }
    match find_dt(model, lhs, 1).or_else(|| find_dt(model, rhs, -1)) {
        Some(sign) => (sign, OrientationBasis::Accumulation),
        None => (1, OrientationBasis::Unoriented),
    }
}

#[derive(Serialize)]
struct SystemIdentity<'a> {
    schema: &'a str,
    root: &'a GlobalDeclId,
    name: &'a str,
    implicit: bool,
    closure_identity: &'a str,
    domains: &'a [SystemDomain],
    regions: &'a [SystemRegion],
    instances: &'a [InstanceRecord],
    variables: &'a [SysVar],
    residuals: &'a [SysRes],
    outputs: &'a [SystemOutput],
    slots: &'a Digest,
    binds: &'a [SystemBind],
    dependency: &'a SystemDependency,
}

impl ScientificSystem {
    fn expected_identity(&self) -> Digest {
        span_independent_digest(&SystemIdentity {
            schema: &self.schema,
            root: &self.root,
            name: &self.name,
            implicit: self.implicit,
            closure_identity: &self.closure_identity,
            domains: &self.domains,
            regions: &self.regions,
            instances: &self.instances,
            variables: &self.variables,
            residuals: &self.residuals,
            outputs: &self.outputs,
            slots: &self.slots.identity,
            binds: &self.binds,
            dependency: &self.dependency,
        })
    }

    pub fn validate(&self) -> Result<(), SystemError> {
        if self.identity != self.expected_identity() {
            return Err(SystemError::Elaboration {
                module: self.name.clone(),
                detail: "SYSTEM_IDENTITY_MISMATCH: identity does not match contents".into(),
            });
        }
        Ok(())
    }

    /// Every `Required` slot that is neither bound by a `bind` nor named in `case_bound`
    /// (acceptance test 4: removing a bind without re-binding the input is an open input).
    pub fn require_closed(&self, case_bound: &BTreeSet<String>) -> Result<(), SystemError> {
        for slot in &self.slots.slots {
            if slot.slot.status == SlotStatus::Required
                && matches!(slot.binding, SlotBinding::Open)
                && !case_bound.contains(&slot.id)
                && matches!(
                    slot.slot.kind,
                    SlotKind::ExternalValue | SlotKind::Parameter | SlotKind::Provider { .. }
                )
            {
                return Err(SystemError::OpenInput(slot.id.clone()));
            }
        }
        Ok(())
    }

    /// Slot ids that are `Required` and still `Open`.
    pub fn open_inputs(&self) -> Vec<&SystemSlot> {
        self.slots
            .slots
            .iter()
            .filter(|slot| {
                slot.slot.status == SlotStatus::Required
                    && matches!(slot.binding, SlotBinding::Open)
            })
            .collect()
    }

    pub fn instance_named(&self, name: &str) -> Option<&InstanceRecord> {
        self.instances.iter().find(|instance| instance.name == name)
    }
}
