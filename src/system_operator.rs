//! SC-W1: `scientia-operator-system/2` (`sinbad/ARCHITECTURE.md` §2.6, §6), the system-level
//! operator artifact keyed by dense system ids. Per-model `scientia-operator-system/1`
//! artifacts are compiled once per model through the unchanged FC2–FC5 chain and referenced by
//! digest (§2.2: reused verbatim by every instance). A same-domain `bind` becomes a
//! `Composed` block: the consumer's per-model kernel keeps its bound input as an external
//! field operand, the producer's `output` kernel (the FC2–FC5 chain over the synthetic
//! functional of [`crate::derive_output_form`]) writes that operand through a Malleus
//! [`KernelComposition`] shared buffer, and the cross block is the composition's JVP. No
//! expression is rewritten and no kernel is fused.

use crate::composition::{
    InstanceId, InstanceRecord, OriginMap, OutputId, ResidualOrigin, ScientificSystem, SysResId,
    SysVar, SysVarId, SystemCompilation,
};
use crate::id::{Digest, span_independent_digest};
use crate::scientific::GlobalDeclId;
use crate::semantic::{SemanticDeclarationKind, SymbolId};
use crate::system::{OperatorSystem, block_digest, compile_operator_system};
use crate::{
    FormRequirements, OperatorFactorization, StructuredOperatorKernels, TensorInputRole,
    VariationalForm, derive_output_form, factor_operator, infer_form_requirements,
    lower_operator_kernels,
};
use malleus::{
    CompositionDerivativeRequest, KernelComposition, SharedBuffer, StageOperand,
    composition_digest, differentiate_composition, validate_composition,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

pub const SYSTEM_OPERATOR_SCHEMA: &str = "scientia-operator-system/2";

/// One residual row: the per-model block it is, by reference.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SysResBlock {
    pub id: SysResId,
    pub origin: ResidualOrigin,
    /// The per-model row symbol (the field whose space supplied the test argument).
    pub row: SymbolId,
    pub orientation: i8,
    /// `artifact_digest` of the instance model's `scientia-operator-system/1`.
    pub model_system: Digest,
    /// [`block_digest`] of the per-model block.
    pub block: Digest,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BlockConstruction {
    /// The instance's own factorization: its structured kernels, by digest.
    Local { kernels: Digest },
    /// A same-domain bind chain (§6): the producer's output kernel is evaluated at the
    /// consumer's quadrature point and feeds the bound input along `path`.
    Composed {
        bind: usize,
        consumer_slot: String,
        producer: OutputId,
        output_kernels: Digest,
        path: ComposedPath,
    },
}

/// How a bound input reaches the consumer's residual (§6). Both are honest about the tangent.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComposedPath {
    /// The consumer's residual kernels read the input as an external operand: the listed
    /// Malleus compositions (one per consumer kernel bundle that reads it) share the producer's
    /// output buffer with that operand, and `jvp_compositions` are their JVPs, the cross-block
    /// tangents.
    KernelInput {
        compositions: Vec<Digest>,
        jvp_compositions: Vec<Digest>,
    },
    /// The consumer reads the input only as an argument of provider calls inside the listed
    /// model-defined properties (`sigma = electrical_conductivity(temperature)`): the
    /// producer's output value feeds those provider evaluations at the quadrature point, which
    /// Finitum realizes through the property's `FieldSource`. The cross-block tangent is the
    /// chain of the consumer's frozen-input parameter JVP with respect to the property operand,
    /// the property's own tangent (GX-A2 `scientia-property-kernel/1`, runtime-provided), and
    /// the output kernel's JVP; Scientia emits no fused composition for it.
    ProviderInput { properties: Vec<PropertyPath> },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropertyPath {
    pub property: SymbolId,
    /// The consumer's `property/<name>` slot.
    pub slot: String,
    /// The `provider/<name>` slots whose calls take the bound input.
    pub providers: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SysBlock {
    pub row: SysResId,
    pub column: SysVarId,
    pub construction: BlockConstruction,
}

/// The digest chain of one output kernel.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputKernelRecord {
    pub output: OutputId,
    pub form: Digest,
    pub requirements: Digest,
    pub factorization: Digest,
    pub kernels: Digest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SystemOperator {
    pub schema: String,
    pub root: GlobalDeclId,
    pub system_identity: Digest,
    pub instances: Vec<InstanceRecord>,
    /// Per instance, the `artifact_digest` of the `scientia-operator-system/1` it reuses.
    pub instance_artifacts: Vec<(InstanceId, Digest)>,
    pub variables: Vec<SysVar>,
    pub residuals: Vec<SysResBlock>,
    /// Sorted by `(row, column)`.
    pub blocks: Vec<SysBlock>,
    pub outputs: Vec<OutputKernelRecord>,
    pub field_order: Vec<SysVarId>,
    pub origin_map: OriginMap,
    pub identity: Digest,
}

/// The complete artifacts of one output kernel.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OutputKernels {
    pub output: OutputId,
    pub form: VariationalForm,
    pub requirements: FormRequirements,
    pub factorization: OperatorFactorization,
    pub kernels: StructuredOperatorKernels,
}

/// One Malleus composition realizing a bind for one consumer kernel bundle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BindComposition {
    pub bind: usize,
    pub row: SysResId,
    pub consumer_integral_index: usize,
    pub consumer_output_index: usize,
    pub composition: KernelComposition,
    pub digest: Digest,
    /// Stage 0 = producer output primal kernel, stage 1 = consumer primal kernel.
    pub producer_operand: StageOperand,
    pub consumer_operands: Vec<StageOperand>,
    pub jvp_digest: Digest,
}

/// Everything `compile_system_operator` produces: the artifact plus the per-model, output, and
/// composition payloads it references by digest.
#[derive(Clone, Debug, PartialEq)]
pub struct SystemOperatorCompilation {
    pub system: ScientificSystem,
    pub operator: SystemOperator,
    /// Per-instance `scientia-operator-system/1` artifacts, verbatim; two instances of one
    /// model carry equal digests.
    pub model_systems: Vec<(InstanceId, OperatorSystem)>,
    pub output_kernels: Vec<OutputKernels>,
    pub compositions: Vec<BindComposition>,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum SystemOperatorError {
    #[error(
        "SYSTEM_OPERATOR_NO_RESIDUALS: instance `{instance}` (model `{model}`) declares no equation"
    )]
    NoResiduals { instance: String, model: String },
    #[error("SYSTEM_OPERATOR_MODEL: instance `{instance}`: {detail}")]
    Model { instance: String, detail: String },
    #[error("SYSTEM_OPERATOR_OUTPUT: output `{output}` failed during {stage}: {detail}")]
    Output {
        output: String,
        stage: &'static str,
        detail: String,
    },
    #[error(
        "SYSTEM_OPERATOR_UNBOUND_COLUMN: instance `{instance}` block column {column} is not a system variable"
    )]
    UnboundColumn { instance: String, column: SymbolId },
    #[error(
        "SYSTEM_OPERATOR_BIND_UNUSED: bind `{consumer_slot}` closes an input no consumer kernel reads"
    )]
    BindUnused { consumer_slot: String },
    #[error("SYSTEM_OPERATOR_COMPOSITION: bind `{consumer_slot}`: {detail}")]
    Composition {
        consumer_slot: String,
        detail: String,
    },
}

/// Compile the system-level operator artifact from an elaborated system.
pub fn compile_system_operator(
    compilation: &SystemCompilation,
) -> Result<SystemOperatorCompilation, SystemOperatorError> {
    let system = &compilation.system;

    // Per-model artifacts, once per model declaration.
    let mut by_model: BTreeMap<GlobalDeclId, OperatorSystem> = BTreeMap::new();
    let mut model_systems = Vec::new();
    for record in &system.instances {
        if !by_model.contains_key(&record.model) {
            let model = compilation.model(record.instance);
            let equations = model
                .declarations
                .iter()
                .filter(|declaration| {
                    matches!(declaration.kind, SemanticDeclarationKind::Equation { .. })
                })
                .map(|declaration| declaration.name.as_str())
                .collect::<Vec<_>>();
            if equations.is_empty() {
                return Err(SystemOperatorError::NoResiduals {
                    instance: record.name.clone(),
                    model: record.model_name.clone(),
                });
            }
            let module = &compilation.compilation(record.instance).semantic;
            let artifact = compile_operator_system(module, &record.model_name, &equations)
                .map_err(|error| SystemOperatorError::Model {
                    instance: record.name.clone(),
                    detail: error.to_string(),
                })?;
            by_model.insert(record.model.clone(), artifact);
        }
        model_systems.push((record.instance, by_model[&record.model].clone()));
    }

    let variable_of = |instance: InstanceId, local: SymbolId| {
        system
            .variables
            .iter()
            .find(|variable| variable.owner == instance && variable.local == local)
            .map(|variable| variable.id)
    };

    // Rows and local blocks.
    let mut residuals = Vec::new();
    let mut blocks = Vec::new();
    for residual in &system.residuals {
        let ResidualOrigin::Equation { instance, name, .. } = &residual.origin;
        let artifact = &model_systems
            .iter()
            .find(|(candidate, _)| candidate == instance)
            .expect("every instance has a model artifact")
            .1;
        let block = artifact
            .blocks
            .iter()
            .find(|block| &block.equation == name)
            .expect("every equation was selected");
        residuals.push(SysResBlock {
            id: residual.id,
            origin: residual.origin.clone(),
            row: block.row,
            orientation: residual.orientation,
            model_system: artifact.artifact_digest.clone(),
            block: block_digest(block),
        });
        for column in &block.columns {
            let column = variable_of(*instance, *column).ok_or_else(|| {
                SystemOperatorError::UnboundColumn {
                    instance: system.instances[instance.index()].name.clone(),
                    column: *column,
                }
            })?;
            blocks.push(SysBlock {
                row: residual.id,
                column,
                construction: BlockConstruction::Local {
                    kernels: block.kernels.artifact_digest.clone(),
                },
            });
        }
    }

    // Output kernels.
    let mut output_kernels = Vec::new();
    let mut output_records = Vec::new();
    for output in &system.outputs {
        let module = &compilation.compilation(output.instance).semantic;
        let model_name = &system.instances[output.instance.index()].model_name;
        let stage = |stage: &'static str, detail: String| SystemOperatorError::Output {
            output: format!(
                "{}.{}",
                system.instances[output.instance.index()].name,
                output.name
            ),
            stage,
            detail,
        };
        let form = derive_output_form(module, model_name, &output.name)
            .map_err(|error| stage("form derivation", error.to_string()))?;
        let requirements = infer_form_requirements(module, &form)
            .map_err(|error| stage("requirement inference", error.to_string()))?;
        let factorization = factor_operator(&form, &requirements)
            .map_err(|error| stage("operator factorization", error.to_string()))?;
        let kernels = lower_operator_kernels(&factorization)
            .map_err(|error| stage("structured-kernel lowering", error.to_string()))?;
        output_records.push(OutputKernelRecord {
            output: output.id,
            form: form.artifact_digest.clone(),
            requirements: requirements.artifact_digest.clone(),
            factorization: factorization.artifact_digest.clone(),
            kernels: kernels.artifact_digest.clone(),
        });
        output_kernels.push(OutputKernels {
            output: output.id,
            form,
            requirements,
            factorization,
            kernels,
        });
    }

    // Composed blocks, one per bind.
    let mut compositions = Vec::new();
    for (bind_index, bind) in system.binds.iter().enumerate() {
        let output = &system.outputs[bind.producer.index()];
        let producer_kernels = &output_kernels[bind.producer.index()];
        let [producer_bundle] = producer_kernels.kernels.bundles.as_slice() else {
            return Err(SystemOperatorError::Composition {
                consumer_slot: bind.consumer_slot.clone(),
                detail: format!(
                    "output kernel has {} bundles; an output is one point function",
                    producer_kernels.kernels.bundles.len()
                ),
            });
        };
        let producer_kernel =
            producer_bundle.module.kernels[producer_bundle.primal_kernel_index].clone();
        let producer_operand = StageOperand::new(0, producer_bundle.primal_output);
        let consumer_artifact = &model_systems
            .iter()
            .find(|(candidate, _)| *candidate == bind.consumer)
            .expect("consumer instance has an artifact")
            .1;
        let mut digests = Vec::new();
        let mut jvp_digests = Vec::new();
        let mut used = false;
        for block in &consumer_artifact.blocks {
            let row = system
                .residuals
                .iter()
                .find(|residual| {
                    let ResidualOrigin::Equation { instance, name, .. } = &residual.origin;
                    *instance == bind.consumer && name == &block.equation
                })
                .expect("every block is a row")
                .id;
            for bundle in &block.kernels.bundles {
                let program = &block.factorization.integrals[bundle.integral_index].primal;
                let consumer_operands = bundle
                    .primal_inputs
                    .iter()
                    .filter(|binding| {
                        program.inputs.iter().any(|input| {
                            input.id == binding.input
                                && input.binding.symbol == bind.consumer_symbol
                        })
                    })
                    .map(|binding| StageOperand::new(1, binding.operand))
                    .collect::<Vec<_>>();
                if consumer_operands.is_empty() {
                    continue;
                }
                used = true;
                let consumer_kernel = bundle.module.kernels[bundle.primal_kernel_index].clone();
                let mut members = vec![producer_operand];
                members.extend(consumer_operands.iter().copied());
                let composition = KernelComposition {
                    name: format!(
                        "{}::{}<-{}.{}",
                        block.kernels.bundles[0].module.name,
                        bind.consumer_slot,
                        system.instances[output.instance.index()].name,
                        output.name
                    ),
                    stages: vec![producer_kernel.clone(), consumer_kernel],
                    shared_buffers: vec![SharedBuffer::new(members)],
                };
                validate_composition(composition.clone()).map_err(|error| {
                    SystemOperatorError::Composition {
                        consumer_slot: bind.consumer_slot.clone(),
                        detail: error.to_string(),
                    }
                })?;
                // The cross-block tangent: producer active field evaluations -> consumer output.
                let producer_program = &producer_kernels.factorization.integrals
                    [producer_bundle.integral_index]
                    .primal;
                let independent = producer_bundle
                    .primal_inputs
                    .iter()
                    .filter(|binding| {
                        producer_program.inputs.iter().any(|input| {
                            input.id == binding.input && input.role == TensorInputRole::Active
                        })
                    })
                    .map(|binding| StageOperand::new(0, binding.operand))
                    .collect::<Vec<_>>();
                let jvp = differentiate_composition(
                    &composition,
                    &CompositionDerivativeRequest {
                        mode: malleus::DerivativeMode::Jvp,
                        independent_operands: independent,
                        dependent_operands: vec![StageOperand::new(1, bundle.primal_output)],
                    },
                )
                .map_err(|error| SystemOperatorError::Composition {
                    consumer_slot: bind.consumer_slot.clone(),
                    detail: format!("JVP: {error}"),
                })?;
                let digest = convert(composition_digest(&composition));
                let jvp_digest = convert(composition_digest(&jvp.composition));
                digests.push(digest.clone());
                jvp_digests.push(jvp_digest.clone());
                compositions.push(BindComposition {
                    bind: bind_index,
                    row,
                    consumer_integral_index: bundle.integral_index,
                    consumer_output_index: bundle.output_index,
                    composition,
                    digest,
                    producer_operand,
                    consumer_operands,
                    jvp_digest,
                });
            }
        }
        let path = if used {
            ComposedPath::KernelInput {
                compositions: digests.clone(),
                jvp_compositions: jvp_digests.clone(),
            }
        } else {
            let consumer_model = compilation.model(bind.consumer);
            let consumer_prefix = &system.instances[bind.consumer.index()].prefix;
            let mut properties = Vec::new();
            for declaration in &consumer_model.declarations {
                let (SemanticDeclarationKind::Property { value }
                | SemanticDeclarationKind::ConstitutiveLaw { value }) = declaration.kind
                else {
                    continue;
                };
                let mut dependencies = std::collections::BTreeSet::new();
                let _ =
                    crate::objective::close_dependencies(consumer_model, value, &mut dependencies);
                if !dependencies.contains(&bind.consumer_symbol) {
                    continue;
                }
                let mut providers = std::collections::BTreeSet::new();
                let _ =
                    crate::objective::collect_provider_slots(consumer_model, value, &mut providers);
                let kind = if matches!(declaration.kind, SemanticDeclarationKind::Property { .. }) {
                    "property"
                } else {
                    "constitutive"
                };
                properties.push(PropertyPath {
                    property: declaration.symbol.expect("properties name a symbol"),
                    slot: format!("{consumer_prefix}{kind}/{}", declaration.name),
                    providers: providers
                        .into_iter()
                        .map(|provider| format!("{consumer_prefix}provider/{provider}"))
                        .collect(),
                });
            }
            if properties.is_empty() {
                return Err(SystemOperatorError::BindUnused {
                    consumer_slot: bind.consumer_slot.clone(),
                });
            }
            ComposedPath::ProviderInput { properties }
        };
        // The composed block occupies every (consumer row that reads the input, producer
        // field) coordinate.
        // Rows that read the input: through a kernel operand, or through any residual of the
        // consumer whose form captures one of the properties on the provider path.
        let mut rows = compositions
            .iter()
            .filter(|composition| composition.bind == bind_index)
            .map(|composition| composition.row)
            .collect::<std::collections::BTreeSet<_>>();
        if let ComposedPath::ProviderInput { properties } = &path {
            for block in &consumer_artifact.blocks {
                if block.form.captures.iter().any(|capture| {
                    properties
                        .iter()
                        .any(|property| property.property == capture.symbol)
                }) {
                    let row = system
                        .residuals
                        .iter()
                        .find(|residual| {
                            let ResidualOrigin::Equation { instance, name, .. } = &residual.origin;
                            *instance == bind.consumer && name == &block.equation
                        })
                        .expect("every block is a row")
                        .id;
                    rows.insert(row);
                }
            }
        }
        for row in rows {
            for field in &output.depends_on_fields {
                let Some(column) = variable_of(output.instance, *field) else {
                    continue;
                };
                blocks.push(SysBlock {
                    row,
                    column,
                    construction: BlockConstruction::Composed {
                        bind: bind_index,
                        consumer_slot: bind.consumer_slot.clone(),
                        producer: bind.producer,
                        output_kernels: producer_kernels.kernels.artifact_digest.clone(),
                        path: path.clone(),
                    },
                });
            }
        }
    }
    blocks.sort_by_key(|block| (block.row, block.column));

    let field_order = system
        .variables
        .iter()
        .map(|variable| variable.id)
        .collect();
    let mut operator = SystemOperator {
        schema: SYSTEM_OPERATOR_SCHEMA.into(),
        root: system.root.clone(),
        system_identity: system.identity.clone(),
        instances: system.instances.clone(),
        instance_artifacts: model_systems
            .iter()
            .map(|(instance, artifact)| (*instance, artifact.artifact_digest.clone()))
            .collect(),
        variables: system.variables.clone(),
        residuals,
        blocks,
        outputs: output_records,
        field_order,
        origin_map: system.origin_map.clone(),
        identity: Digest::blake3(b"unset"),
    };
    operator.identity = operator.expected_identity();
    Ok(SystemOperatorCompilation {
        system: system.clone(),
        operator,
        model_systems,
        output_kernels,
        compositions,
    })
}

fn convert(digest: malleus::Digest) -> Digest {
    Digest {
        algorithm: digest.algorithm,
        hex: digest.hex,
    }
}

#[derive(Serialize)]
struct SystemOperatorIdentity<'a> {
    schema: &'a str,
    root: &'a GlobalDeclId,
    system_identity: &'a Digest,
    instances: &'a [InstanceRecord],
    instance_artifacts: &'a [(InstanceId, Digest)],
    variables: &'a [SysVar],
    residuals: &'a [SysResBlock],
    blocks: &'a [SysBlock],
    outputs: &'a [OutputKernelRecord],
    field_order: &'a [SysVarId],
}

impl SystemOperator {
    fn expected_identity(&self) -> Digest {
        span_independent_digest(&SystemOperatorIdentity {
            schema: &self.schema,
            root: &self.root,
            system_identity: &self.system_identity,
            instances: &self.instances,
            instance_artifacts: &self.instance_artifacts,
            variables: &self.variables,
            residuals: &self.residuals,
            blocks: &self.blocks,
            outputs: &self.outputs,
            field_order: &self.field_order,
        })
    }

    pub fn validate(&self) -> Result<(), SystemOperatorError> {
        if self.identity != self.expected_identity() {
            return Err(SystemOperatorError::Composition {
                consumer_slot: String::new(),
                detail: "SYSTEM_OPERATOR_IDENTITY_MISMATCH".into(),
            });
        }
        if self
            .blocks
            .windows(2)
            .any(|pair| (pair[0].row, pair[0].column) > (pair[1].row, pair[1].column))
        {
            return Err(SystemOperatorError::Composition {
                consumer_slot: String::new(),
                detail: "SYSTEM_OPERATOR_NONCANONICAL: blocks must be sorted".into(),
            });
        }
        Ok(())
    }
}
