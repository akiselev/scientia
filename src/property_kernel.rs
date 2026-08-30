//! GX-A2 property kernels (contract C7): lower a [`PropertyDefinition`] into a Malleus
//! `StructuredModule` holding one value kernel and, for `Symbolic`/`Automatic` differentiability,
//! one tangent kernel per declared input, using [`crate::projection`] for the symbolic
//! differentiation.
//!
//! This is the narrow scalar point-kernel path (the same one `kernel::lower_local_program` uses
//! for a form integral), applied to a standalone property expression instead of a form: a
//! [`PropertyDefinition`] carries its own declared input signature (contract C1.1's provider
//! inputs, reused here), so it needs no surrounding `.res` model to elaborate against --
//! [`crate::projection::lift_standalone_expr`] elaborates it directly.
//!
//! ## Scope
//!
//! - `Constant`/`Expression` models lower fully, with tangent kernels for every
//!   `Symbolic`/`Automatic` input.
//! - `Piecewise` and `Table` models are refused (`PROPERTY_KERNEL_UNSUPPORTED`): a `Piecewise`
//!   kernel needs a branch-selection primitive and `Table` needs an indexing primitive that
//!   Malleus's `StructuredModule` does not yet expose (the frozen contract explicitly allows
//!   refusing table kernels for this reason).
//! - `External` models are refused: an external provider has no closed form to lower.
//! - `AnalyticProvided`, `Piecewise`, `NumericalAllowed`, and `None` differentiability contracts
//!   get a value kernel but no tangent kernel (contract C7 asks only for tangents "from symbolic
//!   differentiation through the projection for `Symbolic`/`Automatic`").
//! - A `Constant` model's unit-bearing literal (contract C7 point 4) canonicalizes to SI via the
//!   passed-in Quantitas registry, both for a simple bare-unit literal (`"300 K"`) and, since
//!   GX-F3 extended the `.res` expression grammar's trailing-unit rule to accept a compound unit
//!   token sequence, for a compound unit expression (`"1.0 m^2/s"`); an unresolvable unit is
//!   refused with `RESOLVE_UNKNOWN_UNIT`.

use crate::id::{Digest, span_independent_digest};
use crate::kernel::{self, KernelLoweringError};
use crate::projection::{AlgebraOperation, AlgebraRefusal, lift_standalone_expr, project};
use crate::scientific::{
    DerivativeContract, Expr, PropertyDefinition, PropertyInput, PropertyModel, ScientificError,
    canonicalize_authored_quantity,
};
use crate::semantic::{SemanticModel, SymbolId};
use crate::source::SourceSpan;
use malleus::{
    AccessMode, IndexingMap, IterationDomain, KernelOperand, KernelRegion, NumericPolicy,
    OperandId, Statement, StructuredKernel, StructuredModule,
};
use quantitas::{QuantityKindId, QuantityLiteral, UnitId, UnitRegistry};
use resolvent::TermAlgebraReceipt;
use serde::Serialize;
use thiserror::Error;

pub const PROPERTY_KERNEL_SCHEMA: &str = "scientia-property-kernel/1";

#[derive(Clone, Debug, PartialEq)]
pub struct PropertyTangent {
    pub input: String,
    pub kernel: usize,
    pub contract: DerivativeContract,
    pub receipt: Option<TermAlgebraReceipt>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PropertyKernel {
    pub schema: String,
    pub definition_digest: Digest,
    pub module: StructuredModule,
    pub value_kernel: usize,
    pub tangents: Vec<PropertyTangent>,
    pub inputs: Vec<crate::binding_slots::SlotInput>,
    pub identity: Digest,
}

#[derive(Debug, Error)]
pub enum PropertyKernelError {
    #[error("PROPERTY_KERNEL_UNSUPPORTED: {0}")]
    Unsupported(String),
    #[error("PROPERTY_KERNEL_PARSE: {0}")]
    Parse(ScientificError),
    #[error("{0}")]
    Projection(AlgebraRefusal),
    #[error(transparent)]
    Kernel(#[from] KernelLoweringError),
}

/// Lower `definition` into a Malleus `StructuredModule` per contract C7.
pub fn lower_property_kernel(
    definition: &PropertyDefinition,
    registry: &UnitRegistry,
) -> Result<PropertyKernel, PropertyKernelError> {
    match &definition.model {
        PropertyModel::Constant(expr) => {
            let canonical = canonicalize_constant_units(expr, registry)?;
            build_kernel(definition, &canonical, &[])
        }
        PropertyModel::Expression(expr) => {
            build_kernel(definition, expr, &definition.signature.inputs)
        }
        PropertyModel::Piecewise(_) => Err(PropertyKernelError::Unsupported(
            "piecewise property kernels need a branch-selection primitive Malleus's \
             StructuredModule does not yet expose"
                .into(),
        )),
        PropertyModel::Table(_) => Err(PropertyKernelError::Unsupported(
            "table property kernels need an indexing primitive Malleus's StructuredModule \
             does not yet expose"
                .into(),
        )),
        PropertyModel::External(reference) => Err(PropertyKernelError::Unsupported(format!(
            "external provider `{}:{}` has no closed-form kernel to lower",
            reference.provider, reference.property
        ))),
    }
}

fn canonicalize_constant_units(
    expr: &Expr,
    registry: &UnitRegistry,
) -> Result<Expr, PropertyKernelError> {
    match expr {
        Expr::Number {
            value,
            unit: Some(unit_name),
            span,
            ..
        } => {
            let literal = QuantityLiteral {
                value: *value,
                unit: UnitId::new(unit_name.clone()),
                kind: QuantityKindId::new("scientia:Unspecified"),
            };
            let quantity = canonicalize_authored_quantity(registry, &literal).map_err(|error| {
                PropertyKernelError::Unsupported(format!("RESOLVE_UNKNOWN_UNIT: {error}"))
            })?;
            let si = quantity.value_si();
            Ok(Expr::Number {
                value: si,
                lexeme: format!("{si}"),
                unit: None,
                span: *span,
            })
        }
        other => Ok(other.clone()),
    }
}

fn build_kernel(
    definition: &PropertyDefinition,
    expr: &Expr,
    inputs: &[PropertyInput],
) -> Result<PropertyKernel, PropertyKernelError> {
    let mut symbols = Vec::new();
    let mut expressions = Vec::new();
    let value_id = lift_standalone_expr(expr, inputs, &mut symbols, &mut expressions)
        .map_err(PropertyKernelError::Parse)?;
    let model = SemanticModel {
        name: definition.signature.id.clone(),
        domains: vec![],
        regions: vec![],
        providers: vec![],
        symbols,
        expressions: expressions.into(),
        declarations: vec![],
        span: SourceSpan::default(),
    };
    let operands: Vec<SymbolId> = inputs
        .iter()
        .map(|input| {
            model
                .symbols
                .iter()
                .find(|symbol| symbol.name == input.name)
                .map(|symbol| symbol.id)
                .expect("lift_standalone_expr declares a symbol for every declared input")
        })
        .collect();

    let mut kernels = Vec::new();
    let value_kernel = kernels.len();
    kernels.push(lower_scalar_kernel(
        &format!("{}::value", definition.signature.id),
        &model.expressions,
        value_id,
        &operands,
    )?);

    let mut tangents = Vec::new();
    if matches!(
        definition.signature.differentiability,
        DerivativeContract::Symbolic | DerivativeContract::Automatic
    ) {
        for input in inputs {
            let wrt = model
                .symbols
                .iter()
                .find(|symbol| symbol.name == input.name)
                .map(|symbol| symbol.id)
                .expect("declared input has a symbol");
            let outcome = project(&model, value_id, &AlgebraOperation::Differentiate { wrt })
                .map_err(PropertyKernelError::Projection)?;
            let kernel_index = kernels.len();
            kernels.push(lower_scalar_kernel(
                &format!("{}::d_d_{}", definition.signature.id, input.name),
                &outcome.arena,
                outcome.result,
                &operands,
            )?);
            tangents.push(PropertyTangent {
                input: input.name.clone(),
                kernel: kernel_index,
                contract: definition.signature.differentiability.clone(),
                receipt: Some(outcome.receipt),
            });
        }
    }

    let module = StructuredModule {
        name: definition.signature.id.clone(),
        kernels,
    };
    let definition_digest = span_independent_digest(definition);
    let slot_inputs: Vec<crate::binding_slots::SlotInput> =
        inputs.iter().map(to_slot_input).collect();

    #[derive(Serialize)]
    struct IdentityPayload<'a> {
        schema: &'static str,
        definition_digest: &'a Digest,
        module: &'a StructuredModule,
        value_kernel: usize,
        tangents: &'a [TangentIdentity<'a>],
        inputs: &'a [crate::binding_slots::SlotInput],
    }
    #[derive(Serialize)]
    struct TangentIdentity<'a> {
        input: &'a str,
        kernel: usize,
        contract: &'a DerivativeContract,
    }
    let tangent_identities: Vec<TangentIdentity> = tangents
        .iter()
        .map(|tangent| TangentIdentity {
            input: &tangent.input,
            kernel: tangent.kernel,
            contract: &tangent.contract,
        })
        .collect();
    let identity = span_independent_digest(&IdentityPayload {
        schema: PROPERTY_KERNEL_SCHEMA,
        definition_digest: &definition_digest,
        module: &module,
        value_kernel,
        tangents: &tangent_identities,
        inputs: &slot_inputs,
    });

    Ok(PropertyKernel {
        schema: PROPERTY_KERNEL_SCHEMA.into(),
        definition_digest,
        module,
        value_kernel,
        tangents,
        inputs: slot_inputs,
        identity,
    })
}

fn to_slot_input(input: &PropertyInput) -> crate::binding_slots::SlotInput {
    crate::binding_slots::SlotInput {
        name: input.name.clone(),
        quantity_kind: Some(input.quantity_kind.clone()),
        dimension: Some(input.dimension),
        shape: Some(input.shape.clone()),
    }
}

fn lower_scalar_kernel(
    name: &str,
    expressions: &[crate::semantic::SemanticExpr],
    id: crate::semantic::ExprId,
    operand_symbols: &[SymbolId],
) -> Result<StructuredKernel, PropertyKernelError> {
    let mut operands = Vec::with_capacity(operand_symbols.len() + 1);
    let mut bindings = std::collections::BTreeMap::new();
    for symbol in operand_symbols {
        let operand_id = OperandId::new(operands.len());
        operands.push(KernelOperand::scalar(
            format!("symbol_{}", symbol.0),
            AccessMode::Read,
        ));
        bindings.insert(*symbol, operand_id);
    }
    let output = OperandId::new(operands.len());
    operands.push(KernelOperand::scalar("value", AccessMode::Write));
    let indexing_maps = (0..operands.len())
        .map(|index| IndexingMap::scalar(OperandId::new(index)))
        .collect();
    let value = kernel::lower_expr(expressions, id, &bindings)?;
    Ok(StructuredKernel {
        name: name.into(),
        iteration_domain: IterationDomain::default(),
        iterators: Vec::new(),
        operands,
        indexing_maps,
        body: KernelRegion {
            statements: vec![Statement::Store {
                operand: output,
                value,
            }],
        },
        numeric_policy: NumericPolicy::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scientific::{
        DerivativeContract, FrameSemantics, PropertyDomain, PropertyEvidence, PropertyOutput,
        PropertySignature, TensorSymmetry, ValueShape,
    };
    use quantitas::Dimension;

    fn definition(differentiability: DerivativeContract, expr_text: &str) -> PropertyDefinition {
        let expr = crate::scientific::parse_expression(expr_text).unwrap();
        PropertyDefinition {
            signature: PropertySignature {
                id: "thermal_conductivity".into(),
                inputs: vec![PropertyInput {
                    name: "T".into(),
                    quantity_kind: QuantityKindId::new("ThermodynamicTemperature"),
                    dimension: Dimension::DIMENSIONLESS,
                    shape: ValueShape::Scalar,
                    physical_min: None,
                    physical_max: None,
                    nominal: None,
                }],
                output: PropertyOutput {
                    quantity_kind: QuantityKindId::new("ThermalConductivity"),
                    dimension: Dimension::DIMENSIONLESS,
                    shape: ValueShape::Scalar,
                    symmetry: TensorSymmetry::None,
                    frame: FrameSemantics::Scalar,
                },
                locality: crate::scientific::PropertyLocality::Pointwise,
                differentiability,
            },
            model: PropertyModel::Expression(expr),
            domain: PropertyDomain {
                physical_bounds: vec![],
                validity_bounds: vec![],
                phase_constraints: vec![],
                composition_constraints: vec![],
                assumptions: vec![],
                out_of_validity: crate::scientific::OutOfValidityPolicy::Warn,
            },
            evidence: PropertyEvidence {
                sources: vec![],
                dataset_digest: None,
                fit_digest: None,
                uncertainty: None,
                notes: Default::default(),
            },
        }
    }

    #[test]
    fn symbolic_property_gets_a_value_kernel_and_one_tangent_per_input() {
        let def = definition(DerivativeContract::Symbolic, "0.5 + 0.01 * T");
        let kernel = lower_property_kernel(&def, &UnitRegistry::si_bootstrap()).unwrap();
        assert_eq!(kernel.module.kernels.len(), 2);
        assert_eq!(kernel.tangents.len(), 1);
        assert_eq!(kernel.tangents[0].input, "T");
        malleus::validate(kernel.module.kernels[kernel.value_kernel].clone()).unwrap();
        malleus::validate(kernel.module.kernels[kernel.tangents[0].kernel].clone()).unwrap();
    }

    #[test]
    fn analytic_provided_property_gets_no_tangent_kernel() {
        let def = definition(DerivativeContract::AnalyticProvided, "0.5 + 0.01 * T");
        let kernel = lower_property_kernel(&def, &UnitRegistry::si_bootstrap()).unwrap();
        assert_eq!(kernel.module.kernels.len(), 1);
        assert!(kernel.tangents.is_empty());
    }

    #[test]
    fn table_and_external_models_are_refused_not_silently_skipped() {
        let mut def = definition(DerivativeContract::Symbolic, "0.5");
        def.model = PropertyModel::External(crate::scientific::PropertyProviderRef {
            provider: "vendor".into(),
            property: "k".into(),
        });
        let err = lower_property_kernel(&def, &UnitRegistry::si_bootstrap()).unwrap_err();
        assert!(matches!(err, PropertyKernelError::Unsupported(_)));
    }
}
