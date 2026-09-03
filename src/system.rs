//! FC8 mixed/block operator systems assembled from the ordinary form-to-kernel artifact chain.

use crate::id::span_independent_digest;
use crate::{
    Digest, FormArgumentRole, FormRequirements, OperatorFactorization, StructuredOperatorKernels,
    SymbolId, TensorInputRole, compile_variational_form, derive_variational_form, factor_operator,
    infer_form_requirements, lower_operator_kernels,
};
use crate::{SemanticModule, VariationalForm};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

pub const OPERATOR_SYSTEM_SCHEMA: &str = "scientia-operator-system/1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct OperatorBlockCoordinate {
    pub row: SymbolId,
    pub column: SymbolId,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperatorSystemBlock {
    pub equation: String,
    pub row: SymbolId,
    pub columns: Vec<SymbolId>,
    pub coordinates: Vec<OperatorBlockCoordinate>,
    pub form: VariationalForm,
    pub requirements: FormRequirements,
    pub factorization: OperatorFactorization,
    pub kernels: StructuredOperatorKernels,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperatorSystem {
    pub schema: String,
    pub model: String,
    pub source_semantic_digest: Digest,
    pub artifact_digest: Digest,
    pub field_order: Vec<SymbolId>,
    pub blocks: Vec<OperatorSystemBlock>,
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum OperatorSystemError {
    #[error("SYSTEM_EMPTY: model `{model}` has no selected equations")]
    Empty { model: String },
    #[error("SYSTEM_DUPLICATE_EQUATION: equation `{equation}` was selected more than once")]
    DuplicateEquation { equation: String },
    #[error("SYSTEM_COMPILE: equation `{equation}` failed during {stage}: {detail}")]
    Compile {
        equation: String,
        stage: &'static str,
        detail: String,
    },
    #[error("SYSTEM_TEST_FIELD: equation `{equation}` has no generated test-space source")]
    MissingTestField { equation: String },
}

/// Compile several strong equations through the same FC2--FC5 chain and retain their explicit
/// block row/column identities. No block is inferred from display names: rows come from the
/// generated test-space receipt and columns from active typed QFunction bindings.
pub fn compile_operator_system(
    module: &SemanticModule,
    model_name: &str,
    equation_names: &[&str],
) -> Result<OperatorSystem, OperatorSystemError> {
    if equation_names.is_empty() {
        return Err(OperatorSystemError::Empty {
            model: model_name.to_owned(),
        });
    }
    let mut selected = BTreeSet::new();
    let mut blocks = Vec::with_capacity(equation_names.len());
    for equation in equation_names {
        if !selected.insert(*equation) {
            return Err(OperatorSystemError::DuplicateEquation {
                equation: (*equation).to_owned(),
            });
        }
        let form = derive_variational_form(module, model_name, equation).map_err(|error| {
            OperatorSystemError::Compile {
                equation: (*equation).to_owned(),
                stage: "form derivation",
                detail: error.to_string(),
            }
        })?;
        let row = form.receipt.test_space_source.ok_or_else(|| {
            OperatorSystemError::MissingTestField {
                equation: (*equation).to_owned(),
            }
        })?;
        blocks.push(compile_block(module, equation, row, form)?);
    }
    finish_system(model_name, blocks)
}

/// Compile authored forms into the same block artifact used for derived strong equations.
/// Each selected form must have exactly one typed test argument; its trial/unknown bindings
/// determine the active block columns.
pub fn compile_authored_operator_system(
    module: &SemanticModule,
    model_name: &str,
    form_names: &[&str],
) -> Result<OperatorSystem, OperatorSystemError> {
    if form_names.is_empty() {
        return Err(OperatorSystemError::Empty {
            model: model_name.to_owned(),
        });
    }
    let mut selected = BTreeSet::new();
    let mut blocks = Vec::with_capacity(form_names.len());
    for form_name in form_names {
        if !selected.insert(*form_name) {
            return Err(OperatorSystemError::DuplicateEquation {
                equation: (*form_name).to_owned(),
            });
        }
        let form = compile_variational_form(module, model_name, form_name).map_err(|error| {
            OperatorSystemError::Compile {
                equation: (*form_name).to_owned(),
                stage: "form compilation",
                detail: error.to_string(),
            }
        })?;
        let tests = form
            .arguments
            .iter()
            .filter(|argument| argument.role == FormArgumentRole::Test)
            .map(|argument| argument.symbol)
            .collect::<Vec<_>>();
        let [row] = tests.as_slice() else {
            return Err(OperatorSystemError::MissingTestField {
                equation: (*form_name).to_owned(),
            });
        };
        blocks.push(compile_block(module, form_name, *row, form)?);
    }
    finish_system(model_name, blocks)
}

fn compile_block(
    module: &SemanticModule,
    name: &str,
    row: SymbolId,
    form: VariationalForm,
) -> Result<OperatorSystemBlock, OperatorSystemError> {
    let requirements =
        infer_form_requirements(module, &form).map_err(|error| OperatorSystemError::Compile {
            equation: name.to_owned(),
            stage: "requirement inference",
            detail: error.to_string(),
        })?;
    let factorization =
        factor_operator(&form, &requirements).map_err(|error| OperatorSystemError::Compile {
            equation: name.to_owned(),
            stage: "operator factorization",
            detail: error.to_string(),
        })?;
    let kernels =
        lower_operator_kernels(&factorization).map_err(|error| OperatorSystemError::Compile {
            equation: name.to_owned(),
            stage: "structured-kernel lowering",
            detail: error.to_string(),
        })?;
    let columns = factorization
        .integrals
        .iter()
        .flat_map(|integral| &integral.primal.inputs)
        .filter(|input| input.role == TensorInputRole::Active)
        .map(|input| input.binding.symbol)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let coordinates = columns
        .iter()
        .copied()
        .map(|column| OperatorBlockCoordinate { row, column })
        .collect::<Vec<_>>();
    Ok(OperatorSystemBlock {
        equation: name.to_owned(),
        row,
        columns,
        coordinates,
        form,
        requirements,
        factorization,
        kernels,
    })
}

fn finish_system(
    model_name: &str,
    blocks: Vec<OperatorSystemBlock>,
) -> Result<OperatorSystem, OperatorSystemError> {
    let mut field_order = BTreeSet::new();
    for block in &blocks {
        field_order.insert(block.row);
        field_order.extend(block.columns.iter().copied());
    }
    let source_semantic_digest = blocks[0].form.source_semantic_digest.clone();
    let field_order = field_order.into_iter().collect::<Vec<_>>();
    let artifact_digest = span_independent_digest(&OperatorSystemDigestPayload {
        schema: OPERATOR_SYSTEM_SCHEMA,
        model: model_name,
        source_semantic_digest: &source_semantic_digest,
        field_order: &field_order,
        blocks: blocks
            .iter()
            .map(|block| OperatorSystemBlockDigest {
                equation: &block.equation,
                row: block.row,
                columns: &block.columns,
                form: &block.form.artifact_digest,
                requirements: &block.requirements.artifact_digest,
                factorization: &block.factorization.artifact_digest,
                kernels: &block.kernels.artifact_digest,
            })
            .collect(),
    });
    Ok(OperatorSystem {
        schema: OPERATOR_SYSTEM_SCHEMA.into(),
        model: model_name.to_owned(),
        source_semantic_digest,
        artifact_digest,
        field_order,
        blocks,
    })
}

/// The by-reference identity of one per-model block (SC-W1 §2.6): the same payload the
/// system digest folds, so a `scientia-operator-system/2` row can name the block without
/// embedding or re-numbering it.
pub fn block_digest(block: &OperatorSystemBlock) -> Digest {
    span_independent_digest(&OperatorSystemBlockDigest {
        equation: &block.equation,
        row: block.row,
        columns: &block.columns,
        form: &block.form.artifact_digest,
        requirements: &block.requirements.artifact_digest,
        factorization: &block.factorization.artifact_digest,
        kernels: &block.kernels.artifact_digest,
    })
}

#[derive(Serialize)]
struct OperatorSystemBlockDigest<'a> {
    equation: &'a str,
    row: SymbolId,
    columns: &'a [SymbolId],
    form: &'a Digest,
    requirements: &'a Digest,
    factorization: &'a Digest,
    kernels: &'a Digest,
}

#[derive(Serialize)]
struct OperatorSystemDigestPayload<'a> {
    schema: &'static str,
    model: &'a str,
    source_semantic_digest: &'a Digest,
    field_order: &'a [SymbolId],
    blocks: Vec<OperatorSystemBlockDigest<'a>>,
}
