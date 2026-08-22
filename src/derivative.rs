//! Scientific objective, design, and derivative-request vocabulary.

use crate::id::{Digest, span_independent_digest};
use quantitas::Dimension;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const DERIVATIVE_REQUEST_SCHEMA: &str = "scientia-derivative-request/1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservableFunctional {
    pub name: String,
    pub semantic_expression: String,
    pub dimension: Dimension,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Objective {
    pub name: String,
    pub functional: ObservableFunctional,
    pub sense: ObjectiveSense,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveSense {
    Measure,
    Minimize,
    Maximize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Control {
    pub name: String,
    pub dimension: Dimension,
    pub support: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DesignVariable {
    pub name: String,
    pub dimension: Dimension,
    pub parameter_owner: String,
    pub admissible_set: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivativeProductSpec {
    Jvp,
    Vjp,
    Gradient,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivativeDependence {
    Partial,
    Total,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivativeLevel {
    Continuous,
    Discrete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarConvention {
    Real,
    ComplexWirtinger,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivativeStateConvention {
    FixedState,
    ConvergedState,
    AcceptedTrajectory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DifferentiabilityDisposition {
    Smooth,
    PiecewiseSmooth,
    TopologyEvent,
    Unsupported,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveSet {
    pub active: Vec<String>,
    pub frozen: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivativeConvention {
    pub dependence: DerivativeDependence,
    pub level: DerivativeLevel,
    pub scalar: ScalarConvention,
    pub state: DerivativeStateConvention,
    pub disposition: DifferentiabilityDisposition,
    pub event_or_refusal_basis: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeDerivativeConvention {
    pub geometry_revision: String,
    pub fixed_topology_stratum: String,
    pub boundary_selection: Vec<String>,
    pub include_normal_variation: bool,
    pub include_measure_variation: bool,
    pub disposition: DifferentiabilityDisposition,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivativeRequest {
    pub schema: String,
    pub parent_semantic_digest: String,
    pub objective: Objective,
    pub design_variables: Vec<DesignVariable>,
    pub controls: Vec<Control>,
    pub product: DerivativeProductSpec,
    pub active_set: ActiveSet,
    pub evaluation_state: String,
    pub convention: DerivativeConvention,
    pub shape: Option<ShapeDerivativeConvention>,
    pub identity: Digest,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivativeRefusal {
    pub code: String,
    pub message: String,
}

#[derive(Serialize)]
struct DerivativeIdentity<'a> {
    schema: &'a str,
    parent_semantic_digest: &'a str,
    objective: &'a Objective,
    design_variables: &'a [DesignVariable],
    controls: &'a [Control],
    product: DerivativeProductSpec,
    active_set: &'a ActiveSet,
    evaluation_state: &'a str,
    convention: &'a DerivativeConvention,
    shape: &'a Option<ShapeDerivativeConvention>,
}

impl DerivativeRequest {
    pub fn finish(mut self) -> Result<Self, DerivativeRefusal> {
        self.design_variables
            .sort_by(|left, right| left.name.cmp(&right.name));
        self.controls
            .sort_by(|left, right| left.name.cmp(&right.name));
        self.active_set.active.sort();
        self.active_set.frozen.sort();
        if let Some(shape) = &mut self.shape {
            shape.boundary_selection.sort();
        }
        self.validate_structure()?;
        self.identity = self.expected_identity();
        Ok(self)
    }

    /// Validate that the request says enough to interpret a derivative product.
    pub fn validate(&self) -> Result<(), DerivativeRefusal> {
        self.validate_structure()?;
        if self.identity != self.expected_identity() {
            return Err(refusal(
                "DERIVATIVE_IDENTITY_MISMATCH",
                "derivative request identity does not match its canonical contents",
            ));
        }
        Ok(())
    }

    fn validate_structure(&self) -> Result<(), DerivativeRefusal> {
        if self.schema != DERIVATIVE_REQUEST_SCHEMA {
            return Err(refusal(
                "DERIVATIVE_SCHEMA",
                "unsupported derivative-request schema",
            ));
        }
        if self.parent_semantic_digest.trim().is_empty()
            || self.objective.name.trim().is_empty()
            || self.objective.functional.name.trim().is_empty()
            || self
                .objective
                .functional
                .semantic_expression
                .trim()
                .is_empty()
        {
            return Err(refusal(
                "DERIVATIVE_INCOMPLETE_OBJECTIVE",
                "parent semantic identity and complete objective meaning are required",
            ));
        }
        if self.design_variables.iter().any(|variable| {
            variable.name.trim().is_empty()
                || variable.parameter_owner.trim().is_empty()
                || variable.admissible_set.trim().is_empty()
        }) || self
            .controls
            .iter()
            .any(|control| control.name.trim().is_empty() || control.support.trim().is_empty())
        {
            return Err(refusal(
                "DERIVATIVE_INCOMPLETE_INPUT_BINDING",
                "design variables and controls require names, owners/support, and admissible meaning",
            ));
        }
        if self.active_set.active.is_empty() || self.evaluation_state.trim().is_empty() {
            return Err(refusal(
                "DERIVATIVE_INCOMPLETE_REQUEST",
                "active inputs and an evaluation state are required",
            ));
        }
        let active = self.active_set.active.iter().collect::<BTreeSet<_>>();
        let frozen = self.active_set.frozen.iter().collect::<BTreeSet<_>>();
        if active.len() != self.active_set.active.len()
            || frozen.len() != self.active_set.frozen.len()
            || !active.is_disjoint(&frozen)
            || self
                .active_set
                .active
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .active_set
                .frozen
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .design_variables
                .windows(2)
                .any(|pair| pair[0].name >= pair[1].name)
            || self
                .controls
                .windows(2)
                .any(|pair| pair[0].name >= pair[1].name)
        {
            return Err(refusal(
                "DERIVATIVE_NONCANONICAL_INPUTS",
                "design, control, active, and frozen inputs must be canonical; active and frozen inputs must be disjoint",
            ));
        }
        let declared_designs = self
            .design_variables
            .iter()
            .map(|variable| variable.name.as_str())
            .collect::<BTreeSet<_>>();
        let declared_controls = self
            .controls
            .iter()
            .map(|control| control.name.as_str())
            .collect::<BTreeSet<_>>();
        if !declared_designs.is_disjoint(&declared_controls)
            || declared_designs.len() != self.design_variables.len()
            || declared_controls.len() != self.controls.len()
        {
            return Err(refusal(
                "DERIVATIVE_INPUT_NAMESPACE_COLLISION",
                "design-variable and control namespaces must be unique and disjoint",
            ));
        }
        let declared = declared_designs
            .union(&declared_controls)
            .copied()
            .collect::<BTreeSet<_>>();
        let partitioned = active
            .union(&frozen)
            .map(|name| name.as_str())
            .collect::<BTreeSet<_>>();
        if partitioned != declared {
            return Err(refusal(
                "DERIVATIVE_INPUT_PARTITION",
                "active and frozen inputs must exactly partition every declared design variable and control",
            ));
        }
        if matches!(
            self.convention.disposition,
            DifferentiabilityDisposition::TopologyEvent | DifferentiabilityDisposition::Unsupported
        ) && self
            .convention
            .event_or_refusal_basis
            .as_deref()
            .is_none_or(|basis| basis.trim().is_empty())
        {
            return Err(refusal(
                "DERIVATIVE_MISSING_DISPOSITION_BASIS",
                "event and unsupported dispositions require an explicit basis",
            ));
        }
        if let Some(shape) = &self.shape {
            if shape.geometry_revision.trim().is_empty()
                || shape.fixed_topology_stratum.trim().is_empty()
                || shape.boundary_selection.is_empty()
                || shape
                    .boundary_selection
                    .iter()
                    .any(|boundary| boundary.trim().is_empty())
            {
                return Err(refusal(
                    "SHAPE_DERIVATIVE_INCOMPLETE",
                    "shape derivatives require geometry revision, topology stratum, and boundary identity",
                ));
            }
            if shape
                .boundary_selection
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            {
                return Err(refusal(
                    "SHAPE_DERIVATIVE_NONCANONICAL_BOUNDARIES",
                    "shape boundary identities must be uniquely sorted",
                ));
            }
            if shape.disposition != self.convention.disposition {
                return Err(refusal(
                    "SHAPE_DERIVATIVE_DISPOSITION_MISMATCH",
                    "shape and derivative dispositions must agree",
                ));
            }
            if !self
                .active_set
                .active
                .iter()
                .any(|name| declared_designs.contains(name.as_str()))
            {
                return Err(refusal(
                    "SHAPE_DERIVATIVE_NO_ACTIVE_DESIGN",
                    "a shape derivative requires at least one active declared design variable",
                ));
            }
        }
        Ok(())
    }

    fn expected_identity(&self) -> Digest {
        span_independent_digest(&DerivativeIdentity {
            schema: &self.schema,
            parent_semantic_digest: &self.parent_semantic_digest,
            objective: &self.objective,
            design_variables: &self.design_variables,
            controls: &self.controls,
            product: self.product,
            active_set: &self.active_set,
            evaluation_state: &self.evaluation_state,
            convention: &self.convention,
            shape: &self.shape,
        })
    }
}

fn refusal(code: &str, message: &str) -> DerivativeRefusal {
    DerivativeRefusal {
        code: code.into(),
        message: message.into(),
    }
}
