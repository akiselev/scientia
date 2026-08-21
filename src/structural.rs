//! Structural equation analysis over the canonical scientific model.
//!
//! Incidence, alias analysis, DAE/index analysis, matching, SCC/BLT and tearing are projections
//! over [`ScientificModel`]. Scientia does not maintain a
//! second equation language for these passes.

pub mod dae;
pub mod scc;
pub mod schedule;

use crate::scientific::{Expr, FieldRole, ScientificModel, transitive_field_dependencies};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use thiserror::Error;

pub use dae::{
    AliasAnalysis, AliasClass, DerivativeVariable, DifferentiationStep, EquationDerivativeProfile,
    IndexReductionPlan, analyze_aliases, derivative_profile, pantelides_plan,
};
pub use schedule::{
    Block, BlockKind, Schedule, StructuralCompileError, compile_schedule,
    compile_schedule_without_tearing,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IncidenceSystem {
    pub variables: Vec<String>,
    pub rows: Vec<Vec<usize>>,
}
impl IncidenceSystem {
    /// Project the equations and unknown/state fields of a scientific model into an incidence
    /// matrix. Names that denote parameters, coefficients, or functions are intentionally not
    /// columns in the structural system.
    pub fn from_model(model: &ScientificModel) -> Result<Self, StructuralError> {
        let variables: Vec<_> = model
            .fields
            .iter()
            .filter(|field| matches!(field.role, FieldRole::State | FieldRole::Unknown))
            .map(|field| field.name.clone())
            .collect();
        let mut columns = BTreeMap::new();
        for (index, variable) in variables.iter().enumerate() {
            if columns.insert(variable.as_str(), index).is_some() {
                return Err(StructuralError::DuplicateVariable(variable.clone()));
            }
        }
        let mut rows = Vec::with_capacity(model.equations.len());
        for equation in &model.equations {
            let symbols = transitive_field_dependencies(model, [&equation.lhs, &equation.rhs]);
            rows.push(
                symbols
                    .into_iter()
                    .filter_map(|name| columns.get(name.as_str()).copied())
                    .collect::<Vec<_>>(),
            )
        }
        Ok(Self { variables, rows })
    }
    pub fn n_equations(&self) -> usize {
        self.rows.len()
    }
    pub fn n_variables(&self) -> usize {
        self.variables.len()
    }
}

pub(crate) fn collect_names<'a>(expr: &'a Expr, out: &mut BTreeSet<&'a str>) {
    match expr {
        Expr::Name { name, .. } => {
            out.insert(name);
        }
        Expr::Unary { arg, .. } => collect_names(arg, out),
        Expr::Binary { lhs, rhs, .. } => {
            collect_names(lhs, out);
            collect_names(rhs, out);
        }
        Expr::Call { args, .. } | Expr::Vector { elements: args, .. } => {
            for arg in args {
                collect_names(arg, out);
            }
        }
        Expr::Index { value, indices, .. } => {
            collect_names(value, out);
            for index in indices {
                collect_names(index, out);
            }
        }
        Expr::Number { .. } | Expr::String { .. } => {}
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Matching {
    pub equation_to_variable: Vec<Option<usize>>,
    pub variable_to_equation: Vec<Option<usize>>,
}
impl Matching {
    pub fn cardinality(&self) -> usize {
        self.equation_to_variable
            .iter()
            .filter(|v| v.is_some())
            .count()
    }
    pub fn is_perfect(&self) -> bool {
        self.equation_to_variable.len() == self.variable_to_equation.len()
            && self.cardinality() == self.equation_to_variable.len()
    }
    pub fn unmatched_equations(&self) -> Vec<usize> {
        self.equation_to_variable
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.is_none().then_some(i))
            .collect()
    }
    pub fn unmatched_variables(&self) -> Vec<usize> {
        self.variable_to_equation
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.is_none().then_some(i))
            .collect()
    }
}

/// Deterministic Hopcroft–Karp maximum matching.
pub fn maximum_matching(system: &IncidenceSystem) -> Matching {
    let n_u = system.n_equations();
    let n_v = system.n_variables();
    let mut pair_u = vec![None; n_u];
    let mut pair_v = vec![None; n_v];
    let mut dist = vec![usize::MAX; n_u];
    while bfs(&system.rows, &pair_u, &pair_v, &mut dist) {
        for u in 0..n_u {
            if pair_u[u].is_none() {
                dfs(u, &system.rows, &mut pair_u, &mut pair_v, &mut dist);
            }
        }
    }
    Matching {
        equation_to_variable: pair_u,
        variable_to_equation: pair_v,
    }
}
fn bfs(
    rows: &[Vec<usize>],
    pair_u: &[Option<usize>],
    pair_v: &[Option<usize>],
    dist: &mut [usize],
) -> bool {
    let mut queue = VecDeque::new();
    for u in 0..pair_u.len() {
        if pair_u[u].is_none() {
            dist[u] = 0;
            queue.push_back(u)
        } else {
            dist[u] = usize::MAX
        }
    }
    let mut nil = usize::MAX;
    while let Some(u) = queue.pop_front() {
        if dist[u] >= nil {
            continue;
        }
        for &v in &rows[u] {
            match pair_v[v] {
                None => nil = nil.min(dist[u] + 1),
                Some(next) if dist[next] == usize::MAX => {
                    dist[next] = dist[u] + 1;
                    queue.push_back(next)
                }
                Some(_) => {}
            }
        }
    }
    nil != usize::MAX
}
fn dfs(
    u: usize,
    rows: &[Vec<usize>],
    pair_u: &mut [Option<usize>],
    pair_v: &mut [Option<usize>],
    dist: &mut [usize],
) -> bool {
    let next_layer = dist[u].wrapping_add(1);
    for &v in &rows[u] {
        let advance = match pair_v[v] {
            None => true,
            Some(w) => dist[w] == next_layer && dfs(w, rows, pair_u, pair_v, dist),
        };
        if advance {
            pair_v[v] = Some(u);
            pair_u[u] = Some(v);
            return true;
        }
    }
    dist[u] = usize::MAX;
    false
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum StructuralError {
    #[error("model declares structural variable `{0}` more than once")]
    DuplicateVariable(String),
}
