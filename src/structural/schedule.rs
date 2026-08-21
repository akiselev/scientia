//! Causalization pipeline over [`IncidenceSystem`]: maximum matching,
//! equation dependency graph, SCC/BLT decomposition and deterministic greedy tearing.
//!
//! This is intentionally a projection of Scientia's canonical scientific model; it does not
//! introduce a second equation AST.

use super::scc::{Digraph, tarjan_scc};
use super::{IncidenceSystem, Matching, maximum_matching};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
struct SccBlock {
    equations: Vec<usize>,
    variables: Vec<usize>,
}

impl SccBlock {
    fn size(&self) -> usize {
        self.equations.len()
    }
}

fn dependency_digraph(system: &IncidenceSystem, matching: &Matching) -> Digraph {
    let mut graph = Digraph::new(system.n_equations());
    for equation in 0..system.n_equations() {
        let own = matching.equation_to_variable[equation];
        for &variable in &system.rows[equation] {
            if Some(variable) == own {
                continue;
            }
            if let Some(solver) = matching.variable_to_equation[variable]
                && solver != equation
            {
                // Endpoints are guaranteed in range by the matching dimensions.
                let _ = graph.add_edge(equation, solver);
            }
        }
    }
    graph.normalize();
    graph
}

fn blt_order(system: &IncidenceSystem, matching: &Matching) -> Vec<SccBlock> {
    let graph = dependency_digraph(system, matching);
    tarjan_scc(&graph)
        .components()
        .iter()
        .map(|component| {
            let equations = component.clone();
            let mut variables: Vec<_> = equations
                .iter()
                .filter_map(|&equation| matching.equation_to_variable[equation])
                .collect();
            variables.sort_unstable();
            SccBlock {
                equations,
                variables,
            }
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TornBlock {
    tearing_vars: Vec<usize>,
    assignment_eqs: Vec<usize>,
    assignment_vars: Vec<usize>,
    residual_eqs: Vec<usize>,
}

/// Deterministic Cellier/Carpanzano-style greedy tearing. When causalization stalls, choose
/// the still-unknown block variable with maximum incidence in unused equations; smallest
/// variable index wins ties.
fn tear_block(system: &IncidenceSystem, block: &SccBlock) -> TornBlock {
    let n_vars = system.n_variables();
    let equations = &block.equations;
    let mut is_block_variable = vec![false; n_vars];
    for &variable in &block.variables {
        is_block_variable[variable] = true;
    }
    let restricted: Vec<Vec<usize>> = equations
        .iter()
        .map(|&equation| {
            system.rows[equation]
                .iter()
                .copied()
                .filter(|&variable| is_block_variable[variable])
                .collect()
        })
        .collect();

    let mut known = vec![false; n_vars];
    let mut equation_used = vec![false; equations.len()];
    let mut remaining = block.variables.len();
    let mut tearing_vars = Vec::new();
    let mut assignment_eqs = Vec::new();
    let mut assignment_vars = Vec::new();

    while remaining > 0 {
        let mut progressed = true;
        while progressed {
            progressed = false;
            for local_equation in 0..equations.len() {
                if equation_used[local_equation] {
                    continue;
                }
                let mut sole_unknown = None;
                let mut multiple = false;
                for &variable in &restricted[local_equation] {
                    if known[variable] {
                        continue;
                    }
                    if sole_unknown.replace(variable).is_some() {
                        multiple = true;
                        break;
                    }
                }
                if !multiple && let Some(variable) = sole_unknown {
                    equation_used[local_equation] = true;
                    known[variable] = true;
                    remaining -= 1;
                    assignment_eqs.push(equations[local_equation]);
                    assignment_vars.push(variable);
                    progressed = true;
                }
            }
        }
        if remaining == 0 {
            break;
        }

        let mut best = None::<(usize, usize)>;
        for &variable in &block.variables {
            if known[variable] {
                continue;
            }
            let incidence = restricted
                .iter()
                .enumerate()
                .filter(|(i, row)| !equation_used[*i] && row.contains(&variable))
                .count();
            match best {
                None => best = Some((variable, incidence)),
                Some((best_variable, best_incidence))
                    if incidence > best_incidence
                        || (incidence == best_incidence && variable < best_variable) =>
                {
                    best = Some((variable, incidence));
                }
                Some(_) => {}
            }
        }

        // `remaining > 0` guarantees at least one unknown block variable.
        if let Some((variable, _)) = best {
            known[variable] = true;
            remaining -= 1;
            tearing_vars.push(variable);
        } else {
            break;
        }
    }

    let residual_eqs = equations
        .iter()
        .enumerate()
        .filter_map(|(i, &equation)| (!equation_used[i]).then_some(equation))
        .collect();

    TornBlock {
        tearing_vars,
        assignment_eqs,
        assignment_vars,
        residual_eqs,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockKind {
    Explicit,
    AlgebraicLoop,
    Torn,
}

/// One step of the structural solve schedule. For a torn block, assignment equations come
/// first and align with `solved_vars`; residual equations follow and close the iteration on
/// `tearing_vars`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Block {
    pub kind: BlockKind,
    pub equations: Vec<usize>,
    pub solved_vars: Vec<usize>,
    pub tearing_vars: Vec<usize>,
}

impl Block {
    pub fn all_variables(&self) -> Vec<usize> {
        let mut variables = self.solved_vars.clone();
        variables.extend(&self.tearing_vars);
        variables.sort_unstable();
        variables
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Schedule {
    pub blocks: Vec<Block>,
    n_equations: usize,
    n_variables: usize,
}

impl Schedule {
    pub fn n_equations(&self) -> usize {
        self.n_equations
    }

    pub fn n_variables(&self) -> usize {
        self.n_variables
    }
}

pub fn compile_schedule(system: &IncidenceSystem) -> Result<Schedule, StructuralCompileError> {
    compile_inner(system, true)
}

pub fn compile_schedule_without_tearing(
    system: &IncidenceSystem,
) -> Result<Schedule, StructuralCompileError> {
    compile_inner(system, false)
}

fn compile_inner(
    system: &IncidenceSystem,
    tearing: bool,
) -> Result<Schedule, StructuralCompileError> {
    if system.n_equations() != system.n_variables() {
        return Err(StructuralCompileError::NotSquare {
            n_equations: system.n_equations(),
            n_variables: system.n_variables(),
        });
    }

    let matching = maximum_matching(system);
    if !matching.is_perfect() {
        return Err(StructuralCompileError::StructurallySingular {
            unmatched_equations: matching.unmatched_equations(),
            unmatched_variables: matching.unmatched_variables(),
        });
    }

    let blocks = blt_order(system, &matching)
        .iter()
        .map(|block| lower_block(system, block, tearing))
        .collect();

    Ok(Schedule {
        blocks,
        n_equations: system.n_equations(),
        n_variables: system.n_variables(),
    })
}

fn lower_block(system: &IncidenceSystem, block: &SccBlock, tearing: bool) -> Block {
    if block.size() == 1 {
        return Block {
            kind: BlockKind::Explicit,
            equations: block.equations.clone(),
            solved_vars: block.variables.clone(),
            tearing_vars: Vec::new(),
        };
    }

    if !tearing {
        return Block {
            kind: BlockKind::AlgebraicLoop,
            equations: block.equations.clone(),
            solved_vars: block.variables.clone(),
            tearing_vars: Vec::new(),
        };
    }

    let torn = tear_block(system, block);
    let mut equations = torn.assignment_eqs;
    equations.extend(torn.residual_eqs);
    Block {
        kind: BlockKind::Torn,
        equations,
        solved_vars: torn.assignment_vars,
        tearing_vars: torn.tearing_vars,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StructuralCompileError {
    NotSquare {
        n_equations: usize,
        n_variables: usize,
    },
    StructurallySingular {
        unmatched_equations: Vec<usize>,
        unmatched_variables: Vec<usize>,
    },
}

impl fmt::Display for StructuralCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotSquare {
                n_equations,
                n_variables,
            } => write!(
                f,
                "system is not square: {n_equations} equation(s) vs {n_variables} variable(s)"
            ),
            Self::StructurallySingular {
                unmatched_equations,
                unmatched_variables,
            } => write!(
                f,
                "structurally singular: {} unmatched equation(s) {unmatched_equations:?}, {} unmatched variable(s) {unmatched_variables:?}",
                unmatched_equations.len(),
                unmatched_variables.len()
            ),
        }
    }
}

impl std::error::Error for StructuralCompileError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn incidence(n_variables: usize, rows: &[&[usize]]) -> IncidenceSystem {
        IncidenceSystem {
            variables: (0..n_variables).map(|i| format!("x{i}")).collect(),
            rows: rows.iter().map(|row| row.to_vec()).collect(),
        }
    }

    #[test]
    fn lower_triangular_chain_is_explicit_in_order() {
        let system = incidence(3, &[&[0], &[0, 1], &[1, 2]]);
        let schedule = compile_schedule(&system).unwrap();
        assert_eq!(schedule.blocks.len(), 3);
        assert!(
            schedule
                .blocks
                .iter()
                .all(|b| b.kind == BlockKind::Explicit)
        );
        assert_eq!(schedule.blocks[0].solved_vars, vec![0]);
        assert_eq!(schedule.blocks[1].solved_vars, vec![1]);
        assert_eq!(schedule.blocks[2].solved_vars, vec![2]);
    }

    #[test]
    fn dense_loop_is_torn_deterministically() {
        let system = incidence(3, &[&[0, 1, 2], &[0, 1, 2], &[0, 1, 2]]);
        let schedule = compile_schedule(&system).unwrap();
        assert_eq!(schedule.blocks.len(), 1);
        let block = &schedule.blocks[0];
        assert_eq!(block.kind, BlockKind::Torn);
        assert_eq!(block.tearing_vars, vec![0, 1]);
        assert_eq!(block.solved_vars.len(), 1);
        assert_eq!(block.all_variables(), vec![0, 1, 2]);
        assert_eq!(block.equations.len(), 3);
    }

    #[test]
    fn coupled_solver_can_keep_raw_algebraic_loop() {
        let system = incidence(2, &[&[0, 1], &[0, 1]]);
        let schedule = compile_schedule_without_tearing(&system).unwrap();
        assert_eq!(schedule.blocks[0].kind, BlockKind::AlgebraicLoop);
        assert_eq!(schedule.blocks[0].solved_vars, vec![0, 1]);
    }

    #[test]
    fn rlc_shape_has_explicit_loop_explicit_blocks() {
        let system = incidence(4, &[&[0], &[0, 1, 2], &[1, 2], &[2, 3]]);
        let schedule = compile_schedule(&system).unwrap();
        assert_eq!(schedule.blocks.len(), 3);
        assert_eq!(schedule.blocks[0].kind, BlockKind::Explicit);
        assert_eq!(schedule.blocks[1].kind, BlockKind::Torn);
        assert_eq!(schedule.blocks[1].all_variables(), vec![1, 2]);
        assert_eq!(schedule.blocks[2].kind, BlockKind::Explicit);
    }

    #[test]
    fn structural_singularity_reports_unmatched_rows_and_columns() {
        let system = incidence(2, &[&[0], &[0]]);
        let error = compile_schedule(&system).unwrap_err();
        match error {
            StructuralCompileError::StructurallySingular {
                unmatched_equations,
                unmatched_variables,
            } => {
                assert_eq!(unmatched_equations.len(), 1);
                assert_eq!(unmatched_variables, vec![1]);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
