use scientia::{
    BlockKind, IncidenceSystem, compile_schedule, compile_schedule_without_tearing,
    maximum_matching,
};

fn system(n_vars: usize, rows: &[&[usize]]) -> IncidenceSystem {
    IncidenceSystem {
        variables: (0..n_vars).map(|i| format!("x{i}")).collect(),
        rows: rows.iter().map(|r| r.to_vec()).collect(),
    }
}

fn brute_max(rows: &[Vec<usize>], n_vars: usize) -> usize {
    fn visit(row: usize, rows: &[Vec<usize>], used: &mut [bool], matched: usize, best: &mut usize) {
        if row == rows.len() {
            *best = (*best).max(matched);
            return;
        }
        visit(row + 1, rows, used, matched, best);
        for &v in &rows[row] {
            if v < used.len() && !used[v] {
                used[v] = true;
                visit(row + 1, rows, used, matched + 1, best);
                used[v] = false;
            }
        }
    }
    let mut best = 0;
    visit(0, rows, &mut vec![false; n_vars], 0, &mut best);
    best
}

#[test]
fn matching_agrees_with_exhaustive_reference_on_all_three_by_three_graphs() {
    // 2^(3*3) = 512 systems: an exhaustive independent oracle for this problem size.
    for mask in 0u16..512 {
        let mut rows = vec![vec![], vec![], vec![]];
        for (eq, row) in rows.iter_mut().enumerate() {
            for var in 0..3 {
                if mask & (1 << (eq * 3 + var)) != 0 {
                    row.push(var);
                }
            }
        }
        let sys = IncidenceSystem {
            variables: (0..3).map(|i| format!("x{i}")).collect(),
            rows,
        };
        assert_eq!(
            maximum_matching(&sys).cardinality(),
            brute_max(&sys.rows, 3)
        );
    }
}

#[test]
fn lower_triangular_chain_is_all_explicit() {
    let sys = system(3, &[&[0], &[0, 1], &[1, 2]]);
    let schedule = compile_schedule(&sys).unwrap();
    assert_eq!(schedule.blocks.len(), 3);
    assert!(
        schedule
            .blocks
            .iter()
            .all(|b| b.kind == BlockKind::Explicit)
    );
}

#[test]
fn two_by_two_loop_tears_one_variable_and_preserves_untorn_block() {
    let sys = system(2, &[&[0, 1], &[0, 1]]);
    let torn = compile_schedule(&sys).unwrap();
    let untorn = compile_schedule_without_tearing(&sys).unwrap();
    assert_eq!(torn.blocks.len(), 1);
    assert_eq!(untorn.blocks.len(), 1);
    assert_eq!(torn.blocks[0].kind, BlockKind::Torn);
    assert_eq!(untorn.blocks[0].kind, BlockKind::AlgebraicLoop);
    assert_eq!(torn.blocks[0].tearing_vars.len(), 1);
    assert_eq!(torn.blocks[0].equations.len(), 2);
}

#[test]
fn structurally_singular_system_reports_both_sides() {
    let sys = system(2, &[&[0], &[0]]);
    let matching = maximum_matching(&sys);
    assert_eq!(matching.cardinality(), 1);
    assert_eq!(matching.unmatched_equations().len(), 1);
    assert_eq!(matching.unmatched_variables().len(), 1);
    assert!(compile_schedule(&sys).is_err());
}

#[test]
fn structural_results_are_deterministic() {
    let sys = system(4, &[&[0, 1], &[0, 1], &[1, 2, 3], &[2, 3]]);
    let first = compile_schedule(&sys).unwrap();
    for _ in 0..32 {
        assert_eq!(compile_schedule(&sys).unwrap(), first);
    }
}
