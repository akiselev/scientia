//! SV2-C1a: conservation-law semantics, generic and authored numerical fluxes lowered to
//! Malleus facet-pair kernels, claim checks and the FV nonsmooth-derivative refusal (D6).
use malleus::prepared::{PreparedKernel, ReadBinding, WriteBinding};
use malleus::{DerivativeMode, DerivativeRequest, Executable, OperandId, validate};
use quantitas::UnitRegistry;
use scientia::{
    BoundaryFluxSemantics, ConservationLawError, ConservationLawProgram, ConservationLawSystem,
    ConservationRelation, FluxClaim, FvFluxSample, FvKernel, FvKernelRole, FvMethodIntent,
    FvOperandMeaning, GradientScheme, LimiterFamily, NumericalFluxIntent, PositivityIntent,
    ReconstructionIntent, ReconstructionVariables, SemanticModule, SourceTreatment,
    VerificationObligationKind, compile_conservation_law_system, compile_semantics,
    lower_conservation_law, run_point,
};
use std::collections::BTreeMap;

const FIXTURE: &str = include_str!("fixtures/scalar-conservation.res");

fn module() -> SemanticModule {
    compile_semantics(FIXTURE, &UnitRegistry::si_bootstrap())
        .unwrap_or_else(|diagnostics| panic!("{diagnostics:#?}"))
        .semantic
}

fn system(module: &SemanticModule, model: &str) -> ConservationLawSystem {
    compile_conservation_law_system(module, model, "balance", "u").unwrap()
}

fn intent(numerical_flux: NumericalFluxIntent) -> FvMethodIntent {
    FvMethodIntent {
        numerical_flux,
        reconstruction: ReconstructionIntent::Constant,
        source: SourceTreatment::Pointwise,
        positivity: PositivityIntent::None,
    }
}

fn lower(
    module: &SemanticModule,
    model: &str,
    flux: NumericalFluxIntent,
    samples: &[FvFluxSample],
) -> Result<ConservationLawProgram, ConservationLawError> {
    lower_conservation_law(module, &system(module, model), &intent(flux), samples)
}

/// Deterministic samples in [-2, 2) with exactly representable signed zero and ties included.
fn states(count: usize) -> Vec<(f64, f64)> {
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 * 4.0 - 2.0
    };
    let mut out = vec![(0.0, -0.0), (1.5, 1.5), (-1.0, 1.0), (0.75, -0.25)];
    while out.len() < count {
        out.push((next(), next()));
    }
    out
}

fn burgers_samples() -> Vec<FvFluxSample> {
    states(64)
        .into_iter()
        .enumerate()
        .map(|(index, (minus, plus))| FvFluxSample {
            minus,
            plus,
            normal: vec![if index % 2 == 0 { 1.0 } else { -1.0 }],
            data: BTreeMap::from([("k".into(), 0.3)]),
        })
        .collect()
}

fn advection_2d_samples() -> Vec<FvFluxSample> {
    states(64)
        .into_iter()
        .enumerate()
        .map(|(index, (minus, plus))| {
            let angle = index as f64 * 0.37;
            FvFluxSample {
                minus,
                plus,
                normal: vec![angle.cos(), angle.sin()],
                data: BTreeMap::from([("ax".into(), 1.0), ("ay".into(), -0.5)]),
            }
        })
        .collect()
}

/// Bind one sample to a kernel's operands by meaning and run the reference interpreter.
fn eval(program: &ConservationLawProgram, kernel: &FvKernel, sample: &FvFluxSample) -> f64 {
    let mut buffers = kernel
        .operands
        .iter()
        .map(|meaning| {
            vec![match meaning {
                FvOperandMeaning::State { side: None } => sample.minus,
                FvOperandMeaning::State {
                    side: Some(malleus::FacetSide::Minus),
                } => sample.minus,
                FvOperandMeaning::State {
                    side: Some(malleus::FacetSide::Plus),
                } => sample.plus,
                FvOperandMeaning::Normal { axis } => sample.normal[usize::from(*axis)],
                FvOperandMeaning::Datum { datum } => sample.data[&program.data[*datum].name],
                FvOperandMeaning::SlopeRatio => sample.minus,
                FvOperandMeaning::Output => 0.0,
            }]
        })
        .collect::<Vec<_>>();
    run_point(kernel, &mut buffers).unwrap()
}

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

#[test]
fn law_is_read_from_res_with_boundary_closures() {
    let module = module();
    let law = system(&module, "LinearAdvection1D");
    assert_eq!(law.schema, "scientia-conservation-law/1");
    assert_eq!(law.spatial_dimension, 1);
    assert_eq!(law.flux_sign, 1);
    assert!(law.source_terms.is_empty());
    assert!(matches!(
        &law.boundaries[..],
        [
            BoundaryFluxSemantics::GhostState { region_name: left, .. },
            BoundaryFluxSemantics::PrescribedFlux { region_name: right, .. },
        ] if left == "left" && right == "right"
    ));
    law.check_boundary_closure(&["left", "right"], &[]).unwrap();
    for (exterior, periodic) in [
        (&["left", "right", "top"][..], &[][..]),
        (&["left"][..], &[][..]),
        (&["left", "right"][..], &["left"][..]),
    ] {
        assert_eq!(
            law.check_boundary_closure(exterior, periodic)
                .unwrap_err()
                .code,
            "FV_BOUNDARY_UNCLOSED"
        );
    }
    let periodic = system(&module, "LinearAdvection2D");
    assert!(periodic.boundaries.is_empty());
    periodic
        .check_boundary_closure(&["x0", "x1"], &["x0", "x1"])
        .unwrap();

    let burgers = system(&module, "Burgers1D");
    assert_eq!(burgers.source_terms.len(), 1);
    // `= -k * u` parses as `(-k) * u`: one right-hand-side term, moved over unchanged.
    assert_eq!(burgers.source_terms[0].sign, 1);
    assert_eq!(
        system(&module, "Burgers1D").artifact_digest,
        burgers.artifact_digest
    );
}

#[test]
fn lowered_llf_equals_the_hand_formula_bitwise() {
    let module = module();
    // Exact basis: the hand formulas below perform the same IEEE operations in the same order
    // as the lowering (F·n = Σ (f_i)·n_i left to right; the Malleus tangent of a product with a
    // unit direction multiplies by 1.0, which is exact), so equality is bitwise.
    let burgers_samples = burgers_samples();
    let burgers = lower(
        &module,
        "Burgers1D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        &burgers_samples,
    )
    .unwrap();
    for sample in &burgers_samples {
        let (um, up, n) = (sample.minus, sample.plus, sample.normal[0]);
        let fm = ((0.5 * um) * um) * n;
        let fp = ((0.5 * up) * up) * n;
        let lambda = (um * n).abs().max((up * n).abs());
        let expected = 0.5 * (fm + fp) - (0.5 * lambda) * (up - um);
        let observed = eval(&burgers, &burgers.face_flux, sample);
        assert!(
            same(observed, expected),
            "{sample:?}: {observed} vs {expected}"
        );
        assert!(same(
            eval(&burgers, &burgers.physical_flux_normal, sample),
            fm
        ));
        assert!(same(
            eval(&burgers, &burgers.characteristic_speed, sample),
            (um * n).abs()
        ));
    }

    let advection_samples = advection_2d_samples();
    let advection = lower(
        &module,
        "LinearAdvection2D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        &advection_samples,
    )
    .unwrap();
    for sample in &advection_samples {
        let (um, up) = (sample.minus, sample.plus);
        let (n0, n1) = (sample.normal[0], sample.normal[1]);
        let (ax, ay) = (sample.data["ax"], sample.data["ay"]);
        let fm = (ax * um) * n0 + (ay * um) * n1;
        let fp = (ax * up) * n0 + (ay * up) * n1;
        let speed = (ax * n0 + ay * n1).abs();
        let expected = 0.5 * (fm + fp) - (0.5 * speed.max(speed)) * (up - um);
        assert!(same(
            eval(&advection, &advection.face_flux, sample),
            expected
        ));
    }
    assert_eq!(
        advection
            .data
            .iter()
            .map(|datum| datum.name.as_str())
            .collect::<Vec<_>>(),
        ["ax", "ay"]
    );
}

#[test]
fn hll_and_central_match_their_hand_formulas() {
    let module = module();
    let samples = burgers_samples();
    let hll = lower(&module, "Burgers1D", NumericalFluxIntent::Hll, &samples).unwrap();
    let central = lower(&module, "Burgers1D", NumericalFluxIntent::Central, &samples).unwrap();
    for sample in &samples {
        let (um, up, n) = (sample.minus, sample.plus, sample.normal[0]);
        let fm = ((0.5 * um) * um) * n;
        let fp = ((0.5 * up) * up) * n;
        let (sm, sp) = (um * n, up * n);
        let (smin, smax) = (sm.min(sp), sm.max(sp));
        let expected = if smin >= 0.0 {
            fm
        } else if smax <= 0.0 {
            fp
        } else {
            ((smax * fm - smin * fp) + (smin * smax) * (up - um)) / (smax - smin)
        };
        assert!(
            same(eval(&hll, &hll.face_flux, sample), expected),
            "{sample:?}"
        );
        assert!(same(
            eval(&central, &central.face_flux, sample),
            0.5 * (fm + fp)
        ));
    }
}

#[test]
fn generic_fluxes_pass_swap_symmetry_exactly() {
    let module = module();
    for (model, samples) in [
        ("Burgers1D", burgers_samples()),
        ("LinearAdvection2D", advection_2d_samples()),
    ] {
        for flux in [
            NumericalFluxIntent::Central,
            NumericalFluxIntent::LocalLaxFriedrichs,
            NumericalFluxIntent::Hll,
        ] {
            let program = lower(&module, model, flux.clone(), &samples).unwrap();
            assert_eq!(program.checks.len(), 2, "{model} {flux:?}");
            for check in &program.checks {
                assert_eq!(check.samples, samples.len());
                assert_eq!(check.tolerance, 0.0);
                assert_eq!(
                    check.max_absolute, 0.0,
                    "{model} {flux:?} {:?}",
                    check.claim
                );
            }
            let pair = program.face_flux.facet_pair().unwrap();
            malleus::validate_facet_pair(pair).unwrap();
            program.validate_identity().unwrap();
        }
    }
}

#[test]
fn authored_fluxes_are_checked_and_refused_with_typed_codes() {
    let module = module();
    let samples = advection_2d_samples();
    let authored = |declaration: &str| NumericalFluxIntent::Authored {
        declaration: declaration.into(),
        tolerance: 0.0,
    };
    let rusanov = lower(&module, "AuthoredBurgers2D", authored("rusanov"), &samples).unwrap();
    assert!(rusanov.checks.iter().all(|check| check.max_absolute == 0.0));
    assert_eq!(
        rusanov
            .checks
            .iter()
            .map(|check| check.claim)
            .collect::<Vec<_>>(),
        [FluxClaim::Consistency, FluxClaim::Antisymmetry]
    );
    for (declaration, code) in [
        ("one_sided", "FV_AUTHORED_FLUX_NOT_ANTISYMMETRIC"),
        ("doubled", "FV_AUTHORED_FLUX_INCONSISTENT"),
        ("untraced", "FV_AUTHORED_FLUX_UNTRACED"),
        ("missing", "FV_AUTHORED_FLUX_MISSING"),
    ] {
        let error = lower(
            &module,
            "AuthoredBurgers2D",
            authored(declaration),
            &samples,
        )
        .unwrap_err();
        assert_eq!(error.code, code, "{declaration}: {error}");
    }
    assert_eq!(
        lower(&module, "AuthoredBurgers2D", authored("rusanov"), &[])
            .unwrap_err()
            .code,
        "FV_AUTHORED_FLUX_UNCHECKED"
    );
}

fn jvp(kernel: &FvKernel, independent: FvOperandMeaning) -> DerivativeRequest {
    DerivativeRequest {
        mode: DerivativeMode::Jvp,
        independent_operands: vec![kernel.operand(&independent).unwrap()],
        dependent_operands: vec![kernel.operand(&FvOperandMeaning::Output).unwrap()],
    }
}

#[test]
fn derivatives_through_flux_kinks_are_refused_and_smooth_ones_kept() {
    let module = module();
    let minus = FvOperandMeaning::State {
        side: Some(malleus::FacetSide::Minus),
    };
    let refused = |kernel: &FvKernel, operand| {
        let error = kernel.differentiate(&jvp(kernel, operand)).unwrap_err();
        assert_eq!(error.code, "FV_NONSMOOTH_DERIVATIVE_REFUSED", "{error}");
    };
    let burgers_llf = lower(
        &module,
        "Burgers1D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        &[],
    )
    .unwrap();
    refused(&burgers_llf.face_flux, minus.clone());
    refused(
        &burgers_llf.characteristic_speed,
        FvOperandMeaning::State { side: None },
    );
    let burgers_hll = lower(&module, "Burgers1D", NumericalFluxIntent::Hll, &[]).unwrap();
    refused(&burgers_hll.face_flux, minus.clone());

    // A smooth flux keeps its derivative: d/du⁻ ½(F(u⁻)+F(u⁺))·n = ½ u⁻ n for Burgers.
    let central = lower(&module, "Burgers1D", NumericalFluxIntent::Central, &[]).unwrap();
    assert!(central.face_flux.nonsmooth.is_empty());
    let product = central
        .face_flux
        .differentiate(&jvp(&central.face_flux, minus.clone()))
        .unwrap();
    let mut buffers = vec![vec![0.0]; product.kernel.operands.len()];
    let values = BTreeMap::from([
        (central.face_flux.operand(&minus).unwrap(), 1.25),
        (
            central
                .face_flux
                .operand(&FvOperandMeaning::State {
                    side: Some(malleus::FacetSide::Plus),
                })
                .unwrap(),
            -0.5,
        ),
        (
            central
                .face_flux
                .operand(&FvOperandMeaning::Normal { axis: 0 })
                .unwrap(),
            1.0,
        ),
    ]);
    for pair in &product.primal_operands {
        if let Some(value) = values.get(&pair.primal) {
            buffers[pair.derivative.index()][0] = *value;
        }
    }
    buffers[product.independent_operands[0].derivative.index()][0] = 1.0;
    let executable = Executable::reference(validate(product.kernel.clone()).unwrap());
    let mut bindings = buffers
        .iter_mut()
        .enumerate()
        .map(|(index, values)| malleus::BufferBinding::new(OperandId::new(index), values))
        .collect::<Vec<_>>();
    malleus::Interpreter::run(&executable, &mut bindings).unwrap();
    assert_eq!(
        buffers[product.dependent_operands[0].derivative.index()][0],
        0.5 * 1.25
    );

    // Linear advection: the LLF speed |a·n| does not depend on the state, so the state
    // direction crosses no kink, but the datum direction does.
    let advection = lower(
        &module,
        "LinearAdvection2D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        &[],
    )
    .unwrap();
    assert!(!advection.face_flux.nonsmooth.is_empty());
    advection
        .face_flux
        .differentiate(&jvp(&advection.face_flux, minus))
        .unwrap();
    refused(&advection.face_flux, FvOperandMeaning::Datum { datum: 0 });
}

#[test]
fn boundary_source_and_limiter_kernels_lower() {
    let module = module();
    let program = lower(
        &module,
        "LinearAdvection1D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        &[],
    )
    .unwrap();
    let sample = FvFluxSample {
        minus: 0.7,
        plus: 0.0,
        normal: vec![1.0],
        data: BTreeMap::from([("a".into(), 2.5)]),
    };
    let [ghost, outflow] = &program.boundaries[..] else {
        panic!("two boundaries");
    };
    assert_eq!(ghost.role, FvKernelRole::Ghost);
    assert_eq!(eval(&program, ghost, &sample), 1.0);
    assert_eq!(outflow.role, FvKernelRole::PrescribedBoundaryFlux);
    assert_eq!(eval(&program, outflow, &sample), 2.5 * 0.7);
    assert!(program.source.is_none());

    let burgers = lower(&module, "Burgers1D", NumericalFluxIntent::Central, &[]).unwrap();
    let source = burgers.source.as_ref().unwrap();
    let sample = FvFluxSample {
        minus: 1.1,
        plus: 0.0,
        normal: vec![1.0],
        data: BTreeMap::from([("k".into(), 0.3)]),
    };
    assert!(same(eval(&burgers, source, &sample), -(0.3 * 1.1)));

    let muscl = |limiter| FvMethodIntent {
        reconstruction: ReconstructionIntent::Muscl {
            variables: ReconstructionVariables::Conserved,
            gradient: GradientScheme::Axis1D,
            limiter,
        },
        ..intent(NumericalFluxIntent::LocalLaxFriedrichs)
    };
    let law = system(&module, "LinearAdvection1D");
    type Phi = fn(f64) -> f64;
    let limiters: [(LimiterFamily, Phi); 4] = [
        (LimiterFamily::Minmod, |r| 0.0f64.max(1.0f64.min(r))),
        (LimiterFamily::VanLeer, |r| (r + r.abs()) / (1.0 + r.abs())),
        (LimiterFamily::MonotonizedCentral, |r| {
            0.0f64.max((2.0 * r).min(0.5 * (1.0 + r)).min(2.0))
        }),
        (LimiterFamily::Superbee, |r| {
            0.0f64.max((2.0 * r).min(1.0).max(r.min(2.0)))
        }),
    ];
    for (family, phi) in limiters {
        let program = lower_conservation_law(&module, &law, &muscl(family.clone()), &[]).unwrap();
        let limiter = program.limiter.as_ref().unwrap();
        for r in [-1.0, 0.0, 0.25, 0.5, 1.0, 1.5, 3.0] {
            let sample = FvFluxSample {
                minus: r,
                plus: 0.0,
                normal: vec![1.0],
                data: BTreeMap::new(),
            };
            assert!(
                same(eval(&program, limiter, &sample), phi(r)),
                "{family:?} {r}"
            );
        }
        let error = limiter
            .differentiate(&jvp(limiter, FvOperandMeaning::SlopeRatio))
            .unwrap_err();
        assert_eq!(error.code, "FV_NONSMOOTH_DERIVATIVE_REFUSED");
        assert!(program.obligations.iter().any(|obligation| matches!(
            obligation.kind,
            VerificationObligationKind::CflBound { forward_euler_limit, .. } if forward_euler_limit == 0.5
        )));
    }
    for (intent, code) in [
        (
            muscl(LimiterFamily::BarthJespersen),
            "FV_LIMITER_UNSUPPORTED",
        ),
        (
            FvMethodIntent {
                reconstruction: ReconstructionIntent::Muscl {
                    variables: ReconstructionVariables::Primitive,
                    gradient: GradientScheme::LeastSquares,
                    limiter: LimiterFamily::VanLeer,
                },
                ..intent(NumericalFluxIntent::Hll)
            },
            "FV_PRIMITIVE_MAP_MISSING",
        ),
        (
            FvMethodIntent {
                positivity: PositivityIntent::ZhangShuScaling {
                    thresholds: vec![1e-12],
                },
                ..intent(NumericalFluxIntent::Hll)
            },
            "FV_POSITIVITY_UNSUPPORTED",
        ),
    ] {
        assert_eq!(
            lower_conservation_law(&module, &law, &intent, &[])
                .unwrap_err()
                .code,
            code
        );
    }
}

#[test]
fn unsupported_laws_refuse_with_typed_codes() {
    let module = module();
    let law = system(&module, "Diffusive1D");
    let error = lower_conservation_law(
        &module,
        &law,
        &intent(NumericalFluxIntent::LocalLaxFriedrichs),
        &[],
    )
    .unwrap_err();
    assert_eq!(error.code, "FV_DIFFUSIVE_FLUX_UNSUPPORTED", "{error}");
    assert_eq!(
        compile_conservation_law_system(&module, "Burgers1D", "balance", "k")
            .unwrap_err()
            .code,
        "FV_METHOD_SELECTION"
    );
}

#[test]
fn face_flux_runs_batched_over_a_face_block_with_uniform_data() {
    let module = module();
    let program = lower(&module, "LinearAdvection2D", NumericalFluxIntent::Hll, &[]).unwrap();
    let samples = advection_2d_samples();
    let kernel = &program.face_flux;
    let executable = Executable::reference(validate(kernel.kernel.clone()).unwrap());
    let prepared = PreparedKernel::new(&executable).unwrap();
    let mut state = prepared.state();
    let column = |meaning: &FvOperandMeaning| -> (Vec<f64>, usize) {
        match meaning {
            FvOperandMeaning::Datum { datum } => {
                (vec![samples[0].data[&program.data[*datum].name]], 0)
            }
            _ => (
                samples
                    .iter()
                    .map(|sample| match meaning {
                        FvOperandMeaning::State {
                            side: Some(malleus::FacetSide::Plus),
                        } => sample.plus,
                        FvOperandMeaning::State { .. } => sample.minus,
                        FvOperandMeaning::Normal { axis } => sample.normal[usize::from(*axis)],
                        _ => unreachable!(),
                    })
                    .collect(),
                1,
            ),
        }
    };
    let output = kernel.operand(&FvOperandMeaning::Output).unwrap();
    let columns = kernel
        .operands
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != output.index())
        .map(|(index, meaning)| (OperandId::new(index), column(meaning)))
        .collect::<Vec<_>>();
    let reads = columns
        .iter()
        .map(|(operand, (values, stride))| ReadBinding {
            operand: *operand,
            values,
            stride: *stride,
        })
        .collect::<Vec<_>>();
    let mut fluxes = vec![0.0; samples.len()];
    prepared
        .run_batch(
            &mut state,
            samples.len(),
            &reads,
            &mut [WriteBinding {
                operand: output,
                values: &mut fluxes,
                stride: 1,
            }],
        )
        .unwrap();
    for (sample, flux) in samples.iter().zip(&fluxes) {
        assert!(same(*flux, eval(&program, kernel, sample)));
    }
}

#[test]
fn obligations_and_identities_are_typed_and_stable() {
    let module = module();
    let llf = lower(
        &module,
        "Burgers1D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        &burgers_samples(),
    )
    .unwrap();
    let kinds = llf
        .obligations
        .iter()
        .map(|obligation| &obligation.kind)
        .collect::<Vec<_>>();
    assert!(matches!(
        kinds[..],
        [
            VerificationObligationKind::NumericalFluxConsistency { .. },
            VerificationObligationKind::NumericalFluxAntisymmetry { .. },
            VerificationObligationKind::Conservation {
                relation: ConservationRelation::DiscreteFluxBalance,
                ..
            },
            VerificationObligationKind::CflBound { .. },
            VerificationObligationKind::TotalVariationDiminishing { .. },
        ]
    ));
    for obligation in &llf.obligations {
        obligation.validate().unwrap();
    }
    let central = lower(&module, "Burgers1D", NumericalFluxIntent::Central, &[]).unwrap();
    assert!(!central.obligations.iter().any(|obligation| matches!(
        obligation.kind,
        VerificationObligationKind::TotalVariationDiminishing { .. }
    )));

    let again = lower(
        &module,
        "Burgers1D",
        NumericalFluxIntent::LocalLaxFriedrichs,
        &burgers_samples(),
    )
    .unwrap();
    assert_eq!(again.artifact_digest, llf.artifact_digest);
    assert_ne!(central.artifact_digest, llf.artifact_digest);
    let mut mutated = llf.clone();
    mutated.face_flux.kernel.name.push('x');
    assert_eq!(
        mutated.validate_identity().unwrap_err().code,
        "FV_IDENTITY_MISMATCH"
    );

    // The case-facing intent encoding Sinbad canonicalizes (D1).
    let muscl = FvMethodIntent {
        reconstruction: ReconstructionIntent::Muscl {
            variables: ReconstructionVariables::Conserved,
            gradient: GradientScheme::LeastSquares,
            limiter: LimiterFamily::VanLeer,
        },
        ..intent(NumericalFluxIntent::LocalLaxFriedrichs)
    };
    assert_eq!(
        serde_json::to_string(&muscl).unwrap(),
        r#"{"numerical_flux":{"kind":"local_lax_friedrichs"},"reconstruction":{"kind":"muscl","variables":"conserved","gradient":"least_squares","limiter":"van_leer"},"source":"pointwise","positivity":{"kind":"none"}}"#
    );
    let round_trip: FvMethodIntent =
        serde_json::from_str(&serde_json::to_string(&muscl).unwrap()).unwrap();
    assert_eq!(round_trip.digest(), muscl.digest());
}
