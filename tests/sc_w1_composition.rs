//! SC-W1 (`sinbad/ARCHITECTURE.md` §1–§3, §6, §11): scoped imports, `model` as the implicit
//! one-instance system, `system`/`instance`/`bind`, the system arena and origin map,
//! `scientia-system/1`, `scientia-operator-system/2`, kernel-level Composed bind chains on
//! Malleus `KernelComposition`, and the ordered module closure. Acceptance tests 1, 2, 4, 5
//! and the Scientia half of 3.

use quantitas::{QuantityKindRegistry, UnitRegistry};
use scientia::{
    BlockConstruction, BoundChain, DeclKind, InstanceId, Registries, SlotBinding, SystemError,
    block_digest, compile_model_system, compile_operator_system, compile_semantics, compile_system,
    compile_system_operator, derive_binding_slots, format_scientific_module,
    parse_scientific_module, resolve_module_closure,
};
use std::collections::{BTreeMap, BTreeSet};

const ELECTRICAL: &str = r#"
module physics.electrical;

pub model ElectricalConduction {
    domain conductor { dimension = 2; coordinates = cartesian; }
    field V: unknown scalar H1(order=1) on conductor;
    input field temperature: ThermodynamicTemperature on conductor;
    provider electrical_conductivity(T: ThermodynamicTemperature) -> ElectricalConductivity { differentiability = symbolic; }
    property sigma = electrical_conductivity(temperature);
    constitutive current_density = -sigma * grad(V);
    equation electrical on conductor { div(current_density) = 0; }
    boundary anode on boundary("anode") { dirichlet V = 1; }
    boundary cathode on boundary("cathode") { dirichlet V = 0; }
    output joule_heat: VolumetricHeatSource on conductor = sigma * dot(grad(V), grad(V));
    output potential = V;
}

model Private {
    domain conductor { dimension = 2; coordinates = cartesian; }
    field V: unknown scalar H1(order=1) on conductor;
    equation electrical on conductor { -div(grad(V)) = 0; }
}
"#;

const THERMAL: &str = r#"
module physics.thermal;

pub model HeatConduction {
    domain body { dimension = 2; coordinates = cartesian; }
    field T: state scalar H1(order=1) on body {
        quantity = ThermodynamicTemperature;
        unit = K;
        nominal = 300 K;
        time_role = differential;
    };
    input field Q: VolumetricHeatSource on body;
    input value ambient: ThermodynamicTemperature;
    provider density(T: ThermodynamicTemperature) -> Density { differentiability = symbolic; }
    provider specific_heat(T: ThermodynamicTemperature) -> SpecificHeat { differentiability = symbolic; }
    provider thermal_conductivity(T: ThermodynamicTemperature) -> ThermalConductivity { differentiability = symbolic; }
    property rho = density(T);
    property cp = specific_heat(T);
    property k = thermal_conductivity(T);
    equation thermal on body { rho * cp * dt(T) - div(k * grad(T)) = Q; }
    initial { T = ambient; }
    output temperature: ThermodynamicTemperature on body = T;
}
"#;

const ELECTROTHERMAL: &str = r#"
module systems.electrothermal;
use physics.electrical.{ElectricalConduction};
use physics.thermal as thermal;

pub system Electrothermal {
    domain body { dimension = 2; coordinates = cartesian; }
    instance electrical: ElectricalConduction(conductor = body);
    instance thermal: thermal.HeatConduction(body = body);
    bind electrical.temperature <- thermal.temperature;
    bind thermal.Q <- electrical.joule_heat;
}
"#;

fn modules() -> BTreeMap<String, String> {
    let mut modules = BTreeMap::new();
    modules.insert("physics.electrical".to_owned(), ELECTRICAL.to_owned());
    modules.insert("physics.thermal".to_owned(), THERMAL.to_owned());
    modules
}

fn registries() -> (UnitRegistry, QuantityKindRegistry) {
    (
        UnitRegistry::si_bootstrap(),
        QuantityKindRegistry::si_bootstrap(),
    )
}

fn system_of(root: &str, name: &str) -> Result<scientia::SystemCompilation, SystemError> {
    let closure = resolve_module_closure(root, &modules()).unwrap();
    let (units, kinds) = registries();
    compile_system(&closure, Registries::new(&units, &kinds), name)
}

#[test]
fn module_closure_is_ordered_and_imports_resolve_by_reference() {
    let closure = resolve_module_closure(ELECTROTHERMAL, &modules()).unwrap();
    let names = closure
        .modules
        .iter()
        .map(|module| module.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec![
            "physics.electrical",
            "physics.thermal",
            "systems.electrothermal"
        ]
    );
    assert_eq!(closure.root, "systems.electrothermal");
    assert_eq!(
        closure.identity,
        resolve_module_closure(ELECTROTHERMAL, &modules())
            .unwrap()
            .identity
    );
    let (units, kinds) = registries();
    let compilation = scientia::compile_module_in_closure(
        &closure,
        &closure.root,
        Registries::new(&units, &kinds),
    )
    .unwrap();
    let imports = compilation
        .semantic
        .imports
        .iter()
        .map(|import| {
            (
                import.name.as_str(),
                import.target.kind,
                import.target.name.as_str(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        imports,
        vec![
            (
                "ElectricalConduction",
                DeclKind::Model,
                "ElectricalConduction"
            ),
            ("thermal.HeatConduction", DeclKind::Model, "HeatConduction"),
        ],
        "private `Private` is not exported by the alias import"
    );
    assert_eq!(
        compilation.semantic.imports[0].target.module,
        closure.module("physics.electrical").unwrap().digest
    );

    let private =
        ELECTROTHERMAL.replace("{ElectricalConduction}", "{ElectricalConduction, Private}");
    let closure = resolve_module_closure(&private, &modules()).unwrap();
    let error = scientia::compile_module_in_closure(
        &closure,
        &closure.root,
        Registries::new(&units, &kinds),
    )
    .unwrap_err();
    assert!(
        error
            .iter()
            .any(|d| d.code == "RESOLVE_PRIVATE_DECLARATION"),
        "{error:?}"
    );
    let unknown = ELECTROTHERMAL.replace("{ElectricalConduction}", "{Nope}");
    let closure = resolve_module_closure(&unknown, &modules()).unwrap();
    let error = scientia::compile_module_in_closure(
        &closure,
        &closure.root,
        Registries::new(&units, &kinds),
    )
    .unwrap_err();
    assert!(
        error.iter().any(|d| d.code == "RESOLVE_UNKNOWN_IMPORT"),
        "{error:?}"
    );
}

#[test]
fn system_grammar_formats_idempotently() {
    for source in [ELECTRICAL, THERMAL, ELECTROTHERMAL] {
        let module = parse_scientific_module(source).unwrap();
        let formatted = format_scientific_module(&module);
        let reparsed = parse_scientific_module(&formatted).unwrap();
        assert_eq!(format_scientific_module(&reparsed), formatted);
        assert_eq!(
            scientia::semantic_digest(&reparsed),
            scientia::semantic_digest(&module)
        );
    }
    let formatted = format_scientific_module(&parse_scientific_module(ELECTROTHERMAL).unwrap());
    assert!(formatted.contains("use physics.electrical.{ElectricalConduction};"));
    assert!(formatted.contains("use physics.thermal;"));
    assert!(formatted.contains("pub system Electrothermal {"));
    assert!(formatted.contains("    instance thermal: thermal.HeatConduction(body = body);"));
    assert!(formatted.contains("    bind thermal.Q <- electrical.joule_heat;"));
    let formatted = format_scientific_module(&parse_scientific_module(ELECTRICAL).unwrap());
    assert!(formatted.contains("pub model ElectricalConduction {"));
    assert!(formatted.contains(
        "    output joule_heat: VolumetricHeatSource on conductor = (sigma * dot(grad(V), grad(V)));"
    ));
    assert!(formatted.contains("    output potential = V;"));
}

/// Acceptance test 3, Scientia half: the composed electrothermal system elaborates with one
/// coupled component, both binds are same-domain `Composed` chains, the per-model artifacts are
/// untouched, and every bind lowers onto a validated Malleus `KernelComposition` with a JVP.
#[test]
fn composed_electrothermal_lowers_binds_onto_kernel_compositions() {
    let compilation = system_of(ELECTROTHERMAL, "Electrothermal").unwrap();
    let system = &compilation.system;
    system.validate().unwrap();
    assert_eq!(system.schema, "scientia-system/1");
    assert!(!system.implicit);
    assert_eq!(system.root.kind, DeclKind::System);
    assert_eq!(
        system
            .instances
            .iter()
            .map(|instance| (instance.name.as_str(), instance.prefix.as_str()))
            .collect::<Vec<_>>(),
        vec![("electrical", "electrical/"), ("thermal", "thermal/")]
    );
    assert_eq!(system.variables.len(), 2);
    assert_eq!(system.variables[0].name, "electrical.V");
    assert_eq!(system.variables[1].name, "thermal.T");
    assert_eq!(system.residuals.len(), 2);
    assert_eq!(system.residuals[1].orientation, 1);
    assert_eq!(
        system.residuals[1].orientation_basis,
        scientia::OrientationBasis::Accumulation
    );
    assert_eq!(system.binds.len(), 2);
    assert!(
        system
            .binds
            .iter()
            .all(|bind| bind.chain == BoundChain::Composed)
    );
    assert_eq!(
        system.binds[0].consumer_slot,
        "electrical/input/temperature"
    );
    assert_eq!(system.binds[1].consumer_slot, "thermal/input/Q");
    let bound = system
        .slots
        .slots
        .iter()
        .filter(|slot| matches!(slot.binding, SlotBinding::Bound { .. }))
        .map(|slot| slot.id.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        bound,
        vec!["electrical/input/temperature", "thermal/input/Q"]
    );
    assert!(
        system
            .slots
            .slots
            .iter()
            .any(|slot| slot.id == "thermal/provider/density")
    );
    assert_eq!(system.slots.schema, "scientia-binding-slots/2");
    // One strongly connected component: the coupling is mutual, with no declaration saying so.
    assert_eq!(system.dependency.edges.len(), 2);
    assert_eq!(
        system.dependency.components,
        vec![vec![InstanceId(0), InstanceId(1)]]
    );
    assert!(!system.dependency.sequential);
    assert_eq!(system.origin_map.variables.len(), 2);
    assert_eq!(system.origin_map.residuals.len(), 2);
    assert_eq!(system.regions.len(), 2);
    assert_eq!(system.regions[0].name, "electrical.anode");

    let operator = compile_system_operator(&compilation).unwrap();
    operator.operator.validate().unwrap();
    assert_eq!(operator.operator.schema, "scientia-operator-system/2");
    assert_eq!(operator.model_systems.len(), 2);
    assert_eq!(operator.output_kernels.len(), 3);
    let kinds = operator
        .operator
        .blocks
        .iter()
        .map(|block| {
            (
                block.row.0,
                block.column.0,
                matches!(block.construction, BlockConstruction::Local { .. }),
            )
        })
        .collect::<Vec<_>>();
    // Row 0 (electrical) has its local V block and a composed T block through `temperature`;
    // row 1 (thermal) has its local T block and a composed V block through `joule_heat`.
    assert_eq!(
        kinds,
        vec![(0, 0, true), (0, 1, false), (1, 0, false), (1, 1, true)]
    );
    // `thermal.Q` is read by the thermal residual kernel: a Malleus composition. `electrical.
    // temperature` is read only inside `sigma = electrical_conductivity(temperature)`: the
    // provider-input path through the property, no fused composition.
    let path_of = |slot: &str| {
        operator
            .operator
            .blocks
            .iter()
            .find_map(|block| match &block.construction {
                BlockConstruction::Composed {
                    consumer_slot,
                    path,
                    ..
                } if consumer_slot == slot => Some(path.clone()),
                _ => None,
            })
            .unwrap()
    };
    assert!(matches!(
        path_of("thermal/input/Q"),
        scientia::ComposedPath::KernelInput { ref compositions, ref jvp_compositions }
            if !compositions.is_empty() && compositions.len() == jvp_compositions.len()
    ));
    let scientia::ComposedPath::ProviderInput { properties } =
        path_of("electrical/input/temperature")
    else {
        panic!("temperature reaches the electrical residual through sigma");
    };
    assert_eq!(
        properties.len(),
        2,
        "sigma and current_density depend on temperature"
    );
    assert_eq!(properties[0].slot, "electrical/property/sigma");
    assert_eq!(
        properties[0].providers,
        vec!["electrical/provider/electrical_conductivity".to_string()]
    );
    assert!(!operator.compositions.is_empty());
    for composition in &operator.compositions {
        assert_eq!(composition.composition.stages.len(), 2);
        malleus::validate_composition(composition.composition.clone()).unwrap();
        assert_eq!(
            composition.digest.hex,
            malleus::composition_digest(&composition.composition).hex
        );
        assert_ne!(composition.jvp_digest, composition.digest);
    }
    // Determinism: compiling again reproduces every identity.
    let again =
        compile_system_operator(&system_of(ELECTROTHERMAL, "Electrothermal").unwrap()).unwrap();
    assert_eq!(again.operator.identity, operator.operator.identity);
    assert_eq!(again.system.identity, system.identity);
    let json = serde_json::to_string(&operator.operator).unwrap();
    let decoded: scientia::SystemOperator = serde_json::from_str(&json).unwrap();
    decoded.validate().unwrap();
    assert_eq!(decoded, operator.operator);
}

/// Acceptance test 1: two instances of one model compile with disjoint prefixed slot ids and
/// reuse the model's artifacts byte-for-byte.
#[test]
fn two_instances_of_one_model_share_artifacts_and_prefix_slots() {
    const TWO: &str = r#"
module systems.two_heat;
use physics.thermal.{HeatConduction};

pub system TwoHeat {
    domain body { dimension = 2; coordinates = cartesian; }
    instance a: HeatConduction(body = body);
    instance b: HeatConduction(body = body);
}
"#;
    let compilation = system_of(TWO, "TwoHeat").unwrap();
    let ids = compilation
        .system
        .slots
        .slots
        .iter()
        .map(|slot| slot.id.clone())
        .collect::<BTreeSet<_>>();
    assert!(ids.contains("a/input/Q") && ids.contains("b/input/Q"));
    assert!(ids.contains("a/provider/density") && ids.contains("b/provider/density"));
    assert_eq!(
        ids.len(),
        compilation.system.slots.slots.len(),
        "no id collides"
    );
    let operator = compile_system_operator(&compilation).unwrap();
    let [(_, first), (_, second)] = operator.model_systems.as_slice() else {
        panic!("two instances");
    };
    assert_eq!(first.artifact_digest, second.artifact_digest);
    assert_eq!(first, second, "the per-model artifact is reused verbatim");
    assert_eq!(
        operator.operator.residuals[0].block,
        operator.operator.residuals[1].block
    );
    assert_eq!(
        compilation.system.instances[0].semantic_digest,
        compilation.system.instances[1].semantic_digest
    );
    assert_eq!(operator.operator.blocks.len(), 2);
    assert!(compilation.system.dependency.sequential);
}

/// Acceptance test 2: a `model` compiled directly and as the implicit one-instance system
/// produce byte-equal per-equation artifacts, the same `scientia-operator-system/1` digest, and
/// unprefixed slot ids.
#[test]
fn model_direct_and_implicit_system_agree_byte_for_byte() {
    let closure = resolve_module_closure(THERMAL, &BTreeMap::<String, String>::new()).unwrap();
    let (units, kinds) = registries();
    let compilation =
        compile_model_system(&closure, Registries::new(&units, &kinds), "HeatConduction").unwrap();
    let system = &compilation.system;
    assert!(system.implicit);
    assert_eq!(system.root.kind, DeclKind::Model);
    assert_eq!(system.instances.len(), 1);
    assert_eq!(system.instances[0].prefix, "");
    assert_eq!(
        system.instances[0].domain_map,
        vec![(scientia::DomainId(0), scientia::SysDomainId(0))]
    );
    let direct = compile_semantics(THERMAL, &UnitRegistry::si_bootstrap()).unwrap();
    let direct_ids = derive_binding_slots(&direct)[0]
        .slots
        .iter()
        .map(|slot| slot.id.clone())
        .collect::<Vec<_>>();
    let system_ids = system
        .slots
        .slots
        .iter()
        .map(|slot| slot.id.clone())
        .collect::<Vec<_>>();
    assert_eq!(system_ids, direct_ids);
    assert!(
        system
            .slots
            .slots
            .iter()
            .all(|slot| matches!(slot.binding, SlotBinding::Open))
    );

    let operator = compile_system_operator(&compilation).unwrap();
    let direct_operator =
        compile_operator_system(&direct.semantic, "HeatConduction", &["thermal"]).unwrap();
    assert_eq!(operator.model_systems[0].1, direct_operator);
    assert_eq!(
        operator.operator.residuals[0].block,
        block_digest(&direct_operator.blocks[0])
    );
    assert_eq!(operator.output_kernels.len(), 1);
}

/// Acceptance test 4: removing a bind and re-binding the input to case data yields a DAG that
/// executes sequentially; removing it without re-binding is refused as an open input.
#[test]
fn dropping_a_bind_opens_the_input_and_rebinding_makes_a_dag() {
    let dropped = ELECTROTHERMAL.replace("    bind thermal.Q <- electrical.joule_heat;\n", "");
    let compilation = system_of(&dropped, "Electrothermal").unwrap();
    let system = &compilation.system;
    assert!(
        system
            .open_inputs()
            .iter()
            .any(|slot| slot.id == "thermal/input/Q")
    );
    // Case data closes every other required slot (providers, regions, ...) but not `Q`.
    let mut bound = system
        .open_inputs()
        .iter()
        .map(|slot| slot.id.clone())
        .collect::<BTreeSet<_>>();
    bound.remove("thermal/input/Q");
    let error = system.require_closed(&bound).unwrap_err();
    assert_eq!(error, SystemError::OpenInput("thermal/input/Q".into()));
    bound.insert("thermal/input/Q".to_owned());
    system.require_closed(&bound).unwrap();
    assert!(system.dependency.sequential);
    assert_eq!(
        system.dependency.components,
        vec![vec![InstanceId(1)], vec![InstanceId(0)]],
        "thermal first, then electrical"
    );
    let operator = compile_system_operator(&compilation).unwrap();
    assert_eq!(
        operator
            .operator
            .blocks
            .iter()
            .filter(|block| matches!(block.construction, BlockConstruction::Composed { .. }))
            .count(),
        1
    );
}

/// Acceptance test 5 (SC-W1 part): duplicate producers, private symbols, kind and support
/// mismatches, and domain-parameter errors are refused before realization with typed codes.
#[test]
fn bind_and_instance_refusals_are_typed() {
    let code = |source: &str| system_of(source, "Electrothermal").unwrap_err().to_string();

    let duplicate = ELECTROTHERMAL.replace(
        "    bind thermal.Q <- electrical.joule_heat;\n",
        "    bind thermal.Q <- electrical.joule_heat;\n    bind thermal.Q <- electrical.joule_heat;\n",
    );
    assert!(code(&duplicate).starts_with("SYSTEM_DUPLICATE_BINDING"));

    let private_input = ELECTROTHERMAL.replace("bind thermal.Q <-", "bind thermal.T <-");
    assert!(code(&private_input).starts_with("SYSTEM_PRIVATE_SYMBOL"));
    let private_output = ELECTROTHERMAL.replace("electrical.joule_heat", "electrical.sigma");
    assert!(code(&private_output).starts_with("SYSTEM_PRIVATE_SYMBOL"));
    let unknown_output = ELECTROTHERMAL.replace("electrical.joule_heat", "electrical.nope");
    assert!(code(&unknown_output).starts_with("SYSTEM_UNKNOWN_OUTPUT"));
    let unknown_input = ELECTROTHERMAL.replace("bind thermal.Q <-", "bind thermal.nope <-");
    assert!(code(&unknown_input).starts_with("SYSTEM_UNKNOWN_INPUT"));
    let unknown_instance = ELECTROTHERMAL.replace("bind thermal.Q <-", "bind nope.Q <-");
    assert!(code(&unknown_instance).starts_with("SYSTEM_UNKNOWN_INSTANCE"));

    let kind = ELECTROTHERMAL.replace("electrical.joule_heat", "thermal.temperature");
    assert!(
        code(&kind).starts_with("SYSTEM_BIND_KIND_MISMATCH"),
        "{}",
        code(&kind)
    );
    let support = ELECTROTHERMAL.replace(
        "bind thermal.Q <- electrical.joule_heat;",
        "bind thermal.ambient <- thermal.temperature;",
    );
    assert!(
        code(&support).starts_with("SYSTEM_BIND_SUPPORT_MISMATCH"),
        "{}",
        code(&support)
    );

    let unmapped = ELECTROTHERMAL.replace(
        "thermal.HeatConduction(body = body)",
        "thermal.HeatConduction",
    );
    assert!(code(&unmapped).starts_with("SYSTEM_DOMAIN_UNMAPPED"));
    let mismatch = ELECTROTHERMAL.replace("dimension = 2;", "dimension = 3;");
    assert!(code(&mismatch).starts_with("SYSTEM_DOMAIN_MISMATCH"));
    let parameter = ELECTROTHERMAL.replace("(conductor = body)", "(wire = body)");
    assert!(code(&parameter).starts_with("SYSTEM_UNKNOWN_PARAMETER"));
    let domain = ELECTROTHERMAL.replace("(conductor = body)", "(conductor = nope)");
    assert!(code(&domain).starts_with("SYSTEM_UNKNOWN_DOMAIN"));
    let model = ELECTROTHERMAL.replace(
        "instance electrical: ElectricalConduction",
        "instance electrical: Nope",
    );
    assert!(code(&model).starts_with("SYSTEM_UNKNOWN_MODEL"));
    let cross = ELECTROTHERMAL.replace(
        "    instance thermal: thermal.HeatConduction(body = body);",
        "    domain other { dimension = 2; coordinates = cartesian; }\n    instance thermal: thermal.HeatConduction(body = other);",
    );
    assert!(
        code(&cross).starts_with("SYSTEM_BIND_CROSS_DOMAIN"),
        "{}",
        code(&cross)
    );
}

/// Every corpus model still elaborates as its implicit one-instance system with exactly the
/// `scientia-binding-slots/1` ids (opt-in through `SINBAD_WORKSPACE`), and 08 compiles the full
/// `scientia-operator-system/2` monolithically.
#[test]
fn sinbad_corpus_models_are_implicit_one_instance_systems() {
    let Some(workspace) = std::env::var_os("SINBAD_WORKSPACE") else {
        eprintln!("skipping: SINBAD_WORKSPACE is not set; corpus sweep is opt-in");
        return;
    };
    let dir = std::path::PathBuf::from(workspace).join("sinbad/physics/corpus");
    let (units, kinds) = registries();
    let mut count = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("res") {
            continue;
        }
        let source = std::fs::read_to_string(&path).unwrap();
        let direct = compile_semantics(&source, &UnitRegistry::si_bootstrap())
            .unwrap_or_else(|e| panic!("{}: {e:?}", path.display()));
        let closure = resolve_module_closure(&source, &BTreeMap::<String, String>::new()).unwrap();
        for (index, model) in direct.semantic.models.iter().enumerate() {
            let compilation =
                compile_model_system(&closure, Registries::new(&units, &kinds), &model.name)
                    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let direct_ids = derive_binding_slots(&direct)[index]
                .slots
                .iter()
                .map(|slot| slot.id.clone())
                .collect::<Vec<_>>();
            let system_ids = compilation
                .system
                .slots
                .slots
                .iter()
                .map(|slot| slot.id.clone())
                .collect::<Vec<_>>();
            assert_eq!(system_ids, direct_ids, "{}", path.display());
            assert_eq!(compilation.system.instances[0].prefix, "");
            count += 1;
        }
        if path.ends_with("08-electrothermal-joule.res") {
            let compilation = compile_model_system(
                &closure,
                Registries::new(&units, &kinds),
                "ElectrothermalJoule",
            )
            .unwrap();
            let operator = compile_system_operator(&compilation).unwrap();
            assert_eq!(operator.operator.residuals.len(), 2);
            assert_eq!(operator.operator.blocks.len(), 4);
        }
    }
    assert!(count >= 50, "{count} corpus models");
}
