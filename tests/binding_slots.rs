use quantitas::UnitRegistry;
use scientia::{
    SlotStatus, compile_semantics, derive_binding_slots, validate_binding_slot_manifest,
};
use std::collections::BTreeMap;

/// A repository-local copy of the source behind `sinbad/physics/corpus/03-nonlinear-heat.res`,
/// used only to keep the parsing/elaboration/slot-derivation unit tests below independent of
/// the sinbad repository. The dedicated corpus-path tests further down intentionally do reach
/// into the sinbad checkout, matching the GX-A1 task's explicit instruction; see their doc
/// comments.
const NONLINEAR_HEAT_LOCAL: &str = r#"
module corpus.thermal.nonlinear_heat;

model NonlinearHeat {
    domain Omega { dimension = 2; coordinates = cartesian; }

    field T: state scalar H1(order=1) on Omega {
        quantity = ThermodynamicTemperature;
        unit = K;
        nominal = 300 K;
        time_role = differential;
    };

    property rho = density(T);
    property cp = specific_heat(T);
    property k = thermal_conductivity(T);
    source Q: VolumetricHeatSource;

    equation energy on Omega {
        rho * cp * dt(T) - div(k * grad(T)) = Q;
    }

    initial { T = exact_T(0); }

    boundary walls on boundary("walls") {
        dirichlet T = exact_T(t);
    }

    observable total_energy { integrate(rho * cp * T); }
}
"#;

#[test]
fn worked_example_c2_2_local_copy_has_four_unbound_providers() {
    let compilation = compile_semantics(NONLINEAR_HEAT_LOCAL, &UnitRegistry::si_bootstrap())
        .expect("nonlinear heat elaborates");
    let manifests = derive_binding_slots(&compilation);
    let [manifest] = manifests.as_slice() else {
        panic!("expected exactly one model");
    };
    validate_binding_slot_manifest(manifest).expect("slot ids are unique");

    let by_id = manifest
        .slots
        .iter()
        .map(|slot| (slot.id.as_str(), slot.status))
        .collect::<BTreeMap<_, _>>();

    assert_eq!(by_id.get("domain/Omega"), Some(&SlotStatus::Required));
    assert_eq!(by_id.get("region/walls"), Some(&SlotStatus::Required));
    assert_eq!(by_id.get("source/Q"), Some(&SlotStatus::Required));
    assert_eq!(by_id.get("boundary/walls/T"), Some(&SlotStatus::Required));
    assert_eq!(by_id.get("initial/T"), Some(&SlotStatus::Required));
    assert_eq!(
        by_id.get("observable/total_energy"),
        Some(&SlotStatus::ModelDefined)
    );
    assert_eq!(by_id.get("property/rho"), Some(&SlotStatus::ModelDefined));
    assert_eq!(by_id.get("property/cp"), Some(&SlotStatus::ModelDefined));
    assert_eq!(by_id.get("property/k"), Some(&SlotStatus::ModelDefined));

    let unbound_providers = [
        "density",
        "specific_heat",
        "thermal_conductivity",
        "exact_T",
    ];
    for provider in unbound_providers {
        assert_eq!(
            by_id.get(format!("provider/{provider}").as_str()),
            Some(&SlotStatus::Unbound),
            "provider `{provider}` should be Unbound before its signature is declared"
        );
    }
    // Exactly the four provider slots named in contract C2.2, no more.
    let provider_slot_count = manifest
        .slots
        .iter()
        .filter(|slot| slot.id.starts_with("provider/"))
        .count();
    assert_eq!(provider_slot_count, unbound_providers.len());
}

/// A model that declares provider signatures for every provider it calls, so every provider
/// slot becomes `Required` with typed inputs instead of `Unbound` (contract C2.2's closing
/// note: "After the corpus gains provider signatures ... the four Unbound rows become
/// Required.").
const DECLARED_PROVIDERS: &str = r#"
module tests.declared_providers;

model DeclaredProviders {
    domain Omega { dimension = 2; coordinates = cartesian; }

    field T: state scalar H1(order=1) on Omega {
        quantity = ThermodynamicTemperature;
        unit = K;
        time_role = differential;
    };

    provider thermal_conductivity(T: ThermodynamicTemperature) -> ThermalConductivity {
        shape = scalar;
        locality = pointwise;
        differentiability = symbolic;
        domain { T in [200 K, 2000 K]; }
    }
    provider density(material: selector) -> Density;
    provider exact_T(t: Time) -> ThermodynamicTemperature;

    property k = thermal_conductivity(T);
    property rho = density(0);

    source Q: VolumetricHeatSource;
    equation energy on Omega { -div(k * grad(T)) = Q; }
    initial { T = exact_T(0 s); }
}
"#;

#[test]
fn declared_providers_become_required_slots_with_typed_inputs() {
    let compilation = compile_semantics(DECLARED_PROVIDERS, &UnitRegistry::si_bootstrap())
        .expect("model with declared providers elaborates");
    assert!(
        compilation.advisories.is_empty(),
        "every provider call is declared, so there should be no RESOLVE_UNDECLARED_PROVIDER advisory"
    );
    let manifests = derive_binding_slots(&compilation);
    let [manifest] = manifests.as_slice() else {
        panic!("expected exactly one model");
    };
    validate_binding_slot_manifest(manifest).expect("slot ids are unique");

    let thermal_conductivity = manifest
        .slots
        .iter()
        .find(|slot| slot.id == "provider/thermal_conductivity")
        .expect("thermal_conductivity provider slot exists");
    assert_eq!(thermal_conductivity.status, SlotStatus::Required);
    assert_eq!(thermal_conductivity.inputs.len(), 1);
    assert_eq!(thermal_conductivity.inputs[0].name, "T");
    assert!(thermal_conductivity.inputs[0].quantity_kind.is_some());
    assert!(thermal_conductivity.inputs[0].dimension.is_some());

    let density = manifest
        .slots
        .iter()
        .find(|slot| slot.id == "provider/density")
        .expect("density provider slot exists");
    assert_eq!(density.status, SlotStatus::Required);
    assert_eq!(density.inputs.len(), 1);
    assert_eq!(density.inputs[0].name, "material");
    // `selector` is the non-physical pseudo-kind: no quantity kind, no dimension.
    assert!(density.inputs[0].quantity_kind.is_none());
    assert!(density.inputs[0].dimension.is_none());

    let exact_t = manifest
        .slots
        .iter()
        .find(|slot| slot.id == "provider/exact_T")
        .expect("exact_T provider slot exists");
    assert_eq!(exact_t.status, SlotStatus::Required);

    // No provider slot is Unbound now that every call resolves to a declared signature.
    assert!(
        manifest
            .slots
            .iter()
            .filter(|slot| slot.id.starts_with("provider/"))
            .all(|slot| slot.status != SlotStatus::Unbound)
    );
}

/// Deliberately reaches into the sinbad checkout, at the request of the GX-A1 task
/// specification (which names this exact path for the C2.2 worked-example assertion). This is
/// an explicit, opt-in exception to the "no compile-time or runtime path into Sinbad's product
/// corpus" invariant recorded in `STATUS.md`: it runs only when the workspace coordinator sets
/// `SINBAD_WORKSPACE` (the sibling-checkout root), and is skipped otherwise, so the ordinary
/// hermetic gate never reads outside this repository.
/// `worked_example_c2_2_local_copy_has_four_unbound_providers` above exercises the same
/// assertions against a repository-local copy.
#[test]
fn worked_example_c2_2_against_the_sinbad_corpus_file() {
    let Some(corpus) = sinbad_corpus_dir() else {
        return;
    };
    let path = corpus.join("03-nonlinear-heat.res");
    let source = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    let compilation =
        compile_semantics(&source, &UnitRegistry::si_bootstrap()).expect("corpus model elaborates");
    let manifests = derive_binding_slots(&compilation);
    let [manifest] = manifests.as_slice() else {
        panic!("expected exactly one model");
    };
    validate_binding_slot_manifest(manifest).expect("slot ids are unique");
    let unbound_providers = manifest
        .slots
        .iter()
        .filter(|slot| slot.id.starts_with("provider/") && slot.status == SlotStatus::Unbound)
        .count();
    assert_eq!(unbound_providers, 4);
}

/// Deliberately reaches into the sinbad checkout; see the doc comment on
/// `worked_example_c2_2_against_the_sinbad_corpus_file` above for why.
#[test]
fn derive_binding_slots_succeeds_for_the_complete_sinbad_corpus() {
    let Some(dir) = sinbad_corpus_dir() else {
        return;
    };
    let entries = std::fs::read_dir(&dir)
        .unwrap_or_else(|error| panic!("reading {}: {error}", dir.display()));
    let mut files = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "res"))
        .collect::<Vec<_>>();
    files.sort();
    assert_eq!(files.len(), 50, "expected the complete 50-model corpus");
    for path in files {
        let source = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
        let compilation = compile_semantics(&source, &UnitRegistry::si_bootstrap()).unwrap_or_else(
            |diagnostics| panic!("{} failed to elaborate: {diagnostics:?}", path.display()),
        );
        for manifest in derive_binding_slots(&compilation) {
            validate_binding_slot_manifest(&manifest).unwrap_or_else(|error| {
                panic!(
                    "{} model {} has invalid binding slots: {error}",
                    path.display(),
                    manifest.model
                )
            });
        }
    }
}

/// The Sinbad corpus directory, only when the workspace coordinator opted in through
/// `SINBAD_WORKSPACE`; `None` skips the cross-repository sweep in hermetic runs.
fn sinbad_corpus_dir() -> Option<std::path::PathBuf> {
    let Some(workspace) = std::env::var_os("SINBAD_WORKSPACE") else {
        eprintln!("skipping: SINBAD_WORKSPACE is not set; corpus sweep is opt-in");
        return None;
    };
    let dir = std::path::PathBuf::from(workspace).join("sinbad/physics/corpus");
    assert!(
        dir.is_dir(),
        "SINBAD_WORKSPACE is set but {} is not a directory",
        dir.display()
    );
    Some(dir)
}
