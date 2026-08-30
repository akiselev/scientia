//! GX-F4: `resolve_modules` wired into `compile_semantics_with` through a `ModuleSource`
//! (contract: `NoImports` refuses every `use` for hermetic callers, `FilesystemModuleSource`
//! maps a dotted module name to a `.res` file under a root). Imported modules contribute only
//! `provider` declarations into the importing model's scope; cross-module duplicates are
//! `RESOLVE_DUPLICATE_NAME`, import cycles are `RESOLVE_IMPORT_CYCLE`, missing modules are
//! `RESOLVE_MISSING_MODULE`.

use quantitas::{QuantityKindRegistry, UnitRegistry};
use scientia::{
    Registries, SlotStatus, compile_semantics_with, derive_binding_slots,
    scientific::{FilesystemModuleSource, NoImports},
};
use std::collections::BTreeMap;

fn registries() -> (UnitRegistry, QuantityKindRegistry) {
    (
        UnitRegistry::si_bootstrap(),
        QuantityKindRegistry::si_bootstrap(),
    )
}

const CATALOG: &str = r#"
module physics.providers.thermal;

model ThermalCatalog {
  domain Placeholder { dimension = 1; coordinates = cartesian; }
  provider thermal_conductivity(T: ThermodynamicTemperature) -> ThermalConductivity {
    unit = W/(m*K);
    domain { T in [200 K, 2000 K]; }
  }
}
"#;

const IMPORTING_MODEL: &str = r#"
module gx_f4.importing;
use physics.providers.thermal;

model Heat {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega {
    quantity = ThermodynamicTemperature;
    unit = K;
    time_role = differential;
  };
  property k = thermal_conductivity(T);
  source Q: VolumetricHeatSource;
  equation energy on Omega { -div(k * grad(T)) = Q; }
}
"#;

fn catalog_source() -> BTreeMap<String, String> {
    let mut modules = BTreeMap::new();
    modules.insert("physics.providers.thermal".to_owned(), CATALOG.to_owned());
    modules
}

#[test]
fn imported_provider_becomes_a_required_slot_with_typed_inputs() {
    let (units, kinds) = registries();
    let compilation = compile_semantics_with(
        IMPORTING_MODEL,
        Registries::new(&units, &kinds),
        &catalog_source(),
    )
    .expect("the importing model resolves its provider through the catalog module");
    assert!(
        compilation.advisories.is_empty(),
        "the provider call is now declared (via import), so no RESOLVE_UNDECLARED_PROVIDER \
         advisory should remain: {:#?}",
        compilation.advisories
    );
    // The importing model's own arena carries the imported provider signature.
    let providers = &compilation.semantic.models[0].providers;
    assert_eq!(providers.len(), 1);
    assert_eq!(providers[0].name, "thermal_conductivity");
    assert_eq!(providers[0].inputs.len(), 1);
    assert!(providers[0].inputs[0].quantity_kind.is_some());
    assert!(providers[0].inputs[0].dimension.is_some());

    let manifests = derive_binding_slots(&compilation);
    let [manifest] = manifests.as_slice() else {
        panic!("expected exactly one model");
    };
    let slot = manifest
        .slots
        .iter()
        .find(|slot| slot.id == "provider/thermal_conductivity")
        .expect("imported provider slot exists");
    assert_eq!(slot.status, SlotStatus::Required);
    assert_eq!(slot.inputs.len(), 1);
    assert_eq!(slot.inputs[0].name, "T");
}

#[test]
fn a_local_provider_that_shadows_an_imported_one_is_a_cross_module_duplicate() {
    const SHADOWING_MODEL: &str = r#"
module gx_f4.shadowing;
use physics.providers.thermal;

model Heat {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field T: state scalar H1(order=1) on Omega {
    quantity = ThermodynamicTemperature;
    unit = K;
    time_role = differential;
  };
  provider thermal_conductivity(T: ThermodynamicTemperature) -> ThermalConductivity;
  property k = thermal_conductivity(T);
}
"#;
    let (units, kinds) = registries();
    let diagnostics = compile_semantics_with(
        SHADOWING_MODEL,
        Registries::new(&units, &kinds),
        &catalog_source(),
    )
    .unwrap_err();
    let diagnostic = diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code == "RESOLVE_DUPLICATE_NAME")
        .expect("a local re-declaration of an imported provider name is a duplicate");
    assert_eq!(diagnostic.related.len(), 1, "both spans are retained");
}

#[test]
fn a_missing_imported_module_is_refused() {
    const SOURCE: &str = r#"
module gx_f4.missing;
use physics.providers.nonexistent;
model M {
  domain Omega { dimension = 1; coordinates = cartesian; }
}
"#;
    let (units, kinds) = registries();
    let diagnostics =
        compile_semantics_with(SOURCE, Registries::new(&units, &kinds), &BTreeMap::new())
            .unwrap_err();
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RESOLVE_MISSING_MODULE"),
        "{diagnostics:#?}"
    );
}

#[test]
fn an_import_cycle_is_refused() {
    let mut modules = BTreeMap::new();
    modules.insert(
        "gx_f4.cycle_a".to_owned(),
        "module gx_f4.cycle_a;\nuse gx_f4.cycle_b;\nmodel A { domain Omega { dimension = 1; coordinates = cartesian; } }\n".to_owned(),
    );
    modules.insert(
        "gx_f4.cycle_b".to_owned(),
        "module gx_f4.cycle_b;\nuse gx_f4.cycle_a;\nmodel B { domain Omega { dimension = 1; coordinates = cartesian; } }\n".to_owned(),
    );
    const ROOT: &str = "module gx_f4.cycle_root;\nuse gx_f4.cycle_a;\nmodel Root { domain Omega { dimension = 1; coordinates = cartesian; } }\n";
    let (units, kinds) = registries();
    let diagnostics =
        compile_semantics_with(ROOT, Registries::new(&units, &kinds), &modules).unwrap_err();
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RESOLVE_IMPORT_CYCLE"),
        "{diagnostics:#?}"
    );
}

#[test]
fn no_imports_refuses_any_use_import_for_a_hermetic_caller() {
    const SOURCE: &str = r#"
module gx_f4.hermetic;
use physics.providers.thermal;
model M {
  domain Omega { dimension = 1; coordinates = cartesian; }
}
"#;
    let (units, kinds) = registries();
    let diagnostics =
        compile_semantics_with(SOURCE, Registries::new(&units, &kinds), &NoImports).unwrap_err();
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "RESOLVE_MISSING_MODULE"),
        "{diagnostics:#?}"
    );
}

#[test]
fn no_imports_still_elaborates_a_model_with_no_use_statement() {
    const SOURCE: &str = r#"
module gx_f4.hermetic_no_imports;
model M {
  domain Omega { dimension = 1; coordinates = cartesian; }
}
"#;
    let (units, kinds) = registries();
    compile_semantics_with(SOURCE, Registries::new(&units, &kinds), &NoImports)
        .expect("no `use` statement means `NoImports` is never consulted");
}

#[test]
fn filesystem_module_source_maps_a_dotted_name_to_a_nested_res_file() {
    let root = std::env::temp_dir().join(format!("scientia-gx-f4-fs-test-{}", std::process::id()));
    let module_dir = root.join("physics").join("providers");
    std::fs::create_dir_all(&module_dir).expect("create scratch module tree");
    std::fs::write(module_dir.join("thermal.res"), CATALOG).expect("write catalog module");

    let loader = FilesystemModuleSource { root: root.clone() };
    let (units, kinds) = registries();
    let result = compile_semantics_with(IMPORTING_MODEL, Registries::new(&units, &kinds), &loader);

    std::fs::remove_dir_all(&root).ok();

    let compilation = result
        .expect("`use physics.providers.thermal;` loads `<root>/physics/providers/thermal.res`");
    assert!(
        compilation.advisories.is_empty(),
        "{:#?}",
        compilation.advisories
    );
}
