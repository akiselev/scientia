use quantitas::{QuantityKindRegistry, UnitRegistry};
use scientia::{NoImports, Registries, compile_system, resolve_module_closure};
const SOURCE: &str = r#"
module ports.test;
pub connector Boundary {
 equal temperature: ThermodynamicTemperature;
 balance heat_flux: HeatFlux conserves Energy;
}
pub model Conduction {
 domain body { dimension = 3; coordinates = cartesian; }
 region wall: boundary of body;
 field T: unknown scalar H1(order=1) on body { quantity = ThermodynamicTemperature; unit = K; };
 provider conductivity() -> ThermalConductivity { differentiability = symbolic; }
 property k = conductivity();
 constitutive q = -k * grad(T);
 equation energy on body oriented by q { div(q) = 0; }
 port contact: Boundary on wall from equation energy {
   temperature = trace(T);
   heat_flux = boundary_flux(energy);
 }
}
system Pair {
 domain left { dimension = 3; coordinates = cartesian; }
 domain right { dimension = 3; coordinates = cartesian; }
 interface joint between boundary(left), boundary(right);
 instance a: Conduction(body = left, wall = joint);
 instance b: Conduction(body = right, wall = joint);
 connect(a.contact, b.contact);
}
"#;
fn compile(source: &str) -> Result<scientia::SystemCompilation, String> {
    let closure = resolve_module_closure(source, &NoImports).map_err(|e| e.to_string())?;
    compile_system(
        &closure,
        Registries::new(
            &UnitRegistry::si_bootstrap(),
            &QuantityKindRegistry::si_bootstrap(),
        ),
        "Pair",
    )
    .map_err(|e| e.to_string())
}
#[test]
fn typed_connection_relations_retain_source_orientation_and_roundtrip() {
    let compiled = compile(SOURCE).unwrap();
    let connection = &compiled.system.connections[0];
    connection.validate().unwrap();
    assert_eq!(connection.equal_coefficients, [1, -1]);
    assert_eq!(connection.balance_coefficients, [1, 1]);
    assert_ne!(connection.ports[0].variable, connection.ports[1].variable);
    assert_ne!(connection.ports[0].region, connection.ports[1].region);
    assert!(connection.ports.iter().all(|p| p.orientation == 1));
    let flipped = compile(&SOURCE.replace("div(q) = 0", "0 = div(q)")).unwrap();
    assert!(
        flipped.system.connections[0]
            .ports
            .iter()
            .all(|p| p.orientation == -1)
    );
    assert_ne!(flipped.system.connections[0].identity, connection.identity);
    let closure = resolve_module_closure(SOURCE, &NoImports).unwrap();
    let formatted = scientia::format_scientific_module(&closure.root_module().module);
    let roundtrip = compile(&formatted).unwrap();
    assert_eq!(roundtrip.system.identity, compiled.system.identity);
    let mut mutation = connection.clone();
    mutation.balance_coefficients = [1, -1];
    assert!(mutation.validate().is_err());
}
#[test]
fn missing_closures_wrong_types_orientation_and_domain_relations_refuse() {
    for (source, expected) in [
        (
            SOURCE.replace(
                "body = left, wall = joint",
                "body = left, body = left, wall = joint",
            ),
            "SYSTEM_ARGUMENT_DUPLICATE",
        ),
        (
            SOURCE.replace("connect(a.contact, b.contact);", ""),
            "SYSTEM_PORT_UNCLOSED",
        ),
        (
            SOURCE.replace(
                "connect(a.contact, b.contact);",
                "connect(a.contact, a.contact);",
            ),
            "SYSTEM_PORT_OVERLAP",
        ),
        (
            SOURCE.replace("oriented by q", ""),
            "PORT_FLUX_ORIENTATION_UNDECIDABLE",
        ),
        (
            SOURCE.replace("heat_flux: HeatFlux", "heat_flux: ElectricPotential"),
            "PORT_KIND_MISMATCH",
        ),
        (
            SOURCE.replace("wall = joint", "wall = missing"),
            "SYSTEM_INTERFACE_UNKNOWN",
        ),
        (
            SOURCE.replace("temperature = trace(T);", ""),
            "PORT_MEMBER_UNASSIGNED",
        ),
    ] {
        let error = compile(&source).unwrap_err();
        assert!(error.contains(expected), "{expected}: {error}");
    }
}
