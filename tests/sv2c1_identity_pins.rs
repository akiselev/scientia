//! Identity pins recorded before SV2-C1a: the FC10 affine finite-volume path must stay
//! byte-stable while the conservation-law lowering is added beside it.
use quantitas::UnitRegistry;
use scientia::{AffineMethodKernelSpec, compile_conservation_law_method, compile_semantics};

const SOURCE: &str = r#"
module fixtures.fc10;

model Conservation {
  domain Cells { dimension = 1; coordinates = cartesian; }
  field q: state scalar DG(order=0) on Cells { time_role = differential; };
  property speed = transport_speed(0);
  equation balance on Cells { dt(q) + div(speed * q) = 0; }
}
"#;

#[test]
fn fc10_affine_finite_volume_identity_is_unchanged() {
    let module = compile_semantics(SOURCE, &UnitRegistry::si_bootstrap())
        .unwrap()
        .semantic;
    let program = compile_conservation_law_method(
        &module,
        "Conservation",
        "balance",
        "q",
        AffineMethodKernelSpec {
            name: "upwind_flux".into(),
            inputs: vec!["minus".into(), "plus".into()],
            coefficients: vec![2.0, 0.0],
            constant: 0.0,
        },
    )
    .unwrap();
    let bytes = serde_json::to_vec(&program).unwrap();
    let serialized = blake3::hash(&bytes).to_hex().to_string();
    assert_eq!(
        program.artifact_digest.hex,
        "9868d4cb01fc7013b30914fca456392c4a7707a88a2864a086be92f3404163d0"
    );
    assert_eq!(
        program.local_kernel.as_ref().unwrap().artifact_digest.hex,
        "a8ac803ab707dbf28298c8e98dfad12141e3f7b3c59045d2b220446489271c39"
    );
    assert_eq!(
        serialized,
        "9d04ac4a275389fae3c413ba908013fedc6ac40217aab7b0e4df975e79a9171a"
    );
}
