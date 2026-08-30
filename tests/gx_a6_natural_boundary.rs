//! GX-A6: region/domain resolution from the equation domain, and the natural-boundary
//! convention that synthesizes an implicit whole-boundary region when integration by parts
//! needs an exterior boundary and the model declares none for that domain.

use quantitas::UnitRegistry;
use scientia::{
    FormAssumption, RegionId, RegionKind, compile_semantics, derive_variational_form,
    infer_form_requirements,
};

/// No `boundary`/`interface` declaration and no authored `form` block name any region for
/// `Omega`, so deriving `balance` needs the natural-boundary convention.
const NO_DECLARED_BOUNDARY: &str = r#"
module gx_a6.no_boundary;
model NoBoundary {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: unknown scalar H1(order=1) on Omega;
  property k = diffusivity(0);
  source f: VolumetricSource;
  equation balance on Omega { -div(k * grad(u)) = f; }
}
"#;

#[test]
fn integration_by_parts_synthesizes_the_implicit_whole_boundary_region() {
    let compilation = compile_semantics(NO_DECLARED_BOUNDARY, &UnitRegistry::si_bootstrap())
        .expect("model with no declared boundary still elaborates");
    let form = derive_variational_form(&compilation.semantic, "NoBoundary", "balance")
        .expect("integration by parts no longer refuses with FORM_BOUNDARY_PARTITION_REQUIRED");

    let domain = compilation.semantic.models[0].domains[0].id;
    let synthetic_region = RegionId::generated_for(domain);
    assert!(synthetic_region.is_generated());
    assert_eq!(synthetic_region.generated_domain(), Some(domain));

    let partition = form
        .receipt
        .assumptions
        .iter()
        .find_map(|assumption| match assumption {
            FormAssumption::ExteriorRegionsPartitionBoundary {
                domain: assumption_domain,
                regions,
                implicit_natural_boundary,
            } if *assumption_domain == domain => Some((regions, *implicit_natural_boundary)),
            _ => None,
        })
        .expect("form records an exterior-boundary-partition assumption");
    assert_eq!(partition.0, &vec![synthetic_region]);
    assert!(
        partition.1,
        "the region was synthesized, so implicit_natural_boundary must be true"
    );

    // The retained boundary term is a natural (Neumann-zero unless bound) flux term: no
    // boundary condition applies to a region the model never declared, so it is `Retained`,
    // not eliminated or substituted.
    let boundary_term = form
        .receipt
        .boundary_terms
        .iter()
        .find(|term| term.region == synthetic_region)
        .expect("a boundary term is retained on the synthetic region");
    assert!(matches!(
        boundary_term.disposition,
        scientia::BoundaryTermDisposition::Retained { .. }
    ));

    // FC3 still discharges a `BoundaryPartitionRequirement` for the synthetic region, so a
    // downstream realization (Finitum) must still supply boundary data for it.
    let requirements = infer_form_requirements(&compilation.semantic, &form)
        .expect("requirements still derive with the synthetic region");
    assert!(
        requirements
            .boundary_partitions
            .iter()
            .any(|partition| partition.domain == domain
                && partition.exterior_regions == vec![synthetic_region])
    );
}

/// A region named only by a form's own facet measure inherits that form's `cell(<Domain>)`
/// measure domain instead of staying domain-less.
const FORM_MEASURE_ONLY_REGION: &str = r#"
module gx_a6.form_measure_region;
model FormMeasureRegion {
  domain Omega { dimension = 2; coordinates = cartesian; }
  field u: trial scalar H1(order=1) on Omega;
  field v: test scalar H1(order=1) on Omega;
  form residual {
    cell(Omega): u * v;
    boundary(walls): u * v;
  }
}
"#;

#[test]
fn a_region_named_only_by_a_form_measure_inherits_the_forms_cell_domain() {
    let compilation = compile_semantics(FORM_MEASURE_ONLY_REGION, &UnitRegistry::si_bootstrap())
        .expect("model elaborates");
    let model = &compilation.semantic.models[0];
    let domain = model.domains[0].id;
    let walls = model
        .regions
        .iter()
        .find(|region| region.name == "walls" && region.kind == RegionKind::ExteriorFacet)
        .expect("the boundary(walls) measure declares a region");
    assert_eq!(
        walls.domain,
        Some(domain),
        "a region named only by a form measure must inherit that form's cell domain"
    );
}
