// Scalar conservation laws for the SV2-C1a finite-volume lowering (Scientia test fixture; the
// product model under sinbad/physics is owned by Sinbad). Conserved fields use DG(order=0)
// (decision D1). A 1-D flux is a scalar; a 2-D flux is a vector with one component per axis.
module fixtures.scalar_conservation;

model LinearAdvection1D {
  domain Line { dimension = 1; coordinates = cartesian; }
  field u: state scalar DG(order=0) on Line { time_role = differential; };
  parameter a;
  constitutive flux = a * u;
  equation balance on Line { dt(u) + div(flux) = 0; }
  boundary inflow on boundary("left") { dirichlet u = 1; }
  boundary outflow on boundary("right") { neumann u = a * u; }
}

model LinearAdvection2D {
  domain Square { dimension = 2; coordinates = cartesian; }
  field u: state scalar DG(order=0) on Square { time_role = differential; };
  parameter ax;
  parameter ay;
  equation balance on Square { dt(u) + div([ax * u, ay * u]) = 0; }
}

model Burgers1D {
  domain Line { dimension = 1; coordinates = cartesian; }
  field u: state scalar DG(order=0) on Line { time_role = differential; };
  parameter k;
  constitutive flux = 0.5 * u * u;
  equation balance on Line { dt(u) + div(flux) = -k * u; }
}

// Authored two-point fluxes: traces select the side, `normal_component` dots with n.
model AuthoredBurgers2D {
  domain Square { dimension = 2; coordinates = cartesian; }
  field u: state scalar DG(order=0) on Square { time_role = differential; };
  constitutive flux = [0.5 * u * u, 0.5 * u * u];
  equation balance on Square { dt(u) + div(flux) = 0; }

  constitutive rusanov = average(normal_component(flux)) - 0.5 * max(abs(trace_minus(u)), abs(trace_plus(u))) * (trace_plus(u) - trace_minus(u));
  constitutive one_sided = normal_component_minus(flux);
  constitutive doubled = 2 * average(normal_component(flux));
  constitutive untraced = normal_component(flux) + u;
}

model Diffusive1D {
  domain Line { dimension = 1; coordinates = cartesian; }
  field u: state scalar DG(order=0) on Line { time_role = differential; };
  parameter a;
  equation balance on Line { dt(u) + div(a * u - grad(u)) = 0; }
}

// SV2-C1c: diffusive terms. `div(k * grad(u))` is a separate additive divergence; `k` is a
// parameter (uniform), a constitutive of u (k(u)) or a property (bound per cell by the realizer).
model AdvectionDiffusion1D {
  domain Line { dimension = 1; coordinates = cartesian; }
  field u: state scalar DG(order=0) on Line { time_role = differential; };
  parameter a;
  parameter k;
  equation balance on Line { dt(u) + div(a * u) - div(k * grad(u)) = 0; }
  boundary left on boundary("left") { dirichlet u = 1; }
  boundary right on boundary("right") { dirichlet u = 0; }
}

model NonlinearDiffusion1D {
  domain Line { dimension = 1; coordinates = cartesian; }
  field u: state scalar DG(order=0) on Line { time_role = differential; };
  parameter k0;
  constitutive k = k0 * (1 + u);
  equation balance on Line { dt(u) - div(k * grad(u)) = 0; }
}

// D6: a kink in k(u) refuses every state derivative of the diffusive face kernel.
model NonsmoothDiffusion1D {
  domain Line { dimension = 1; coordinates = cartesian; }
  field u: state scalar DG(order=0) on Line { time_role = differential; };
  parameter k0;
  constitutive k = k0 * (1 + abs(u));
  equation balance on Line { dt(u) - div(k * grad(u)) = 0; }
}

// SV9 E5 stage 1: conduction through two material regions on one FV mesh. `.res` has no volume
// sub-regions; the conductivity is a property whose provider product the realizer binds per
// cell from the case's region-to-material map (`CellDatum` on each side of a face).
model TwoRegionConduction2D {
  domain Slab { dimension = 2; coordinates = cartesian; }
  field T: state scalar DG(order=0) on Slab { time_role = differential; };
  provider conductivity(material: selector) -> ThermalConductivity;
  property k = conductivity(0);
  equation balance on Slab { dt(T) - div(k * grad(T)) = 0; }
  boundary hot on boundary("left") { dirichlet T = 1; }
  boundary cold on boundary("right") { dirichlet T = 0; }
  boundary insulated on boundary("top") { neumann T = 0; }
}

// Refusals: anti-diffusive sign, two diffusive divergences.
model AntiDiffusion1D {
  domain Line { dimension = 1; coordinates = cartesian; }
  field u: state scalar DG(order=0) on Line { time_role = differential; };
  parameter k;
  equation balance on Line { dt(u) + div(k * grad(u)) = 0; }
}

model DoubleDiffusion1D {
  domain Line { dimension = 1; coordinates = cartesian; }
  field u: state scalar DG(order=0) on Line { time_role = differential; };
  parameter k1;
  parameter k2;
  equation balance on Line { dt(u) - div(k1 * grad(u)) - div(k2 * grad(u)) = 0; }
}
