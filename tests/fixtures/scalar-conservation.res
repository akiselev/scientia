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
