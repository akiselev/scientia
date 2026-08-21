# Local iteration ownership

Scientia's current `LocalFormProgram` is a point-local QFunction contract. One invocation
evaluates one integral contribution at one quadrature point. Accordingly,
`LocalIterationContract::QuadraturePoint` lowers to a Malleus `IterationDomain` with no axes;
Malleus defines that domain to execute exactly once.

This is not a placeholder for an implicit loop:

- Scientia identifies the integration domain or facet region and the values, gradients,
  time derivatives, or traces required at the selected point.
- Finitum selects the element, quadrature rule, quadrature point, basis/restriction data, and
  concrete extents in its realization plan. It maps the point QFunction across those choices.
- Malleus schedules and executes the fixed local iteration domain it is given. It does not infer
  a quadrature rule or a scientific integration domain.

A later batching optimization may introduce explicit fixed Malleus axes. Finitum must bind their
extents when constructing the realization plan, and the batched program must remain equivalent to
mapping this point contract. No persistent Scientia artifact uses an empty iterator list to mean
"quadrature loop to be supplied somehow later."
