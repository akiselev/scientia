# Re-requests from lane `scientia-systems` (batch 13)

Built against scientia `84188ac` (lane branch `claude/b13-scientia-systems`, sha `f411354`),
sinbad corpus read at `fb11d4a`.

## R1 (sinbad): make `physics/corpus/28-compressible-euler.res` lowerable without Euler Rust

The corpus declares `provider heat_capacity_ratio(...)`, `provider identity(dimension:
selector) -> Dimensionless { shape = tensor(3,3); }` and `provider outer(momentum, velocity) ->
Pressure { shape = tensor(3,3); }`, and uses them in `pressure`, `momentum_flux` and `gamma`.
Scientia's FV lowering refuses state-dependent provider calls (`FV_FLUX_PROVIDER_UNSUPPORTED`,
by design: a provider's meaning is external), so the model can be *read* as a system but cannot
*lower*. Scientia cannot add `outer`/`identity` intrinsics while the corpus declares providers of
those names (`provider ... cannot reuse the reserved intrinsic function name` would fail
Sinbad's `tests/fc10_methods.rs`, which compiles this file).

Requested, either:
- (a) Corpus rewrite with existing language only: `parameter gamma;` (ideal gas), `momentum_flux`
  as a row-major tensor literal of components (`[[m0*v0 + p, m0*v1, m0*v2], ...]`, typing added
  on this branch), and `@wave_speeds(absolute_maximum = ..., minimum = ..., maximum = ...)` as
  described in `docs/notes/batch13-reports/scientia-systems.md` §8; or
- (b) a coordinated step: Sinbad drops the `provider outer`/`provider identity` declarations in
  the same consolidation in which Scientia adds `outer(a, b)` and `identity(n)` intrinsics
  (a separate Scientia lane item; not on this branch). `gamma` still needs (a)'s `parameter`.

Gate it unblocks: the SV2-C1b goal "corpus Euler reads, lowers and checks" and the later
Sinbad Euler acceptance cases (SV9 F1 stage 1).
