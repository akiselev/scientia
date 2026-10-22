# Re-requests from lane `scientia-diffusion` (built against scientia `84188ac` → `f5e5919`)

Provisional (the lane is half-done; see the handoff report).

1. **Sinbad**: `FvMethodIntent` gains `diffusion: DiffusionIntent` (`[discretization]
   diffusion = { averaging = "harmonic" | "arithmetic" }`, default harmonic, serde-omitted when
   default). Every literal `FvMethodIntent { .. }` needs the field; the `[discretization]` parser
   should accept the optional table. `VerificationObligationKind` gains
   `DiffusiveFluxDissipativity { equation }` and `GridPeclet { field }` for its exhaustive
   matches. Unblocks: compiling Sinbad against this lane's head.
2. **Finitum** (next phase): bind `FvOperandMeaning::FaceNormalGradient`, `CellDatum { datum,
   side }`, `LimiterBound`, `LimiterIncrement`, `CellSize` and run `FvKernelRole::
   DiffusiveFaceFlux` / `DiffusionCoefficient` as described in the report §4.
