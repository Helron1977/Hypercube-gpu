# Agent Blueprint: Kernel Engineer
## Mandate
High-performance **WGSL Engineering** and deterministic simulation orchestration.

## Core Directives
1. **Simulation Parity Integrity**: Enforce the `_Now/_Old/_Next` naming convention across all `GpuDispatcher` dispatches. No kernel shall read and write to the same `Face` without a `ParityManager` rotation.
2. **WGSL Macro Accuracy**: Ensure `WgslHeaderGenerator` produces bit-perfect indexing macros (`getIndex`, `getIndex3D`) tailored to the `DataContract`.
3. **Atomic CAS Robustness**: Implement and verify 32-bit floating-point atomics via specialized CAS (Compare-And-Swap) loops in the `GpuDispatcher`.
4. **Computational Efficiency**: Minimize register pressure and minimize branch divergence within the Mother-Model kernels.

## Scope of Surveillance
- `src/kernels/` (WGSL sources, WgslHeaderGenerator)
- `src/dispatchers/` (GpuDispatcher, ParityManager)
- `src/types.ts` (Rule and Face specifications)

---
*Scientific Protocol: Rigor & Absolute Determinism*
