# Hypercube GPU Core — Technical Documentation Hub

This hub serves as the primary technical reference for the Hypercube multi-physics simulation engine. It documents the underlying mathematical frameworks, GPU execution patterns, and physical mother-models that constitute the framework's core.

---

## Architectural Core

The engine is built on a high-concurrency, zero-copy architecture designed for maximum throughput and numerical stability.

### [GpuDispatcher](../src/dispatchers/GpuDispatcher.ts)
The central execution brain. Handles pipeline management, parity rotation (Ping-Pong), and multi-face dispatch contracts. It enforces strict data alignment between the TypeScript application and WGSL kernels.

### [MasterBuffer](../src/MasterBuffer.ts) & [MemoryLayout](../src/MemoryLayout.ts)
Universal VRAM management. All simulations operate on unified memory buffers, eliminating CPU-GPU transfer bottlenecks during iterative solver execution.

### [ParityManager](../src/ParityManager.ts)
Ensures temporal coherence for iterative algorithms. Manages the swap-chain logic for `Read/Write` face suffixes, preventing race conditions in state-dependent simulations.

---

## Solver Taxonomy (Mother-Models)

The core framework provides a library of validated "Mother-Models" representing the state-of-the-art in numerical physics.

| Discipline | Model | Methodology | Status |
| :--- | :--- | :--- | :--- |
| **Fluid Dynamics** | [LBM 2D](./fluid-dynamics-lbm/README.md) | Lattice Boltzmann / D2Q9 | **CERTIFIED** |
| **Electromagnetics** | [FDTD Maxwell](./electromagnetics-fdtd/README.md) | Leapfrog Yee-Cell | **CERTIFIED** |
| **Potential Fields** | [Poisson Solver](./numerical-physics/README.md#1-poisson--laplace-champs-de-potentiel) | Iterative Jacobi | **CERTIFIED** |
| **Data Science** | [Tensor-CP ALS](./tensor-cp/README.md) | Alternating Least Squares | **CERTIFIED** |
| **Chemistry/Bio** | [Diffusion](./thermodynamics-diffusion/README.md) | Isotropic Heat Equation | **CERTIFIED** |
| **Complex Systems** | [Cellular Life](./biology-cellular-life/README.md) | Parallel Automata | **CERTIFIED** |
| **Geometry** | [JFA Fields](./distance-fields-jfa/README.md) | Jump Flooding Algorithm | **CERTIFIED** |
| **Signal Physics** | [Wave Equation](./numerical-physics/README.md#2-l’équation-donde-2d) | Advection/Diffraction | **CERTIFIED** |
| **Synth. Assets** | [Fractals/Noise](./numerical-physics/README.md#3-fractals--mandelbulb-3d) | Ray-marching / Simplex | **CERTIFIED** |

---

## Verification & Validation (V&V)

Mathematical integrity is non-negotiable. Every model is audited against analytical solutions.

- **Scientific Proofs**: Consult [PROOFS.md](./validation/PROOFS.md) for empirical error analysis and conservation logs.
- **Audit Hub**: Live visual verification suite available at [tests/renders/index.html](../tests/renders/index.html) (WebGPU context required).

---
*Hypercube GPU Framework — Build v4.7.0 — Scientific Integrity First.*
