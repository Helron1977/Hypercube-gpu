# hypercube-gpu-core (v6.0.6)
**Scientific Audit Node for WebGPU Compute & Coupled Physics.**

A deterministic, high-concurrency architecture designed for Lattice Boltzmann (LBM), FDTD, and Poisson-Boltzmann numerical methods. The core implements a **Zero-Stall** memory model and a v9 **Direct-Read Manifest Architecture** to maximize effective VRAM bandwidth.

## v6.0.6 Highlight: Enterprise DX & Coupled Physics
The v6.0.6 release stabilizes the **Multi-Cube Support** for coupled physics domains sharing a single contiguous physical `GPUBuffer`. It introduces **Semantic Aliasing** (`SimulationParams`), **Custom EntryPoints**, and **Transient Execution** for one-shot physical impulses (e.g., heat injection, pressure bursts) with zero parity overhead.

## Technical Specifications
- **Synchronous MasterBuffer Layout**: Host-mirrored VRAM partitioning for low-latency state synchronization.
- **Batched Command Orchestration**: Minimal command encoder overhead via consolidated rule dispatching.
- **Zero-Stall Pipeline** : Asynchronous uniform updates and multi-rule compute passes.
- **Direct-Read Manifest (v9)** : Read individual global variables (Cd, Cl) without host shadowing.
- **Scientific Macro Engine (v6.0.6)** : Automated WGSL `read_NAME_Now(x, y)` and `write_NAME_Next(x, y, val)` helpers with **Idiomatic Semantic Aliasing** via `SimulationParams`.
- **Transient Dispatch (New)** : `executeTransient()` for one-shot compute passes without temporal parity rotation.
- **Dynamic Insertion** : `injectData()` for mid-simulation data manipulation without host read-back.

## Numerical Validation (v6.0 Certified)
- **Spatial Order**: Verified second-order spatial convergence ($O(\Delta x^2)$).
- **Physical Accuracy**: Drag coefficient ($C_D$) validated within 1.2% error on the Schäfer & Turek (1996) benchmark.
- **Computational Throughput**: 1042.12 MLUPS (D3Q19) recorded on modern GPU architecture.

### Reproducibility Suite
Absolute throughput and numerical precision can be verified through the integrated audit suite:
1. Initialize: `npm run dev`
2. Access the Hub: `http://localhost:5173/docs/index.html`

## Multi-Cube Usage (New in v6.0.1)

```typescript
import { createSimulation, linkSimulation, SharedMasterBuffer } from 'hypercube-gpu-core';

// 1. Create a large Shared Buffer (1GB)
const shared = new SharedMasterBuffer(1024 * 1024 * 1024);

// 2. Link Multiple Engines to the same physical memory
const fluidEngine = await linkSimulation(lbmManifest, shared);
const thermalEngine = await linkSimulation(heatManifest, shared);

// Both engines now share the same GPUBuffer using internal offsets.
// Kernels in 'thermalEngine' can read 'fluidEngine' data faces directly.
```

## Scientific Solver taxonomy
| Discipline | Model | Status |
| :--- | :--- | :--- |
| **Fluid Dynamics** | [LBM 2D/3D](./docs/fluid-dynamics-lbm/README.md) | **v6.0 CERTIFIED** |
| **Electromagnetics** | [FDTD Maxwell](./docs/electromagnetics-fdtd/README.md) | **v6.0 CERTIFIED** |
| **Potential Fields** | [Poisson Solver](./docs/numerical-physics/README.md) | **v6.0 CERTIFIED** |
| **Signal Physics** | [Wave Equation](./docs/numerical-physics/README.md) | **v6.0 CERTIFIED** |
| **AI Agents** | [Agent Guidelines](./agents/README.md) | **ACTIVE** |

## 🤖 AI Collaborator Support
AI agents developing with Hypercube should initialize by reading `agents/README.md` and parsing `docs/index.html` for technical specifications of the **Manifest DSL** and **WGSL Macro** system.

## License
MIT — Hypercube GPU Core v6.0.6 (Helron/Hypercube)
