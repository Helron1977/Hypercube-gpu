# hypercube-gpu-core
**Direct WebGPU Compute Core for Scientific Research & Industrial Simulation.**

A deterministic, high-concurrency architecture designed for Lattice Boltzmann (LBM), FDTD, and Poisson-Boltzmann numerical methods. The core implements a zero-copy memory model to maximize effective VRAM bandwidth.

## Technical Specifications
- **Synchronous MasterBuffer Layout**: Host-mirrored VRAM partitioning for low-latency state synchronization.
- **Batched Command Orchestration**: Minimal command encoder overhead via consolidated rule dispatching.
- **Zero-Stall Pipeline**: Asynchronous uniform updates and compute passes.
- **Scientific Kernel Registry**: Native implementations for Navier-Stokes (LBM D2Q9/D3Q19) and Wave Equations (FDTD).

## Numerical Validation
- **Spatial Order**: Verified second-order spatial convergence ($O(\Delta x^2)$) via Taylor-Green Vortex (TGV) study.
- **Physical Accuracy**: Drag coefficient ($C_D$) validated within 1.2% error on the Schäfer & Turek (1996) cylinder benchmark at $Re=100$.
- **Computational Throughput**: 937.12 MLUPS (D3Q19) recorded on NVIDIA RTX 2080 architecture.

### Reproducibility Suite
Absolute throughput and numerical precision can be verified through the integrated audit suite:
1. Initialize the compute server: `npm run dev`
2. Access the formal audit interface: `http://localhost:5173/benchmark.html`

### Comparative Performance Analysis (Ref: RTX 2080)
| Implementation | Environment | MLUPS (Sustained) |
| :--- | :--- | :--- |
| **FluidX3D** (CUDA) | Native (C++/CUDA) | `3000 - 8000` |
| **Hypercube** (Zero-Stall) | **WebGPU (Browser)** | **`1042`** |
| **WebGPU Reference** (2023) | WebGPU (Browser) | `400 - 800` |
| **PyLBM** (Python) | Native (CPU) | `5 - 50` |

*Technical Note: The 1042 MLUPS baseline is a sustained stress-test result recorded on an NVIDIA RTX 2080. It represents ~35% VRAM bandwidth efficiency.*


## Scientific Solver Taxonomy (Mother-Models)
The core framework provides a library of validated "Mother-Models". Refer to the [Technical Documentation Hub](./docs/README.md) and our [User Quickstart](./docs/USER_QUICKSTART.md).

| Discipline | Model | Methodology | Status |
| :--- | :--- | :--- | :--- |
| **Fluid Dynamics** | [LBM 2D/3D](./docs/fluid-dynamics-lbm/README.md) | Lattice Boltzmann / D2Q9-D3Q19 | **CERTIFIED** |
| **Electromagnetics** | [FDTD Maxwell](./docs/electromagnetics-fdtd/README.md) | Leapfrog Yee-Cell | **CERTIFIED** |
| **Potential Fields** | [Poisson Solver](./docs/numerical-physics/README.md#1-poisson--laplace-champs-de-potentiel) | Iterative Jacobi | **CERTIFIED** |
| **Data Science** | [Tensor-CP (Lite/Pro)](./docs/tensor-cp/README.md) | ALS / CORCONDIA Diagnostics | **CERTIFIED** |
| **Chemistry/Bio** | [Diffusion](./docs/thermodynamics-diffusion/README.md) | Isotropic Heat Equation | **CERTIFIED** |
| **Complex Systems** | [Cellular Life](./docs/biology-cellular-life/README.md) | Parallel Automata | **CERTIFIED** |
| **Geometry** | [JFA Fields](./docs/distance-fields-jfa/README.md) | Jump Flooding Algorithm | **CERTIFIED** |
| **Signal Physics** | [Wave Equation](./docs/numerical-physics/README.md#2-l’équation-donde-2d) | Advection/Diffraction | **CERTIFIED** |
| **Synth. Assets** | [Fractals/Noise](./docs/numerical-physics/README.md#3-fractals--mandelbulb-3d) | Ray-marching / Simplex | **CERTIFIED** |

---

## Release History
See [CHANGELOG.md](./CHANGELOG.md) for full details of the **v5.0.0** Major Release.

## Installation & Usage
```bash
npm install @hypercube/gpu-core 
```

## Usage

```typescript
import { GpuCoreFactory, HypercubeGPUContext } from '@hypercube/gpu-core';

// 1. Initialize GPU Context
await HypercubeGPUContext.init();

// 2. Build Engine from Manifest
const factory = new GpuCoreFactory();
const engine = await factory.build(config, descriptor);

// 3. Execution Loop
async function loop() {
    await engine.step(1, { 
        'lbm-ocean': oceanWgslSource 
    });
    
    // Optional: Synchronize specific data for HUD/Viz
    await engine.buffer.syncFacesToHost(['rho', 'vx']);
    
    requestAnimationFrame(loop);
}
```

## Project Structure
- `src/memory`: MasterBuffer and memory orchestration.
- `src/dispatchers`: GpuDispatcher and pipeline management.
- `src/topology`: VirtualGrid, Joints, and Topology Resolution.
- `src/kernels`: Reference WGSL implementations.
- `src/GpuEngine.ts`: The unified simulation interface.

## License
MIT — Hypercube GPU Core v5.0.0
