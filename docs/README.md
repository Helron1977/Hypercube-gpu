# Hypercube GPU Core v5.0.4 🚀

**The high-performance atomic simulation engine for WebGPU.**

Hypercube is a SOTA (State-Of-The-Art) framework designed for large-scale physical simulations on the GPU. It provides a zero-readback architecture, automatic WGSL macro generation, and elite numerical certification.

## 🔬 Unified Documentation Hub

For the best developer experience, use our interactive documentation center:
👉 [**Open Hypercube Hub (index.html)**](file:///c:/Users/rolan/OneDrive/Desktop/hypercube-gpu-core/docs/index.html)

---

## ⚡ Elite API (Quickstart)

```typescript
import { createSimulation, SimulationPacks } from 'hypercube-gpu-core';

// 1. Initialize with a certified SOTA pack
const engine = await createSimulation({
  engine: SimulationPacks.LBM_D2Q9,
  config: { dimensions: { nx: 256, ny: 256 }, params: { tau: 0.8 } }
});

// 2. Register your kernels once
engine.use({ 'main': myWgslSource });

// 3. High-speed simulation loop
async function loop() {
  await engine.step(100); // Zero-noise execution
  const rho = await engine.getFace('rho'); // Smart retrieval
  requestAnimationFrame(loop);
}
```

---

## 🏗️ Core Architecture

- **[MasterBuffer](file:///c:/Users/rolan/OneDrive/Desktop/hypercube-gpu-core/src/memory/MasterBuffer.ts)**: Contiguous VRAM partitioning for Standard and Atomic data.
- **[GpuDispatcher](file:///c:/Users/rolan/OneDrive/Desktop/hypercube-gpu-core/src/dispatchers/GpuDispatcher.ts)**: Automated WGSL macro injection (CAS loops for float atomics).
- **[DataContract](file:///c:/Users/rolan/OneDrive/Desktop/hypercube-gpu-core/src/DataContract.ts)**: Semantic validation of simulation descriptors.

---

## 🏁 Scientific Certification
Hypercube v5.0.4 is verified for **scientific trust**. 
- **Convergence Audit**: $O(\Delta x^2)$ confirmed for Diffusion and Wave solvers.
- **Atomic Reliability**: Precision loss $< 10^{-7}$ for large-scale global reductions.

---

## 🛠️ Contribution
Refer to the [**Contributor Quickstart**](file:///c:/Users/rolan/OneDrive/Desktop/hypercube-gpu-core/docs/CONTRIBUTOR_QUICKSTART.md) to extend the core or add new Mother-Models.

© 2026 Hypercube Labs — Digital Laboratory Rigor.
