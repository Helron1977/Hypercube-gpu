# Hypercube GPU Core v5.0.4 — Extensions Guide

Professional-grade extensions for high-performance multi-physics coupling and real-time control.

## 1. Professional Atomics (Binding 2)

Commonly used in reduction kernels (energy conservation, force integration), atomics in Hypercube are separated from standard data to ensure maximum performance for non-atomic operations.

### Atomic Face Types
- `atomic_u32`: Standard 32-bit unsigned integer accumulation.
- `atomic_f32`: High-precision float accumulation using a framework-injected **CAS (Compare-And-Swap) loop**.

### Usage in WGSL
The framework automatically generates `atomicAdd_faceName` and `atomicLoad_faceName` macros.

```wgsl
// Face name: 'energy_sum' (type: 'atomic_f32')

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let flux = calculate_flux(id.xy);
    
    // Transparent thread-safe accumulation
    // No bitcasting required by the user
    atomicAdd_energy_sum(id.x, id.y, flux);
}
```

> [!TIP]
> **Performance**: While atomics are thread-safe, they are computationally more expensive than standard writes. Use them only for reduction or rare collision scenarios.

---

## 2. Dynamic Control (`stepWithParams`)

Inject transients directly into the simulation step without modifying the static configuration. This is the foundation for high-frequency PID controllers or real-time UI interactions.

### API Usage
```typescript
// Static params in manifest: { tau: 0.55, ux0: 0.05 }
// p0 = 0.55, p1 = 0.05

// Transient override during step
await engine.step(kernels, { 
    tau: 0.85,  // Overrides p0
    ux0: 0.12   // Overrides p1
});
```

The overrides take precedence over the `config.params` values for the duration of the step.

---

## 3. Remote Kernel Loading (`loadKernel`)

Enable the "Plugin-First" ecosystem by fetching validated physics kernels from any remote repository.

```typescript
import { HypercubeGPUContext } from 'hypercube-gpu-core';

// Load remote source
const lbmSource = await HypercubeGPUContext.loadKernel(
    "https://cdn.hypercube.io/kernels/lbm_d2q9_v5.0.4.wgsl"
);

// Inject into rules
await engine.step({ 'collision': lbmSource });
```

> [!IMPORTANT]
> **Security**: Ensure remote sources are served over HTTPS and from trusted domains when building production applications.
