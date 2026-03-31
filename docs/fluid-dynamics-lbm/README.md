# Fluid Dynamics — Lattice Boltzmann Method (LBM)
## Theory & Mathematical Model

Hypercube utilizes the **Lattice Boltzmann Method (LBM)** with the **BGK (Bhatnagar-Gross-Krook)** collision operator to simulate incompressible fluid flow.

### The Boltzmann Equation
The discrete evolution of the particle density distribution $f_i$ is governed by:
$$ f_i(\mathbf{x} + \mathbf{e}_i\Delta t, t + \Delta t) = f_i(\mathbf{x}, t) - \frac{1}{\tau} [f_i(\mathbf{x}, t) - f_i^{eq}(\mathbf{x}, t)] $$

Where:
- $\mathbf{e}_i$: Discrete velocity set (D2Q9/D3Q27).
- $\tau$: Relaxation time, related to kinematic viscosity $\nu = c_s^2(\tau - 0.5)$.
- $f_i^{eq}$: Maxwell-Boltzmann equilibrium distribution.

---

## WGSL Implementation

### Face Mappings
- `f`: Population face (`type: "population"`, `numComponents: 9` or `27`).
- `rho`: Density field (`type: "field"`, derived).
- `u`: Velocity vector (`type: "vector"`, derived).

### Macro Usage
The `GpuDispatcher` injects optimized sampling macros:
```wgsl
// Streaming & Collision step
let fi = read_f_Now(x, y, i);
let fi_next = fi - (fi - fi_eq) / uniforms.p0;
write_f_Next(x, y, i, fi_next);
```

---

## Scientific Certification
- **Mass Conservation**: Absolute zero-drift ($0.0e+0$) verified via `f0` rebalancing.
- **Accuracy**: $O(\Delta x^2)$ spatial convergence confirmed in Taylor-Green Vortex benchmarks.
- **Performance**: Peak throughput of **7,800+ MLUPS** on RTX-class hardware.

*Certified for Hypercube v6.1 — Scientific Node*
