# Hypercube GPU Core: Agent Onboarding Guideline
*Instructions for AI Agents consuming or developing with the Hypercube GPU Core.*

## 1. Documentation Source of Truth (docs/index.html)
The `docs/index.html` file serves as the centralized **Developer Experience (DX) Hub**.
- **Mandatory Action**: Agents MUST read/parse `docs/index.html` to understand the full architectural details, Manifest DSL, and the WGSL Header injection system.
- **Visual Auditing**: Because Hypercube relies on hardware-backed WebGPU renders, agents MUST ask the Human User to verify the dashboard and browser console outputs for any visual regressions or shader validation errors.

## 2. High-Level Language (Manifest DSL)
Hypercube simulations are defined declaratively via a `SimulationManifest`.
- **Faces**: Discrete data layers in VRAM (Standard, Population, Mask, Atomic). Memory layout is deterministic and handled by the `DataContract`.
- **Rules**: Sequence of compute kernels executed per simulation tick. All kernels use a shared physical `GPUBuffer`.
- **Globals**: Simulation-wide shared variables (e.g., $C_D$, $C_L$, Energy totals) stored in a dedicated `gpuGlobalBuffer`.

## 3. The Automated Macro System
Hypercube injects an automated header into every WGSL kernel. **DO NOT** use raw buffer offsets.
- **Reading Data**: 
  - `read_FACENAME_Now(x, y)` for scalars/masks.
  - `read_FACENAME_Now(x, y, component)` for multi-scalar populations.
- **Writing Data**:
  - `write_FACENAME_Next(x, y, value)` (if `pingPong: true`).
  - `write_FACENAME_Now(x, y, value)` (if `pingPong: false`).
- **Atomic Operations**:
  - `atomicAdd_FACENAME_Now(x, y, value)`.
  - `atomicAdd_global_GLOBNAME(index, value)`.

## 4. Operation Lifecycle
1. **Instantiation**: `createSimulation(manifest)` validates the contract and allocates the `MasterBuffer`.
2. **Registration**: `engine.use({ ruleName: wgslSource })` pre-compiles GPU pipelines.
3. **Execution**: `engine.step(n)` invokes the `GpuDispatcher` for $n$ iterations with zero host-to-device communication overhead.
4. **Data Retrieval**: `engine.getFace('name')` or `engine.getGlobal('name')` for asynchronous, host-shadow-free results.

## 5. Topology & Scale Management
Simulation size is determined by two synchronized parameters:
- **`dimensions`**: The logical size of the grid (`nx`, `ny`, `nz`).
- **`chunks`**: Physical partitioning of the grid. If `chunks.x > 1`, the grid is split into multiple independent areas for better scaling and multi-GPU compatibility.
- **Rule**: `nx` must be a multiple of `chunks.x`, and so on.

## 6. Multi-Engine & Side-by-Side Simulations
To have multiple simulations sharing memory (Coupled Physics):
1. **`SharedMasterBuffer`**: Allocate a single, large VRAM block.
2. **`linkSimulation(manifest, shared)`**: Link multiple engines to the same physical buffer. Each engine gets a unique `offset` but shares the same memory space.
3. **`MultiPhysicsHub`**: Use the hub to merge multiple `EngineDescriptors` (faces, rules, globals) into a single unified simulation if tight coupling (e.g., fluid/thermal) is required.

---
*Scientific Protocol: Rigor & Absolute Clarity*
