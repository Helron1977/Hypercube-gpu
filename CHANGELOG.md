# Changelog — Hypercube GPU Core

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [5.0.1] - 2026-03-28

### Fixed
- **Anti-Drift (Temporal)** : Resolved "sliding" visual artifacts in static kernels by decoupling `uniforms.t` from the default evaluation logic. Added `p2` time-multiplier parameter to `FunctionCore` for opt-in animation.
- **Anti-Drift (Spatial)** : Verified `strideRow` alignment between GPU indexing and JS rendering across all ghost cell configurations (0, 1).
- **Coordinate Stability** : Implemented a comprehensive Anti-Drift test suite (`tests/unit/gpu_v5_drift.test.ts`) to prevent future regressions in grid indexing.

## [5.0.0] - 2026-03-28

### 🚀 Major Architectural Shift (Standalone Core)
This version marks the official separation of the GPU Compute Core from the monolithic Hypercube framework.

### Added
- **Generic Typings** : `GpuEngine<TParams, TFaces>` now provides full TypeScript autocompletion and type safety for simulation parameters and data faces.
- **Professional Macro Engine (v5.0)** : Automatic, high-level `read_NAME(x, y)` and `write_NAME(x, y, val)` helpers for all 64 potential faces. No more manual index math.
- **Native 3D Logic** : Includes `getIndex3D(x, y, z)` and corresponding 3D macros (`read3D_NAME`).
- **Simulation Builder** : Fluent API for declarative simulation setup (`SimulationBuilder`).
- **Simulation Packs** : Pre-configured presets for LBM, Wave, Poisson, and Cellular Automata.
- **JSON Schema** : Official schema for `HypercubeManifest` providing IDE validation and autocompletion.
- **Triple-Buffering** : Native support for 2nd-order wave simulations via `ParityManager` modulo 3.
- **Ghost Cell Stripping** : `getInnerFaceData()` helper for seamless integration with rendering engines.
- **Unified Lifecycle** : `ready()` promise ensuring GPU buffers are initialized before usage.

### Changed
- **NPM Integration** : Packaged as `hypercube-gpu-core` (unscoped) for consistent registry history.
- **Parameter Proxying** : Direct access to parameters via `engine.params`.
- **Expanded Capacity** : Uniform slots increased from 16 to 64 to support massive 3D models.
- **Uniform Layout Shift** : Roles moved to **offset 80** to accommodate 64 faces without data corruption.
- **Performance Guideline** : Documentation of the **Zero-Readback** philosophy (keeping data in VRAM).

### Fixed
- **Uniform Guard** : Fixed a critical bug where boundary roles were overwriting face indices. Moved roles/ghosts to offset 80+.
- **Coordinate Shift** : Fixed manual index math in `FunctionCore` by standardizing on `getIndex(x, y)` and `getIndex3D(x, y, z)` macros.
- **Shader Compatibility** : Standardized on a single `Uniforms` struct, eliminating shader compilation errors caused by redundant `struct Params` definitions.
- **DataContract Sync** : Fixed bug where `numComponents` was ignored in memory allocation.
- **D3Q27 Stability** : Verified 27-velocity model for 3D LBM with new unit tests.

---

## [4.7.0] - 2026-03-20
*Baseline: Scientific Integrity Certification.*
- Stable LBM D2Q9 execution.
- Verified Taylor-Green Vortex (TGV) study.
- Initial MasterBuffer implementation.
