# Changelog — Hypercube GPU Core

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [5.0.0] - 2026-03-28

### 🚀 Major Architectural Shift (Standalone Core)
This version marks the official separation of the GPU Compute Core from the monolithic Hypercube framework.

### Added
- **Generic Typings** : `GpuEngine<TParams, TFaces>` now provides full TypeScript autocompletion and type safety for simulation parameters and data faces.
- **WGSL Header Generator (v2)** : Automatic generation of `read_NAME(x, y)` and `write_NAME(x, y, val)` macros, eliminating manual index calculations in shaders.
- **Simulation Builder** : Fluent API for declarative simulation setup (`SimulationBuilder`).
- **Simulation Packs** : Pre-configured presets for LBM, Wave, Poisson, and Cellular Automata.
- **JSON Schema** : Official schema for `HypercubeManifest` providing IDE validation and autocompletion.
- **Triple-Buffering** : Native support for 2nd-order wave simulations via `ParityManager` modulo 3.
- **Ghost Cell Stripping** : `getInnerFaceData()` helper for seamless integration with rendering engines.
- **Unified Lifecycle** : `ready()` promise ensuring GPU buffers are initialized before usage.

### Changed
- **NPM Scoping** : Re-scoped to `@hypercube/gpu-core`.
- **Parameter Proxying** : Direct access to parameters via `engine.params`.
- **Expanded Capacity** : Uniform slots increased from 16 to 64 to support massive 3D models.
- **Performance Guideline** : Documentation of the **Zero-Readback** philosophy (keeping data in VRAM for rendering).

### Fixed
- Fixed memory alignment issues in uniform buffer updates.
- Resolved ghost cell synchronization synchronization race conditions.
- Improved error messaging for buffer partitioning.

---

## [4.7.0] - 2026-03-20
*Baseline: Scientific Integrity Certification.*
- Stable LBM D2Q9 execution.
- Verified Taylor-Green Vortex (TGV) study.
- Initial MasterBuffer implementation.
