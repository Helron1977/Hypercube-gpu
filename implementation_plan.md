# Direct-Read Manifest Architecture (v9)

Goal: Eliminate host-side global workspace 'shadowing' to prevent collisions with grid data. All global results (Cd/Cl) will be extracted on-demand from a dedicated GPU buffer directly into local variables.

## Proposed Changes

### [Component] Hardware-Level Isolation (GPU Only)

#### [MODIFY] [MasterBuffer.ts](file:///c:/Users/rolan/OneDrive/Desktop/hypercube-gpu-core/src/memory/MasterBuffer.ts)
- **New GPU Buffer**: Add `public readonly gpuGlobalBuffer: GPUBuffer`.
- **Dedicated GPU Allocation**: Allocate `gpuGlobalBuffer` based on `DataContract.calculateGlobalBytes()`.
- **NO HOST SHADOW**: Remove all global workspace references from `rawBuffer`. The host-side grid memory remains pure and untouched by scalars.
- **Direct Global Access**: Refactor `getGlobalData()` to become `readGlobalData()`, which performs a direct, asynchronous read-back from the GPU into a local buffer.

#### [MODIFY] [GpuDispatcher.ts](file:///c:/Users/rolan/OneDrive/Desktop/hypercube-gpu-core/src/dispatchers/GpuDispatcher.ts)
- **Direct GPU Binding**: Update Binding 3 to use `buffer.gpuGlobalBuffer` starting at **offset 0**.

#### [MODIFY] [MemoryLayout.ts](file:///c:/Users/rolan/OneDrive/Desktop/hypercube-gpu-core/src/memory/MemoryLayout.ts)
- **Absolute GPU Offsets**: `getGlobalOffset()` now provides the address relative to the dedicated Global Workspace buffer on the GPU.

## Verification Plan

### Manual Verification
- `aerodynamics.html`:
    - Confirm dynamic Cd/Cl reporting via the direct read-back engine.
    - Confirm zero-interference between grid visualizations and force data.
