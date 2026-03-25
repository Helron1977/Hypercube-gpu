# ADR 001: Unified MasterBuffer vs Ad-hoc Buffers

## Status
Accepted (Phase 1)

## Context
Initial implementations of GPU physics often use individual textures or small buffers for each variable (e.g., one texture for Velocity, one for Density). This leads to high draw-call overhead and complex synchronization.

## Decision
All simulation data shall reside in a single **Unified MasterBuffer**. This buffer is partitioned into chunks and faces using bitwise offsets calculated by the `MemoryLayout`.

## Consequences
- **Positive**: Zero-copy execution. Minimal command-buffer overhead.
- **Positive**: Guaranteed deterministic memory access for massively parallel kernels.
- **Negative**: Increased complexity in the initial `VirtualGrid` setup.
- **Neutral**: Requires strict alignment management (256-byte uniform blocks).
