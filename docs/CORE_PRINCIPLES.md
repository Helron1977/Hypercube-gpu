# @hypercube/gpu-core — Development Principles

These principles govern all contributions and architectural evolution of the Hypercube GPU framework.

## 1. Absolute Rigor & Scientific Honesty
- **No Marketing Fluff**: Communication must remain factual, neutral, and technical. The project is an engineering utility, not a promotional asset.
- **Realistic Benchmarking**: Performance comparisons must be conducted against verified state-of-the-art implementations (e.g., FluidX3D, waLBerla) using identical hardware and boundary conditions.

## 2. Fundamental Stability (Bottom-Up Construction)
- **Zero Shortcuts**: Architectural integrity takes precedence over quick feature deployment. Every layer must be verified before proceeding to the next.
- **Deterministic Methodology**: Code modifications are driven by technical necessity and verified theory. Heuristic or randomized "guessing" is prohibited.

## 3. Verification-Driven Development (VDD)
- **Mandatory Validation**: No feature is considered complete without a corresponding, reproducible validation test (Unit, Integration, or Physical Audit).
- **Zero Regression Tolerance**: All tests (Unit, Performance, Conservation) must be re-executed at every iteration. Regression is a critical failure.

## 4. Decoupled Scientific Proofs
- **Autonomous Audit Scripts**: Rendering test scripts (`tests/renders/*.ts`) must remain "raw" and 100% autonomous. No refactoring for the sake of brevity if it compromises their role as independent "Raw Proof" evidence.
- **Independence Over Elegance**: In the validation layer, mathematical transparency and decoupling take precedence over code reuse.

## 5. Professionalism & Precision
The primary objective is the development of the most robust and high-performing GPU simulation core within the constraints of WebGPU. All documentation, comments, and logs must reflect this commitment to precision.

## 6. Technical Integration Principles
- **Vitest Environment Isolation**: Systematically exclude browser-resident rendering directories (`tests/renders/`) from Node.js test environments to prevent runtime conflicts.
- **WebMCP Automation Compliance**: All new visual audits must implement the standard HTTP reporting boilerplate to ensure compatibility with the automated `run_webgpu_suite` bridge.
- **High-Fidelity Documentation**: Maintain and evolve the `ARCHITECTURE.md` and ADR registry as professional sources of truth, using isometric and high-fidelity visuals for complex orchestration logic.
