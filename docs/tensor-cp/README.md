# Tensor-CP: Canonical Polyadic Decomposition

Technical specification of the Tensor-CP implementation for high-dimensional data analysis.

## 1. Algorithmic Foundation
The module implements the **Canonical Polyadic (CP) Decomposition**, also known as CANDECOMP/PARAFAC. The primary optimization routine uses a **Hybrid Alternating Least Squares (ALS)** approach:
- **Matrix Inversion**: Executed via the Javascript V8 engine for numerical stability in small-matrix operations.
- **Khatri-Rao Product & Contraction**: Orchestrated via WebGPU compute kernels to accelerate the heavy-lifting of large-scale tensor products.

## 2. Data Structures: Sparse Tensors
The engine natively supports **Sparse Tensors**, adhering to the **FROSTT/TNS (.tns)** standard. 
- **Coordinate Format (COO)**: Only non-zero elements are stored and processed, allowing for the analysis of high-dimensional datasets that would otherwise exceed VRAM limits.
- **Scalability**: The system is designed to handle up to $10^6$ non-zero elements within the standard WebGPU memory heap limitations.

## 3. Hardware Orchestration (WebGPU)
Unlike standard CPU-based tensor libraries, Hypercube-gpu-core utilizes a zero-copy memory pattern between the tensor update phases. This minimizes host-device synchronization latency, a critical bottleneck in iterative ALS routines.

## 4. Operational Constraints
- **Memory Scaling**: While sparse formats mitigate cubical memory growth ($I \times J \times K$), dense tensor representations are strictly limited by the WebGPU 4GB heap constraint.
- **Convergence Protocol**: The solver monitors the **Fit-Error** and **Residual Norm** per iteration. Users are expected to calibrate the **Rank ($R$)** based on the observed approximation quality.

## 5. Export & Integration
Factor matrices are exported in standard **JSON (Export Factors)** format, allowing for direct downstream integration into specialized analytics pipelines.
