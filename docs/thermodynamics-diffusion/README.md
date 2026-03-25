# Heat Diffusion & Thermal Equation Solver

Technical specification of the parabolic partial differential equation (PDE) solver for thermal transport.

## 1. Physical Model
The module solves the **Heat Equation** for isotropic media. The evolution of the temperature field ($T$) is governed by the diffusion coefficient ($\alpha$), mapping the second-order spatial derivatives to the first-order temporal rate of change.

## 2. Numerical Scheme
The solver utilizes a **Finite Difference Method (FDM)** with an explicit time-integration scheme:
- **Laplacian Operator**: Computed via a 5-point stencil (2D) to approximate the local curvature of the temperature field.
- **Equilibrium Recovery**: Each integration step updates the local temperature based on the weighted mean of the adjacent neighborhood, strictly adhering to the laws of thermodynamics (heat flow from high to low potential).

## 3. Boundary Conditions
The simulation supports specialized boundary roles via the MasterBuffer orchestration:
- **Isothermal (Dirichlet)**: Nodes maintained at a constant temperature (e.g., heat sinks or sources).
- **Adiabatic (Neumann)**: Zero-flux boundaries representing perfect thermal insulation.

## 4. Implementation Details (WebGPU)
- **Kernel Orchestration**: The diffusion kernel is executed as a dedicated compute pass in WGSL, leveraging shared memory patterns where applicable to optimize stencil lookups.
- **Real-Time Audit**: Support for live parameter injection (thermal conductivity adjustment) without reset of the integration state.

## 5. Application Constraints
- **Stability Limit**: The diffusion rate is constrained by the numerical stability criterion ($\alpha \Delta t / \Delta x^2 \leq 0.25$ in 2D). Exceeding this threshold results in unphysical oscillation and numerical divergence.
