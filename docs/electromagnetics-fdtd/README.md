# FDTD Maxwell Solver: Electromagnetics Implementation

Technical specification of the Finite-Difference Time-Domain (FDTD) solver for electromagnetic wave propagation.

## 1. Algorithmic Foundation
The solver implements the **Yee Algorithm** for solving Maxwell's equations in the time domain. It utilizes a staggered grid approach where:
- **Electric Field ($\vec{E}$)** and **Magnetic Field ($\vec{H}$)** are computed at alternating time steps (leapfrog integration).
- Central difference approximations are applied to spatial and temporal derivatives, ensuring second-order accuracy.

## 2. Boundary Conditions: Absorbing Boundary Layers (ABC)
To simulate open-space propagation within a finite computational domain, the engine utilizes **Absorbing Boundary Conditions (ABC)**:
- **Sponge Layers**: A boundary region (typically 12-24 pixels) where field amplitudes are attenuated via a spatially varying conductivity profile.
- **Role Specification**: Boundary nodes are assigned the `absorbing` role in the configuration manifest to neutralize reflections and prevent wave-mirroring artifacts.

## 3. Excitation & Sources
- **Soft Source Injection**: Electric field values are programmatically incremented at specified coordinates to excite the Yee grid.
- **Harmonic Oscillation**: Supports continuous-wave (CW) sources for frequency-domain analysis.

## 4. Visualization & Diagnostics
- **Bi-polar Field Mapping**: Visualization of the $E_z$ or $H_z$ component where magnitude and phase are represented via a divergence color scale.
- **Field Intensity**: Real-time computation of energy density for link-budget and interference analysis.

## 5. Technical Specification
- **Precision**: f32 (Single Precision) floating-point arithmetic.
- **Courant Stability (CFL)**: The time step ($\Delta t$) is automatically constrained by the lattice spacing ($\Delta x$) to ensure numerical stability at the speed of light.
