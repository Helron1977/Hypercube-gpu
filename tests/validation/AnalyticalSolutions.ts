/**
 * Analytical Exact Solutions for LBM Validation
 */

export class AnalyticalSolutions {
    /**
     * Exact 2D Taylor-Green Vortex Solution for Navier-Stokes equations.
     * @param x X coordinate (continuous, 0 to nx)
     * @param y Y coordinate (continuous, 0 to ny)
     * @param t Current time in LBM steps
     * @param u0 Initial maximum velocity amplitude
     * @param nx Grid width
     * @param ny Grid height
     * @param nu Kinematic viscosity in LBM units
     * @returns { u, v, p } (Velocity X, Velocity Y, Pressure/Density variation)
     */
    static getTaylorGreenVortex2D(x: number, y: number, t: number, u0: number, nx: number, ny: number, nu: number) {
        const kx = 2.0 * Math.PI / nx;
        const ky = 2.0 * Math.PI / ny;
        const decay = Math.exp(-2.0 * nu * (kx*kx + ky*ky) * t);
        
        const u = -u0 * Math.cos(kx * x) * Math.sin(ky * y) * decay;
        const v =  u0 * Math.sin(kx * x) * Math.cos(ky * y) * decay;
        
        const p_decay = Math.exp(-4.0 * nu * (kx*kx + ky*ky) * t);
        const p = -(u0 * u0 / 4.0) * (Math.cos(2.0 * kx * x) + Math.cos(2.0 * ky * y)) * p_decay;
        const rho = 1.0 + 3.0 * p; // Using c_s^2 = 1/3
        
        return { u, v, p, rho };
    }

    /**
     * Exact 2D Poiseuille Flow (steady-state duct flow)
     * @param y Y coordinate (0 to ny, walls at y=0 and y=ny)
     * @param ny Grid height (distance between walls)
     * @param uMax Maximum velocity at the center of the duct
     * @returns { u }
     */
    static getPoiseuilleFlow2D(y: number, ny: number, uMax: number) {
        // Parabolic profile: u(y) = 4 * uMax * (y/H) * (1 - y/H)
        // With H = ny (if boundaries are exactly on nodes 0 and ny)
        const H = ny;
        const u = 4.0 * uMax * (y / H) * (1.0 - y / H);
        return { u };
    }
}
