/**
 * Convergence Metrics and Error Calculation for Grid Refinement Studies
 */

export class Metrics {
    /**
     * Computes the relative L2 error norm between two fields.
     * E_L2 = sqrt( sum((u_c - u_e)^2) / sum(u_e^2) )
     * @param computed The computed array of values (e.g. from GPU buffer)
     * @param exact The exact analytical array of values
     * @param length The number of elements to compare
     */
    static computeL2Error(computed: Float32Array | number[], exact: Float32Array | number[], length: number): number {
        let diffSqrSum = 0.0;
        let exactSqrSum = 0.0;
        
        for (let i = 0; i < length; i++) {
            const diff = computed[i] - exact[i];
            diffSqrSum += diff * diff;
            exactSqrSum += exact[i] * exact[i];
        }

        if (exactSqrSum === 0) return 0;
        return Math.sqrt(diffSqrSum / exactSqrSum);
    }

    /**
     * Computes the order of convergence between two grid resolutions.
     * Order = ln(Error_coarse / Error_fine) / ln(Resolution_fine / Resolution_coarse)
     * @param errorCoarse L2 error on the coarse grid
     * @param errorFine L2 error on the fine grid
     * @param resCoarse Size of the coarse grid (e.g. 32)
     * @param resFine Size of the fine grid (e.g. 64)
     */
    static computeConvergenceOrder(errorCoarse: number, errorFine: number, resCoarse: number, resFine: number): number {
        return Math.log(errorCoarse / errorFine) / Math.log(resFine / resCoarse);
    }
}
