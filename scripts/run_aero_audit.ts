import { Aerodynamics } from '../tests/validation/Aerodynamics';

async function main() {
    console.log("Hypercube GPU - Aerodynamics Physical Validation Suite");
    const result = await Aerodynamics.runCylinderDragAudit();
    
    console.log("\n-------------------------------------------");
    if (result.status === 'stable') {
        process.exit(0);
    } else {
        console.error("Validation FAILED: Drag coefficient out of acceptable industrial range.");
        process.exit(1);
    }
}

main().catch(err => {
    console.error(err);
    process.exit(1);
});
