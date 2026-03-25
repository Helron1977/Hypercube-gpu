import http from 'http';
import { exec } from 'child_process';
import fs from 'fs';
import path from 'path';

/**
 * Hypercube Autonomous WebGPU Test Suite Runner
 * --------------------------------------------
 * This script orchestrates the execution of multiple scientific audit pages in a browser environment.
 * It acts as a bridge between the WebGPU-enabled browser and the Node.js filesystem for result logging.
 * 
 * Flow:
 * 1. Starts a local HTTP server on port 3000 to listen for test results.
 * 2. Iteratively launches browser instances for each test URL defined in ALL_TESTS.
 * 3. Captures the 'X-Test-Name' header and body payload from the browser's fetch() call.
 * 4. Persists the raw results as .log files in docs/validation/reports/.
 */
const REPORTS_DIR = path.join(process.cwd(), 'docs', 'validation', 'reports');

if (!fs.existsSync(REPORTS_DIR)) {
    fs.mkdirSync(REPORTS_DIR, { recursive: true });
}

// Ordered list of tests to run
const ALL_TESTS = [
    { name: 'SOTA MLUPS Benchmark', id: 'perf-test', url: 'http://localhost:5173/tests/renders/benchmark.html' },
    { name: 'Mass & Momentum Conservation Audit', id: 'conservation-audit', url: 'http://localhost:5173/tests/renders/conservation_audit.html' },
    { name: 'Unified Scientific Audit (LBM+FDTD+CP)', id: 'scientific-audit', url: 'http://localhost:5173/tests/renders/scientific_audit.html' },
    { name: 'Von Karman Vortex Street', id: 'von-karman', url: 'http://localhost:5173/tests/renders/von_karman.html' },
    { name: 'Multi-Chunk Halo Sync', id: 'multichunk-test', url: 'http://localhost:5173/tests/renders/verify_multichunk.html' },
    { name: 'LDC Stability Audit (Re=1000)', id: 'ldc-re1000', url: 'http://localhost:5173/tests/renders/lid_driven_cavity.html?re=1000' },
    { name: 'LDC Stability Audit (Re=5000)', id: 'ldc-re5000', url: 'http://localhost:5173/tests/renders/lid_driven_cavity.html?re=5000' },
    { name: 'GpuDispatcher Alignment Proof', id: 'dispatcher-alignment', url: 'http://localhost:5173/tests/renders/dispatcher_alignment_verify.html' },
    { name: 'Beacon Data Injection Audit', id: 'beacon-alignment', url: 'http://localhost:5173/tests/renders/beacon_alignment_verify.html' }
];

// CLI Filtering
const filterArg = process.argv.find(arg => arg.startsWith('--test='));
const targetId = filterArg ? filterArg.split('=')[1] : null;

const tests = targetId ? ALL_TESTS.filter(t => t.id === targetId) : ALL_TESTS;

if (tests.length === 0) {
    console.error(`❌ No test found matching id: ${targetId}`);
    process.exit(1);
}

let currentTestIndex = 0;

function runNextTest() {
    if (currentTestIndex >= tests.length) {
        console.log("\n=======================================");
        console.log("✅ ALL REQUESTED TESTS COMPLETED SUCCESSFULLY");
        console.log("=======================================\n");
        server.close();
        process.exit(0);
        return;
    }

    const test = tests[currentTestIndex];
    console.log(`\n▶️ [${currentTestIndex + 1}/${tests.length}] Executing ${test.name} -> ${test.url}`);
    
    // Launch browser
    const startCmd = process.platform === 'win32' ? 'start' : (process.platform === 'darwin' ? 'open' : 'xdg-open');
    exec(`${startCmd} ${test.url}`);
}

const server = http.createServer((req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'OPTIONS, POST');
    res.setHeader('Access-Control-Allow-Headers', '*');

    if (req.method === 'OPTIONS') {
        res.writeHead(200);
        res.end();
        return;
    }

    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
        const testId = req.headers['x-test-name'] || `unknown`;
        const expectedTest = tests[currentTestIndex];

        if (!expectedTest || testId !== expectedTest.id) {
            console.log(`\x1b[33m⚠️ Ignoring background noise payload from: ${testId}\x1b[0m`);
            res.writeHead(200);
            res.end('OK');
            return;
        }
        
        console.log(`\n⬇️  [RESULT RECEIVED] Mapped Payload: ${testId}\n`);
        console.log(body);
        
        // Write to reports directory
        const timestamp = new Date().toISOString().replace(/:/g, '-').replace(/\..+/, '');
        const filename = path.join(REPORTS_DIR, `${testId}_${timestamp}.log`);
        fs.writeFileSync(filename, body, 'utf8');
        
        console.log(`📝 Wrote official run report to: ${filename}`);

        res.writeHead(200);
        res.end('OK');
        
        currentTestIndex++;
        
        // Wait a few seconds for resources to release before next test
        console.log("⏳ Waiting 2s for resource cooling...");
        setTimeout(() => runNextTest(), 2000);
    });
});

console.log("🚀 Starting Hypercube Autonomous WebGPU Test Suite Runner...");
server.listen(3000, () => {
    runNextTest();
});
