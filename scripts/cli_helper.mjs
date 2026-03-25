import http from 'http';
import { exec } from 'child_process';

const server = http.createServer((req, res) => {
    // Handle CORS preflight
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
        console.log('\n=======================================');
        console.log('       NATIVE WEBGPU LOG OUTPUT        ');
        console.log('=======================================\n');
        console.log(body);
        console.log('\n=======================================\n');
        
        res.writeHead(200);
        res.end('OK');
        
        // Shut down silently after receiving the payload
        setTimeout(() => {
            server.close();
            process.exit(0);
        }, 500);
    });
});

server.listen(3000, () => {
    const startCmd = process.platform === 'win32' ? 'start' : (process.platform === 'darwin' ? 'open' : 'xdg-open');
    const targetUrl = process.argv[2] || 'http://localhost:5173/conservation.html';
    
    console.log(`Listening on port 3000... Launching browser to run native physics at ${targetUrl}`);
    exec(`${startCmd} ${targetUrl}`);
});
