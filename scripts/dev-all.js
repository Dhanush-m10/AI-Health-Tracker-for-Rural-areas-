import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import net from 'node:net';
import path from 'node:path';

const isWin = process.platform === 'win32';
const npmCmd = isWin ? 'npm.cmd' : 'npm';

const projectRoot = process.cwd();
const pythonFromEnv = process.env.PYTHON_EXECUTABLE;
const pythonCandidates = [
  pythonFromEnv,
  path.join(projectRoot, '.venv', 'Scripts', 'python.exe'),
  path.join(projectRoot, '.venv', 'bin', 'python'),
  'python',
].filter(Boolean);

const pythonCmd = pythonCandidates.find((candidate) => {
  if (!candidate) return false;
  if (candidate === 'python') return true;
  return existsSync(candidate);
});

if (!pythonCmd) {
  console.error('Could not locate a Python executable. Set PYTHON_EXECUTABLE or create .venv.');
  process.exit(1);
}

const children = [];
let isShuttingDown = false;

function shutdown(exitCode = 0) {
  if (isShuttingDown) return;
  isShuttingDown = true;

  for (const child of children) {
    if (!child.killed) {
      child.kill('SIGTERM');
    }
  }

  setTimeout(() => {
    for (const child of children) {
      if (!child.killed) {
        child.kill('SIGKILL');
      }
    }
    process.exit(exitCode);
  }, 1500);
}

function startProcess(commandLine, name, envOverrides = {}) {
  const child = spawn(commandLine, {
    stdio: 'inherit',
    cwd: projectRoot,
    shell: true,
    env: {
      ...process.env,
      ...envOverrides,
    },
  });

  child.on('exit', (code) => {
    if (!isShuttingDown && code && code !== 0) {
      console.error(`${name} exited with code ${code}`);
      shutdown(code);
    }
  });

  child.on('error', (err) => {
    console.error(`${name} failed to start:`, err.message);
    shutdown(1);
  });

  children.push(child);
  return child;
}

function checkPortAvailable(port, host = '127.0.0.1') {
  return new Promise((resolve) => {
    const tester = net
      .createServer()
      .once('error', () => resolve(false))
      .once('listening', () => {
        tester.close(() => resolve(true));
      })
      .listen(port, host);
  });
}

async function findFreePort(startPort, endPort = startPort + 20) {
  for (let port = startPort; port <= endPort; port += 1) {
    const free = await checkPortAvailable(port);
    if (free) {
      return port;
    }
  }
  throw new Error(`No free port found in range ${startPort}-${endPort}.`);
}

process.on('SIGINT', () => shutdown(0));
process.on('SIGTERM', () => shutdown(0));

async function main() {
  const apiPort = await findFreePort(8000);
  const backendTarget = `http://localhost:${apiPort}`;

  console.log(`Using Python executable: ${pythonCmd}`);
  console.log(`Starting backend API on ${backendTarget} ...`);
  startProcess(`"${pythonCmd}" main.py api --host 0.0.0.0 --port ${apiPort}`, 'backend');

  console.log(`Starting frontend with proxy target ${backendTarget} ...`);
  startProcess(`${npmCmd} run dev:web`, 'frontend', {
    VITE_BACKEND_TARGET: backendTarget,
  });
}

main().catch((err) => {
  console.error(`Failed to start dev services: ${err.message}`);
  shutdown(1);
});