import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
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

function startProcess(commandLine, name) {
  const child = spawn(commandLine, {
    stdio: 'inherit',
    cwd: projectRoot,
    shell: true,
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

process.on('SIGINT', () => shutdown(0));
process.on('SIGTERM', () => shutdown(0));

console.log(`Using Python executable: ${pythonCmd}`);
console.log('Starting backend API on http://localhost:8000 ...');
startProcess(`"${pythonCmd}" ml/main.py api --host 0.0.0.0 --port 8000`, 'backend');

console.log('Starting frontend on http://localhost:3000 ...');
startProcess(`${npmCmd} run dev:web`, 'frontend');