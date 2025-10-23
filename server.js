const express = require('express');
const path = require('path');
const os = require('os');
const fs = require('fs');
const { spawn } = require('child_process');
const { randomUUID } = require('crypto');

const app = express();
app.use(express.json({ limit: '25mb' }));

// Static assets (serves index.html, viewer.js, styles, etc.)
const ROOT_DIR = __dirname;
app.use(express.static(ROOT_DIR));

// Resolve local resources (mirrors Electron paths)
const PYSCF_PATH = path.join(ROOT_DIR, 'pyscf');
const SKALA_SRC_PATH = path.join(ROOT_DIR, 'skala', 'src');
const DFT_SCRIPT_PATH = path.join(ROOT_DIR, 'scripts', 'run_dft.py');

function hasLocalPyscfLibraries() {
  try {
    const libDir = path.join(PYSCF_PATH, 'pyscf', 'lib');
    const entries = fs.readdirSync(libDir);
    return entries.some((name) => name.startsWith('libxc'));
  } catch (_e) {
    return false;
  }
}

// Simple in-memory job registry for streaming logs and final result
const jobs = new Map();

app.post('/api/run-dft', async (req, res) => {
  try {
    const { format = 'xyz', content, label, task, includeForces, optimize } = req.body || {};
    if (!content || typeof content !== 'string') {
      return res.status(400).json({ success: false, error: 'No structure content provided.' });
    }

    // Prepare temp structure file
    const requestedFormat = String(format || '').toLowerCase();
    const normalizedFormat = requestedFormat === 'mmcif' ? 'cif' : requestedFormat;
    const supported = new Set(['xyz', 'pdb', 'pyscf', 'sdf', 'sd', 'mol', 'mol2', 'cml', 'cif']);
    const fmt = supported.has(normalizedFormat) ? normalizedFormat : 'xyz';
    const extension = fmt;
    const tmpDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'rebel-dft-'));
    const structurePath = path.join(tmpDir, `structure.${extension}`);
    await fs.promises.writeFile(structurePath, content, 'utf8');

    // Python env
    const pythonExecutable = process.env.PYTHON || 'python3';
    const pythonPathEntries = [SKALA_SRC_PATH];
    const localPyscfAvailable = hasLocalPyscfLibraries();
    if (localPyscfAvailable) {
      pythonPathEntries.unshift(PYSCF_PATH);
    }
    if (process.env.PYTHONPATH) {
      pythonPathEntries.push(process.env.PYTHONPATH);
    }
    const env = {
      ...process.env,
      PYTHONPATH: pythonPathEntries.filter(Boolean).join(path.delimiter),
      RF_USE_LOCAL_PYSCF: localPyscfAvailable ? '1' : '0',
      RF_FAST_MODE: process.env.RF_FAST_MODE || '1',
      RF_DEFAULT_BASIS: process.env.RF_DEFAULT_BASIS || 'sto-3g',
      RF_GRID_LEVEL: process.env.RF_GRID_LEVEL || '1',
      RF_ATOM_GRID: process.env.RF_ATOM_GRID || '30x110',
      RF_MAX_CYCLE: process.env.RF_MAX_CYCLE || '50',
      RF_CONV_TOL: process.env.RF_CONV_TOL || '1e-6',
      RF_VERBOSE: process.env.RF_VERBOSE || '4',
    };

    // Prepare args
    const args = [DFT_SCRIPT_PATH, '--structure', structurePath, '--format', extension];
    const requestedTask = typeof task === 'string' ? task.toLowerCase() : 'dft';
    const knownTasks = new Set(['dft', 'geomopt', 'optimize']);
    const normalizedTask = knownTasks.has(requestedTask)
      ? (requestedTask === 'optimize' ? 'geomopt' : requestedTask)
      : 'dft';
    const shouldOptimize = Boolean(optimize) || normalizedTask === 'geomopt';
    const shouldIncludeForces = Boolean(includeForces) || shouldOptimize;
    args.push('--task', normalizedTask);
    if (shouldIncludeForces) args.push('--include-forces');
    if (shouldOptimize) args.push('--optimize');
    if (label) args.push('--label', label);

    // Create job and spawn
    const jobId = randomUUID();
    const job = {
      id: jobId,
      emitter: new (require('events').EventEmitter)(),
      done: false,
      result: null,
    };
    jobs.set(jobId, job);

    const child = spawn(pythonExecutable, args, { env });
    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data) => {
      const text = data.toString();
      stdout += text;
      try { job.emitter.emit('log', { stream: 'stdout', text }); } catch (_) {}
    });
    child.stderr.on('data', (data) => {
      const text = data.toString();
      stderr += text;
      try { job.emitter.emit('log', { stream: 'stderr', text }); } catch (_) {}
    });

    const finalize = (exitCode) => {
      // Parse potentially loose JSON from stdout (mirrors Electron code)
      const parseLooseJSON = (text) => {
        if (!text) return null;
        const trimmed = String(text).trim();
        try { return JSON.parse(trimmed); } catch (_) {}
        const lastBrace = trimmed.lastIndexOf('{');
        if (lastBrace >= 0) {
          const tail = trimmed.slice(lastBrace);
          try { return JSON.parse(tail); } catch (_) {}
        }
        const lines = trimmed.split(/\r?\n/).reverse();
        for (const line of lines) {
          const s = line.trim();
          if (s.startsWith('{') && s.endsWith('}')) {
            try { return JSON.parse(s); } catch (_) {}
          }
        }
        return null;
      };

      let result = parseLooseJSON(stdout);
      if (!result || typeof result !== 'object') {
        result = { raw: (stdout || '').trim() };
      }
      if (Number(exitCode) !== 0) {
        result = { success: false, error: (stderr || stdout || 'DFT process failed.').trim() };
      }

      // Attach downloadable URLs for result artifacts via server endpoint
      const attachUrl = (field) => {
        if (!result || typeof result[field] !== 'string' || !result[field]) return;
        const p = result[field];
        result[`${field}_url`] = `/api/file?path=${encodeURIComponent(p)}`;
      };
      attachUrl('result_file');
      attachUrl('forces_json');
      attachUrl('optimized_xyz');
      if (stderr.trim()) {
        result.stderr = stderr.trim();
      }

      job.done = true;
      job.result = { success: result.success !== false, ...result };
      try {
        job.emitter.emit('result', job.result);
        job.emitter.emit('end');
      } catch (_) {}
    };

    child.on('error', () => finalize(1));
    child.on('close', (code) => finalize(code));

    // Respond to client with the created job id
    res.json({ success: true, jobId });
  } catch (err) {
    res.status(500).json({ success: false, error: err?.message || String(err) });
  }
});

// Server-Sent Events for live logs and final result
app.get('/api/run-dft/events/:id', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) {
    res.status(404).end();
    return;
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();

  const writeEvent = (event, data) => {
    res.write(`event: ${event}\n`);
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  const onLog = (msg) => writeEvent('log', msg);
  const onResult = (payload) => writeEvent('result', payload);
  const onEnd = () => {
    writeEvent('end', {});
    res.end();
  };

  job.emitter.on('log', onLog);
  job.emitter.once('result', onResult);
  job.emitter.once('end', onEnd);

  req.on('close', () => {
    job.emitter.removeListener('log', onLog);
  });
});

// Download/serve a result file by absolute path
app.get('/api/file', async (req, res) => {
  const filePath = req.query.path;
  if (!filePath || typeof filePath !== 'string') {
    return res.status(400).send('Missing path');
  }
  try {
    await fs.promises.access(filePath, fs.constants.R_OK);
  } catch (_) {
    return res.status(404).send('Not found');
  }
  res.sendFile(path.resolve(filePath));
});

// Default route -> index
app.get('*', (_req, res) => {
  res.sendFile(path.join(ROOT_DIR, 'index.html'));
});

const HOST = process.env.HOST || '127.0.0.1';
const START_PORT = parseInt(process.env.PORT || '3000', 10) || 3000;

function startServer(port, attemptsLeft = 20) {
  const server = app
    .listen(port, HOST, () => {
      const actual = server.address();
      const usedPort = typeof actual === 'object' && actual ? actual.port : port;
      console.log(`Rebel Forge server listening on http://${HOST}:${usedPort}`);
    })
    .on('error', (err) => {
      if (err && err.code === 'EADDRINUSE' && attemptsLeft > 0) {
        const nextPort = port + 1;
        console.warn(`Port ${port} in use, trying ${nextPort}…`);
        setTimeout(() => startServer(nextPort, attemptsLeft - 1), 100);
      } else {
        console.error('Failed to start server:', err);
        process.exit(1);
      }
    });
}

startServer(START_PORT);
