const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const os = require('os');
const fs = require('fs');
const { spawn } = require('child_process');
const { pathToFileURL } = require('url');

const fsPromises = fs.promises;
// When packaged, resources are copied to process.resourcesPath via electron-builder extraResources
const ROOT_DIR = app.isPackaged ? process.resourcesPath : __dirname;
const PYSCF_PATH = path.join(ROOT_DIR, 'pyscf');
const SKALA_SRC_PATH = path.join(ROOT_DIR, 'skala', 'src');
const DFT_SCRIPT_PATH = path.join(ROOT_DIR, 'scripts', 'run_dft.py');

function hasLocalPyscfLibraries() {
  try {
    const libDir = path.join(PYSCF_PATH, 'pyscf', 'lib');
    const entries = fs.readdirSync(libDir);
    return entries.some((name) => name.startsWith('libxc'));
  } catch (error) {
    return false;
  }
}

function createWindow() {
  const mainWindow = new BrowserWindow({
    width: 1024,
    height: 768,
    webPreferences: {
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'index.html'));

  if (!app.isPackaged) {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

ipcMain.handle('run-dft', async (_event, payload) => {
  try {
    const {
      format = 'xyz',
      content,
      label,
      task,
      includeForces,
      optimize,
    } = payload || {};
    if (!content || typeof content !== 'string') {
      throw new Error('No structure content provided.');
    }

    const requestedFormat = (format || '').toLowerCase();
    const normalizedFormat = requestedFormat === 'mmcif' ? 'cif' : requestedFormat;
    const supported = new Set(['xyz', 'pdb', 'pyscf', 'sdf', 'sd', 'mol', 'mol2', 'cml', 'cif']);
    const fmt = supported.has(normalizedFormat) ? normalizedFormat : 'xyz';
    const extension = fmt;
    const tmpDir = await fsPromises.mkdtemp(path.join(os.tmpdir(), 'rebel-dft-'));
    const structurePath = path.join(tmpDir, `structure.${extension}`);
    await fsPromises.writeFile(structurePath, content, 'utf8');

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
      USE_LOCAL_PYSCF: localPyscfAvailable ? '1' : '0',
      // Fast-mode defaults to speed up SCF for interactive runs
      RF_FAST_MODE: process.env.RF_FAST_MODE || '1',
      RF_DEFAULT_BASIS: process.env.RF_DEFAULT_BASIS || 'sto-3g',
      RF_GRID_LEVEL: process.env.RF_GRID_LEVEL || '1',
      RF_ATOM_GRID: process.env.RF_ATOM_GRID || '30x110',
      RF_MAX_CYCLE: process.env.RF_MAX_CYCLE || '50',
      RF_CONV_TOL: process.env.RF_CONV_TOL || '1e-6',
      RF_VERBOSE: process.env.RF_VERBOSE || '4',
    };

    const args = [DFT_SCRIPT_PATH, '--structure', structurePath, '--format', extension];
    const requestedTask = typeof task === 'string' ? task.toLowerCase() : 'dft';
    const knownTasks = new Set(['dft', 'geomopt', 'optimize']);
    const normalizedTask = knownTasks.has(requestedTask)
      ? (requestedTask === 'optimize' ? 'geomopt' : requestedTask)
      : 'dft';
    const shouldOptimize = Boolean(optimize) || normalizedTask === 'geomopt';
    const shouldIncludeForces = Boolean(includeForces) || shouldOptimize;
    args.push('--task', normalizedTask);
    if (shouldIncludeForces) {
      args.push('--include-forces');
    }
    if (shouldOptimize) {
      args.push('--optimize');
    }
    if (label) {
      args.push('--label', label);
    }

    const child = spawn(pythonExecutable, args, { env });

    let stdout = '';
    let stderr = '';
    // Stream logs to renderer as they arrive
    child.stdout.on('data', (data) => {
      const text = data.toString();
      stdout += text;
      try { _event?.sender?.send?.('dft-log', { stream: 'stdout', text }); } catch (_) {}
    });
    child.stderr.on('data', (data) => {
      const text = data.toString();
      stderr += text;
      try { _event?.sender?.send?.('dft-log', { stream: 'stderr', text }); } catch (_) {}
    });

    const exitCode = await new Promise((resolve, reject) => {
      child.on('error', reject);
      child.on('close', resolve);
    });

    if (exitCode !== 0) {
      throw new Error(stderr.trim() || stdout.trim() || 'DFT process failed.');
    }

    const parseLooseJSON = (text) => {
      if (!text) return null;
      const trimmed = String(text).trim();
      try {
        return JSON.parse(trimmed);
      } catch (_) {
        // Try to parse from the last JSON-looking block
        const lastBrace = trimmed.lastIndexOf('{');
        if (lastBrace >= 0) {
          const tail = trimmed.slice(lastBrace);
          try { return JSON.parse(tail); } catch (_) {}
        }
        // Fall back to last line
        const lines = trimmed.split(/\r?\n/).reverse();
        for (const line of lines) {
          const s = line.trim();
          if (s.startsWith('{') && s.endsWith('}')) {
            try { return JSON.parse(s); } catch (_) {}
          }
        }
        return null;
      }
    };

    let result = parseLooseJSON(stdout);
    if (!result || typeof result !== 'object') {
      result = { raw: (stdout || '').trim() };
    }

    const attachFileUrl = async (field) => {
      if (!result || typeof result[field] !== 'string' || !result[field]) {
        return;
      }
      const candidatePath = result[field];
      try {
        await fsPromises.access(candidatePath, fs.constants.F_OK);
        result[`${field}_url`] = pathToFileURL(candidatePath).toString();
      } catch (fileErr) {
        console.warn(`[run-dft] file unavailable for ${field} at ${candidatePath}: ${fileErr.message}`);
      }
    };

    // Common result file and geomopt artifacts
    await attachFileUrl('result_file');
    await attachFileUrl('forces_json');
    await attachFileUrl('optimized_xyz');
    // Additional analysis artifacts removed

    if (stderr.trim()) {
      result.stderr = stderr.trim();
    }

    return { success: true, ...result };
  } catch (error) {
    console.error('[run-dft] error:', error);
    return { success: false, error: error.message };
  }
});
