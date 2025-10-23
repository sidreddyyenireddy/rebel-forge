(function () {
  const SAMPLE_LIBRARY = {
    local1crn: {
      label: '1CRN (Local Benchmark)',
      source: './miew/packages/miew/demo/data/1CRN.pdb',
    },
    remote4hhb: {
      label: 'Hemoglobin 4HHB (RCSB)',
      source: '4HHB',
    },
    remote2hhb: {
      label: 'Hemoglobin 2HHB (RCSB)',
      source: '2HHB',
    },
    remote4xn6: {
      label: '4XN6 (Electron Density)',
      source: 'cif:4XN6',
    },
  };

  const PRESET_MOLECULES = [
    { label: 'H2O', token: 'H2O' },
    { label: 'CO2', token: 'CO2' },
    { label: 'CH4', token: 'CH4' },
    { label: 'NH3', token: 'NH3' },
    { label: 'O2', token: 'O2' },
    { label: 'C6H6', token: 'benzene' },
    { label: 'NaCl', token: 'NaCl' },
  ];

  let currentStructureProvider = null;
  let currentStructureLabel = '';
  let runButton = null;
  let lastRawUpload = null; // { content: string, format: string }
  const DEFAULT_FILE_LABEL = 'No file chosen.';
  let fileNameIndicator = null;
  let statusContainer = null;
  let statusTextNode = null;
  let downloadLink = null;
  let geomButton = null;
  let geomSection = null;
  let geomProgress = null;
  let geomProgressText = null;
  let geomResult = null;
  let geomEnergyNode = null;
  let geomConvergenceNode = null;
  let geomWarning = null;
  let downloadForcesLink = null;
  let downloadOptimizedLink = null;
  let lastDftPayload = null;
  let lastDftResponse = null;

  function updateFileIndicator(text, state = 'idle') {
    if (!fileNameIndicator) {
      return;
    }
    fileNameIndicator.textContent = text;
    fileNameIndicator.setAttribute('data-state', state);
  }

  function resetFileIndicator() {
    updateFileIndicator(DEFAULT_FILE_LABEL, 'empty');
  }

  function readSession() {
    const raw = window.sessionStorage.getItem('rebel-forge-auth');
    if (!raw) {
      return null;
    }
    try {
      return JSON.parse(raw);
    } catch (error) {
      console.error('Invalid session payload, clearing.', error);
      window.sessionStorage.removeItem('rebel-forge-auth');
      return null;
    }
  }

  function resolveFileHref(payload, pathProp, urlProp) {
    if (!payload) {
      return '';
    }
    if (urlProp && typeof payload[urlProp] === 'string') {
      const candidate = payload[urlProp].trim();
      if (candidate) {
        return candidate;
      }
    }
    if (typeof payload[pathProp] === 'string') {
      const rawPath = payload[pathProp].trim();
      if (rawPath) {
        const normalized = rawPath.replace(/\\/g, '/');
        const prefixed = normalized.startsWith('/') ? normalized : `/${normalized}`;
        try {
          return `file://${encodeURI(prefixed)}`;
        } catch (_) {
          return `file://${prefixed}`;
        }
      }
    }
    return '';
  }

  function ensureStatusElements() {
    if (!statusContainer) {
      statusContainer = document.getElementById('load-status');
    }
    if (!statusTextNode) {
      statusTextNode = document.getElementById('load-status-text');
    }
    if (!downloadLink) {
      downloadLink = document.getElementById('download-dft-result');
    }
    if (!geomSection) {
      geomSection = document.getElementById('geom-opt-section');
      geomProgress = document.getElementById('geom-progress');
      geomProgressText = document.getElementById('geom-progress-text');
      geomResult = document.getElementById('geom-result');
      geomEnergyNode = document.getElementById('geom-energy');
      geomConvergenceNode = document.getElementById('geom-convergence');
      geomWarning = document.getElementById('geom-warning');
      downloadForcesLink = document.getElementById('download-forces');
      downloadOptimizedLink = document.getElementById('download-optimized');
    }
  }

  function resetGeomUI() {
    ensureStatusElements();
    if (!geomSection) {
      return;
    }
    geomSection.hidden = true;
    if (geomProgress) {
      geomProgress.hidden = true;
    }
    if (geomProgressText) {
      geomProgressText.textContent = 'Optimizing geometry and computing forces…';
    }
    if (geomResult) {
      geomResult.hidden = true;
    }
    if (geomWarning) {
      geomWarning.hidden = true;
      geomWarning.textContent = '';
    }
    if (geomEnergyNode) {
      geomEnergyNode.textContent = 'Final optimized energy: —';
    }
    if (geomConvergenceNode) {
      geomConvergenceNode.textContent = 'Convergence: —';
      geomConvergenceNode.setAttribute('data-state', 'unknown');
    }
    if (downloadForcesLink) {
      downloadForcesLink.hidden = true;
      downloadForcesLink.removeAttribute('href');
    }
    if (downloadOptimizedLink) {
      downloadOptimizedLink.hidden = true;
      downloadOptimizedLink.removeAttribute('href');
    }
    // Hide live log panel until the next run
    try {
      ensureLogElements();
      if (logContainer) {
        logContainer.hidden = true;
      }
    } catch (_) {}
  }

  function showGeomProgress(message) {
    ensureStatusElements();
    if (!geomSection) {
      return;
    }
    geomSection.hidden = false;
    if (geomWarning) {
      geomWarning.hidden = true;
    }
    if (geomResult) {
      geomResult.hidden = true;
    }
    if (geomProgress) {
      geomProgress.hidden = false;
    }
    if (geomProgressText) {
      geomProgressText.textContent = message || 'Optimizing geometry and computing forces…';
    }
  }

  function showGeomWarning(message) {
    ensureStatusElements();
    if (!geomSection || !geomWarning) {
      return;
    }
    geomSection.hidden = false;
    if (geomProgress) {
      geomProgress.hidden = true;
    }
    geomWarning.textContent = message;
    geomWarning.hidden = false;
  }

  function setDownloadLink(anchor, href) {
    if (!anchor) {
      return;
    }
    if (href) {
      anchor.href = href;
      anchor.hidden = false;
    } else {
      anchor.hidden = true;
      anchor.removeAttribute('href');
    }
  }

  function showGeomResult(response) {
    ensureStatusElements();
    if (!geomSection) {
      return;
    }
    const warningMessage =
      (response && typeof response.optimization_unavailable === 'string' && response.optimization_unavailable) ||
      (response && typeof response.optimization_error === 'string' && response.optimization_error) ||
      '';

    geomSection.hidden = false;
    if (geomProgress) {
      geomProgress.hidden = true;
    }
    if (geomResult) {
      geomResult.hidden = false;
    }
    if (geomWarning) {
      if (warningMessage) {
        geomWarning.textContent = warningMessage;
        geomWarning.hidden = false;
      } else {
        geomWarning.textContent = '';
        geomWarning.hidden = true;
      }
    }

    if (geomEnergyNode) {
      const candidates = [];
      if (response && response.energy_final !== undefined && response.energy_final !== null) {
        candidates.push(Number(response.energy_final));
      }
      if (response && response.energy_initial !== undefined && response.energy_initial !== null) {
        candidates.push(Number(response.energy_initial));
      }
      if (response && response.energy !== undefined && response.energy !== null) {
        candidates.push(Number(response.energy));
      }
      const energyValue = candidates.find((value) => Number.isFinite(value));
      if (Number.isFinite(energyValue)) {
        geomEnergyNode.textContent = `Final optimized energy: ${energyValue.toFixed(6)} Ha`;
      } else {
        geomEnergyNode.textContent = 'Final optimized energy: unavailable';
      }
    }

    if (geomConvergenceNode) {
      if (response && typeof response.converged === 'boolean') {
        const converged = Boolean(response.converged);
        geomConvergenceNode.textContent = converged ? 'Convergence: ✓' : 'Convergence: ✗';
        geomConvergenceNode.setAttribute('data-state', converged ? 'success' : 'failure');
      } else if (warningMessage) {
        geomConvergenceNode.textContent = 'Convergence: unavailable';
        geomConvergenceNode.setAttribute('data-state', 'unknown');
      } else {
        geomConvergenceNode.textContent = 'Convergence: —';
        geomConvergenceNode.setAttribute('data-state', 'unknown');
      }
    }

    setDownloadLink(downloadForcesLink, resolveFileHref(response, 'forces_json', 'forces_json_url'));
    setDownloadLink(downloadOptimizedLink, resolveFileHref(response, 'optimized_xyz', 'optimized_xyz_url'));
  }

  function setStatus(message, intent = 'idle') {
    ensureStatusElements();
    if (!statusContainer) {
      return;
    }
    if (statusTextNode) {
      statusTextNode.textContent = message;
    } else {
      statusContainer.textContent = message;
    }
    statusContainer.setAttribute('data-intent', intent);
    if (downloadLink) {
      downloadLink.hidden = true;
      downloadLink.removeAttribute('href');
    }
  }

  // --- PySCF live log panel ---
  let logContainer = null;
  let logNode = null;
  function ensureLogElements() {
    if (!logContainer) {
      logContainer = document.getElementById('pyscf-log-container');
    }
    if (!logNode) {
      logNode = document.getElementById('pyscf-log');
    }
  }
  function clearDFTLog() {
    ensureLogElements();
    if (logNode) {
      logNode.textContent = '';
    }
    if (logContainer) {
      logContainer.hidden = false;
    }
  }
  function appendDFTLog(text) {
    ensureLogElements();
    if (!logNode) return;
    if (logContainer) logContainer.hidden = false;
    logNode.textContent += String(text || '');
    try { logNode.scrollTop = logNode.scrollHeight; } catch (_) {}
  }

  function updateDownloadLink(payload) {
    ensureStatusElements();
    if (!downloadLink) {
      return;
    }
    const href = resolveFileHref(payload, 'result_file', 'result_file_url');
    if (!href) {
      downloadLink.hidden = true;
      downloadLink.removeAttribute('href');
      return;
    }
    downloadLink.href = href;
    downloadLink.hidden = false;
    if (!downloadLink.hasAttribute('download')) {
      downloadLink.setAttribute('download', 'dft_result.json');
    }
  }

  function updateRunButtonState() {
    if (runButton) {
      runButton.disabled = !currentStructureProvider;
    }
  }

  function extractXYZ(viewer) {
    const visual = viewer && typeof viewer._getComplexVisual === 'function' ? viewer._getComplexVisual() : null;
    const complex = visual ? visual.getComplex() : null;
    if (!complex) {
      return null;
    }
    const atoms = [];
    complex.forEachAtom((atom) => {
      if (!atom || !atom.element || !atom.position) {
        return;
      }
      const symbol = (atom.element.name || 'X').trim() || 'X';
      const { x, y, z } = atom.position;
      atoms.push(`${symbol} ${x.toFixed(6)} ${y.toFixed(6)} ${z.toFixed(6)}`);
    });
    if (!atoms.length) {
      return null;
    }
    return `${atoms.length}\nGenerated by Rebel Forge\n${atoms.join('\n')}`;
  }

  function buildStructureProvider(viewer, label) {
    return async () => {
      const xyz = extractXYZ(viewer);
      if (xyz) {
        return { format: 'xyz', content: xyz, label };
      }
      if (viewer && typeof viewer._export === 'function') {
        try {
          const pdb = await viewer._export('pdb');
          if (pdb) {
            return { format: 'pdb', content: pdb, label };
          }
        } catch (exportError) {
          console.warn('[Viewer] export fallback unavailable:', exportError);
        }
      }
      if (lastRawUpload && lastRawUpload.content && lastRawUpload.format) {
        return { format: lastRawUpload.format, content: lastRawUpload.content, label };
      }
      throw new Error('Unable to export structure from viewer.');
    };
  }

  function registerStructure(viewer, label) {
    currentStructureLabel = label;
    currentStructureProvider = buildStructureProvider(viewer, label);
    lastDftPayload = null;
    lastDftResponse = null;
    if (geomButton) {
      geomButton.hidden = true;
      geomButton.disabled = true;
    }
    resetGeomUI();
    updateRunButtonState();
  }

  function registerRawUploadFallback(label) {
    if (!lastRawUpload || !lastRawUpload.content || !lastRawUpload.format) {
      return null;
    }
    const { content, format } = lastRawUpload;
    const payload = { content, format, label };
    currentStructureLabel = label;
    currentStructureProvider = async () => ({ ...payload });
    lastDftPayload = null;
    lastDftResponse = null;
    if (geomButton) {
      geomButton.hidden = true;
      geomButton.disabled = true;
    }
    resetGeomUI();
    updateRunButtonState();
    const canvas = document.getElementById('miew-container');
    if (canvas) {
      canvas.classList.add('rf-hidden');
    }
    console.info('[Viewer] Using raw upload fallback for', label, 'as', format);
    return format;
  }

  function populateSamples() {
    const select = document.getElementById('sample-select');
    if (!select) {
      return;
    }
    while (select.options.length > 1) {
      select.remove(1);
    }
    Object.entries(SAMPLE_LIBRARY).forEach(([key, entry]) => {
      const option = document.createElement('option');
      option.value = key;
      option.textContent = entry.label;
      select.appendChild(option);
    });
  }

  function bindSessionMeta(session) {
    const emailEl = document.getElementById('operator-email');
    if (emailEl && session && session.email) {
      emailEl.textContent = session.email;
    }
    const signOutBtn = document.getElementById('signout-btn');
    if (signOutBtn) {
      signOutBtn.addEventListener('click', () => {
        window.sessionStorage.removeItem('rebel-forge-auth');
        // Return to the welcome screen instead of an auth page
        window.location.replace('index.html');
      });
    }
  }

  // Read any existing session, but do not require authentication.
  const session = readSession();

  document.addEventListener('DOMContentLoaded', () => {
    bindSessionMeta(session);
    populateSamples();

    // Additional Analysis entry removed

    // Stream PySCF logs from the backend into the UI
    try {
      if (window.desktopAPI && typeof window.desktopAPI.onDFTLog === 'function') {
        window.desktopAPI.onDFTLog((msg) => {
          if (msg && typeof msg.text === 'string') {
            appendDFTLog(msg.text);
          }
        });
      }
    } catch (_) {}

    runButton = document.getElementById('run-dft-btn');
    geomButton = document.getElementById('run-geom-btn');
    let runButtonLabel = 'Run PySCF with Skala';
    const canRunGeom =
      !!(window.desktopAPI && typeof window.desktopAPI.runGeometryOptimization === 'function');

    resetGeomUI();

    if (geomButton) {
      geomButton.hidden = true;
      geomButton.disabled = true;
      if (!canRunGeom) {
        geomButton.title = 'Geometry optimisation requires an updated desktop backend.';
      }
    }

    if (runButton) {
      runButtonLabel = runButton.textContent.trim() || runButtonLabel;
      runButton.disabled = true;
      runButton.addEventListener('click', async () => {
        if (!currentStructureProvider) {
          setStatus('Load a structure before running PySCF.', 'error');
          return;
        }
        if (!window.desktopAPI || typeof window.desktopAPI.runDFT !== 'function') {
          setStatus('Desktop API unavailable. Cannot run PySCF.', 'error');
          return;
        }
        try {
          lastDftPayload = null;
          lastDftResponse = null;
          resetGeomUI();
          if (geomButton) {
            geomButton.hidden = true;
            geomButton.disabled = true;
          }
          runButton.disabled = true;
          runButton.textContent = 'Running…';
          setStatus('Running PySCF with Skala…');
          clearDFTLog();
          const payload = await currentStructureProvider();
          if (!payload || !payload.content) {
            throw new Error('No structure data available for DFT.');
          }
          const response = await window.desktopAPI.runDFT(payload);
          if (response && response.success) {
            lastDftPayload = { ...payload };
            lastDftResponse = { ...response };
            if (response.pyscf_path) {
              console.info('[DFT] PySCF atom file:', response.pyscf_path);
            }
            if (typeof response.result_file === 'string') {
              console.info('[DFT] Result JSON:', response.result_file);
            }
            if (typeof response.sequence === 'string' && response.sequence.trim()) {
              try {
                const ta = document.getElementById('text-structure');
                const tf = document.getElementById('text-format');
                if (ta) ta.value = response.sequence.trim();
                if (tf) tf.value = 'pyscf';
                console.info('[DFT] Parsed sequence (PySCF atom):\n' + response.sequence.trim());
              } catch (_) {}
            }
            const e = Number(response.energy);
            if (Number.isFinite(e)) {
              const extras = [];
              if (typeof response.basis === 'string' && response.basis) extras.push(response.basis);
              if (typeof response.charge === 'number') extras.push(`q=${response.charge}`);
              if (typeof response.spin === 'number') extras.push(`spin=${response.spin}`);
              const extraText = extras.length ? ` (${extras.join(', ')})` : '';
              setStatus(
                `DFT complete. Energy: ${e.toFixed(6)} Ha${extraText}.`,
                'success',
              );
            } else {
              setStatus('DFT complete.', 'success');
            }
            updateDownloadLink(lastDftResponse);
            if (geomButton && canRunGeom) {
              geomButton.hidden = false;
              geomButton.disabled = false;
            }
          } else {
            const message = (response && response.error) || 'DFT run failed.';
            throw new Error(message);
          }
        } catch (error) {
          console.error(error);
          setStatus(error.message || 'DFT run failed. See console for details.', 'error');
          lastDftPayload = null;
          lastDftResponse = null;
          resetGeomUI();
          if (geomButton) {
            geomButton.hidden = true;
            geomButton.disabled = true;
          }
        } finally {
          runButton.textContent = runButtonLabel;
          updateRunButtonState();
        }
      });
    }

    if (geomButton && canRunGeom) {
      const geomButtonLabel = geomButton.textContent.trim() || 'Run Geometry Optimization + Forces';
      geomButton.addEventListener('click', async () => {
        if (!currentStructureProvider) {
          showGeomWarning('Load a structure and run DFT before optimizing geometry.');
          return;
        }
        if (!lastDftPayload) {
          showGeomWarning('Run the initial DFT calculation before starting optimization.');
          return;
        }
        if (!window.desktopAPI || typeof window.desktopAPI.runGeometryOptimization !== 'function') {
          showGeomWarning('Desktop API unavailable. Cannot perform geometry optimisation.');
          return;
        }
        try {
          geomButton.disabled = true;
          geomButton.textContent = 'Optimizing…';
          showGeomProgress('Optimizing geometry and computing forces…');
          clearDFTLog();
          const payload = { ...lastDftPayload };
          const response = await window.desktopAPI.runGeometryOptimization(payload);
          if (response && response.success) {
            lastDftResponse = { ...(lastDftResponse || {}), ...response };
            setStatus('Geometry optimization and forces complete.', 'success');
            updateDownloadLink(lastDftResponse);
            showGeomResult(lastDftResponse);
          } else {
            const message = (response && response.error) || 'Geometry optimization failed.';
            throw new Error(message);
          }
        } catch (error) {
          console.error(error);
          const message =
            (error && error.message) ||
            'Geometry optimization failed. See console for details.';
          showGeomWarning(message);
        } finally {
          geomButton.textContent = geomButtonLabel;
          geomButton.disabled = false;
        }
      });
    }

    if (typeof window.Miew === 'undefined') {
      setStatus('Miew assets failed to load. Verify the local bundle.', 'error');
      return;
    }

    initializeViewer();
  });

  function initializeViewer() {
    const container = document.getElementById('miew-container');
    if (!container) {
      setStatus('Viewer container missing from document.', 'error');
      return;
    }

    const viewer = new window.Miew({
      container,
      load: null,
      reps: [
        { mode: 'CA', colorer: 'SQ', material: 'DF' },
        { mode: 'BS', colorer: 'EL', material: 'DF' },
      ],
      settings: {
        axes: true,
        fps: false,
        bg: { color: 0x0f172a, transparent: false },
        resolution: 'high',
      },
    });

    if (!viewer.init()) {
      setStatus('Unable to initialise Miew. Check console for details.', 'error');
      return;
    }

    viewer.run();
    window.miewViewer = viewer;
    viewer.logger.addEventListener('message', (event) => {
      const prefix = '[Miew]';
      if (event.level === 'error') {
        console.error(prefix, event.message);
      } else if (event.level === 'warn') {
        console.warn(prefix, event.message);
      } else {
        console.info(prefix, event.message);
      }
    });

    wireControls(viewer);
    setStatus('Viewer ready. Load or paste a structure to begin.');
  }

  function loadSource(viewer, source, label, options) {
    setStatus(`Loading ${label}…`);
    const previousProvider = currentStructureProvider;
    const previousLabel = currentStructureLabel;
    currentStructureProvider = null;
    currentStructureLabel = '';
    updateRunButtonState();
    const canvas = document.getElementById('miew-container');
    if (canvas) {
      canvas.classList.remove('rf-hidden');
    }
    return viewer
      .load(source, options)
      .then(() => {
        if (typeof viewer.resetView === 'function') {
          viewer.resetView();
        }
        setStatus(`${label} ready.`, 'success');
        registerStructure(viewer, label);
      })
      .catch((error) => {
        console.error(error);
        let fallbackFormat = null;
        if (typeof File !== 'undefined' && source instanceof File) {
          fallbackFormat = registerRawUploadFallback(label);
        }
        if (fallbackFormat) {
          const fmtText = fallbackFormat.toUpperCase();
          setStatus(`Viewer cannot render ${label}. Using ${fmtText} data for PySCF runs.`, 'idle');
          return;
        }
        setStatus(`Unable to load ${label}.`, 'error');
        currentStructureProvider = previousProvider;
        currentStructureLabel = previousLabel;
        updateRunButtonState();
        throw error;
      });
  }

  function loadSample(viewer, key) {
    const sample = SAMPLE_LIBRARY[key];
    const select = document.getElementById('sample-select');
    if (!sample) {
      return;
    }
    resetFileIndicator();
    if (select) {
      select.value = key;
    }
    loadSource(viewer, sample.source, sample.label);
  }

  async function handleFile(viewer, file) {
    const sampleSelect = document.getElementById('sample-select');
    if (sampleSelect) {
      sampleSelect.selectedIndex = 0;
    }
    const fileLabel = file && file.name ? file.name : 'uploaded-structure';
    updateFileIndicator(fileLabel, 'selected');
    // Stash raw content for formats we can parse in Python, used as a final fallback.
    const rawExt = (file.name.split('.').pop() || '').toLowerCase();
    const normalizedExt = rawExt === 'mmcif' ? 'cif' : rawExt;
    const supported = new Set(['xyz', 'pdb', 'pyscf', 'sdf', 'sd', 'mol', 'mol2', 'cml', 'cif']);
    lastRawUpload = null;
    if (supported.has(normalizedExt) && typeof file.text === 'function') {
      try {
        const text = await file.text();
        lastRawUpload = { content: text, format: normalizedExt };
      } catch (_) {
        lastRawUpload = null;
      }
    }
    return loadSource(viewer, file, fileLabel)
      .then((result) => {
        updateFileIndicator(fileLabel, 'ready');
        return result;
      })
      .catch((error) => {
        updateFileIndicator(`${fileLabel} (load failed)`, 'error');
        throw error;
      });
  }

  function detectTextExtension(text) {
    const head = text.slice(0, 2000).toUpperCase();
    const lines = text.split(/\r?\n/);
    if (lines.length > 2) {
      if (/^\s*\d+\s*$/.test(lines[0]) && lines[1]) {
        return 'xyz';
      }
      if (lines.some((line) => line.includes('V2000') || line.includes('V3000'))) {
        return 'mol';
      }
    }
    if (head.includes('_ATOM_SITE') || head.includes('_CELL_LENGTH')) {
      return 'cif';
    }
    if (head.includes('HETATM') || head.includes('ATOM')) {
      return 'pdb';
    }
    return 'pdb';
  }

  function handleText(viewer, text) {
    const trimmed = text.trim();
    if (!trimmed) {
      return;
    }
    resetFileIndicator();
    const sampleSelect = document.getElementById('sample-select');
    if (sampleSelect) {
      sampleSelect.selectedIndex = 0;
    }

    const simpleToken = /^[A-Za-z0-9]+$/;
    if (trimmed.length <= 25 && simpleToken.test(trimmed)) {
      const token = trimmed.toLowerCase();
      return loadSource(viewer, `pc:${token}`, trimmed.toUpperCase());
    }

    const ext = detectTextExtension(trimmed);
    const file = new File([trimmed], `dropped.${ext}`, { type: 'text/plain' });
    return loadSource(viewer, file, `dropped.${ext}`);
  }

  function setupDragAndDrop(viewer) {
    const dropTarget = document.getElementById('miew-container');
    const viewerCard = document.querySelector('.rf-viewer-card');
    if (!dropTarget || !viewerCard) {
      return;
    }

    let dragDepth = 0;
    const highlight = () => viewerCard.classList.add('rf-drop-active');
    const unhighlight = () => viewerCard.classList.remove('rf-drop-active');

    const prevent = (event) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = 'copy';
    };

    dropTarget.addEventListener('dragenter', (event) => {
      prevent(event);
      if (dragDepth === 0) {
        highlight();
      }
      dragDepth += 1;
    });

    dropTarget.addEventListener('dragover', prevent);

    dropTarget.addEventListener('dragleave', (event) => {
      prevent(event);
      dragDepth = Math.max(dragDepth - 1, 0);
      if (dragDepth === 0) {
        unhighlight();
      }
    });

    dropTarget.addEventListener('drop', (event) => {
      event.preventDefault();
      dragDepth = 0;
      unhighlight();

      const dt = event.dataTransfer;
      if (!dt) {
        return;
      }

      if (dt.files && dt.files.length) {
        const promise = handleFile(viewer, dt.files[0]);
        if (promise && typeof promise.catch === 'function') {
          promise.catch(() => {});
        }
        return;
      }

      const directText = dt.getData('text/plain');
      if (directText) {
        const promise = handleText(viewer, directText);
        if (promise && typeof promise.catch === 'function') {
          promise.catch(() => {});
        }
        return;
      }

      if (dt.items) {
        const stringItem = Array.from(dt.items).find((item) => item.kind === 'string');
        if (stringItem) {
          stringItem.getAsString((value) => {
            if (value) {
              const promise = handleText(viewer, value);
              if (promise && typeof promise.catch === 'function') {
                promise.catch(() => {});
              }
            }
          });
        }
      }
    });

    document.addEventListener('dragover', (event) => {
      event.preventDefault();
    });
    document.addEventListener('drop', (event) => {
      if (!dropTarget.contains(event.target)) {
        event.preventDefault();
      }
    });
  }

  function wireControls(viewer) {
    const sampleSelect = document.getElementById('sample-select');
    if (sampleSelect) {
      sampleSelect.addEventListener('change', () => {
        if (sampleSelect.value) {
          loadSample(viewer, sampleSelect.value);
        }
      });
    }
    fileNameIndicator = document.getElementById('structure-file-name');
    resetFileIndicator();

    // Text-based intake for PySCF/XYZ/PDB
    const textArea = document.getElementById('text-structure');
    const textFormat = document.getElementById('text-format');
    const useTextBtn = document.getElementById('use-text-btn');
    if (useTextBtn) {
      useTextBtn.addEventListener('click', () => {
        const content = (textArea && typeof textArea.value === 'string') ? textArea.value.trim() : '';
        if (!content) {
          setStatus('Paste a structure first (PySCF/XYZ/PDB).', 'error');
          return;
        }
        resetFileIndicator();
        const fmt = (textFormat && textFormat.value) ? String(textFormat.value).toLowerCase() : 'pyscf';
        currentStructureLabel = 'pasted structure';
        currentStructureProvider = async () => ({ format: fmt, content, label: currentStructureLabel });
        updateRunButtonState();
        const canvas = document.getElementById('miew-container');
        if (canvas) {
          if (fmt === 'pyscf') {
            // Hide viewer when using PySCF atom text — no visual representation
            canvas.classList.add('rf-hidden');
          } else {
            canvas.classList.remove('rf-hidden');
          }
        }
        if (fmt === 'xyz' || fmt === 'pdb') {
          const file = new File([content], `pasted.${fmt}`, { type: 'text/plain' });
          const p = loadSource(viewer, file, `pasted.${fmt}`);
          if (p && typeof p.catch === 'function') {
            p.catch(() => {});
          }
          setStatus('Structure loaded from pasted text.', 'idle');
        } else {
          // Best-effort scene clear so no reference structure is visible
          try {
            if (typeof viewer.load === 'function') {
              viewer.load(null);
            }
          } catch (_) {}
          setStatus('Using pasted PySCF atom text for DFT. Viewer hidden.', 'idle');
        }
      });
    }

    const fileInput = document.getElementById('structure-file');
    if (fileInput) {
      fileInput.addEventListener('change', (event) => {
        const inputEl = event.currentTarget;
        if (!inputEl) {
          return;
        }
        const fileList = inputEl.files;
        if (!fileList || !fileList.length) {
          return;
        }
        const file = fileList[0];

        const promise = handleFile(viewer, file);
        if (promise && typeof promise.catch === 'function') {
          promise.catch(() => {}).finally(() => {
            inputEl.value = '';
          });
        } else {
          inputEl.value = '';
        }
      });
    }

    setupDragAndDrop(viewer);
    setupPresetPalette(viewer);

    const resetBtn = document.getElementById('viewer-reset');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        if (typeof viewer.resetView === 'function') {
          viewer.resetView();
          setStatus('View reset to default.', 'idle');
        } else {
          setStatus('Reset view is not available in this build.', 'error');
        }
      });
    }

    const fitBtn = document.getElementById('viewer-fit');
    if (fitBtn) {
      fitBtn.addEventListener('click', () => {
        if (typeof viewer.resetView === 'function') {
          viewer.resetView();
        }
        setStatus('Structure fitted to window.', 'idle');
      });
    }

    const toggleAxesBtn = document.getElementById('toggle-axes');
    if (toggleAxesBtn) {
      toggleAxesBtn.addEventListener('click', () => {
        const nextAxes = !viewer.get('axes');
        const nextFps = !viewer.get('fps');
        viewer.set({ axes: nextAxes, fps: nextFps });
        setStatus(`Axes ${nextAxes ? 'shown' : 'hidden'}.`, 'idle');
      });
    }

    const toggleRotationBtn = document.getElementById('toggle-rotation');
    if (toggleRotationBtn) {
      toggleRotationBtn.addEventListener('click', () => {
        const current = viewer.get('autoRotation');
        const next = current ? 0 : 0.4;
        viewer.set('autoRotation', next);
        setStatus(`Auto rotation ${next ? 'enabled' : 'disabled'}.`, 'idle');
      });
    }
  }

  function setupPresetPalette(viewer) {
    const container = document.getElementById('preset-list');
    if (!container) {
      return;
    }

    container.innerHTML = '';
    PRESET_MOLECULES.forEach((preset) => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'rf-preset-chip';
      chip.draggable = true;
      chip.textContent = preset.label;

      chip.addEventListener('click', () => {
        const promise = handleText(viewer, preset.token);
        if (promise && typeof promise.catch === 'function') {
          promise.catch(() => {});
        }
      });

      chip.addEventListener('dragstart', (event) => {
        event.dataTransfer.effectAllowed = 'copy';
        event.dataTransfer.setData('text/plain', preset.token);
      });

      container.appendChild(chip);
    });
  }

  // Additional vibrational/electronic analysis UI removed
})();
