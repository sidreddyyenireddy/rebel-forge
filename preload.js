const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('desktopAPI', {
  runDFT: (payload) => ipcRenderer.invoke('run-dft', payload),
  runGeometryOptimization: (payload) =>
    ipcRenderer.invoke('run-dft', {
      ...(payload || {}),
      task: 'geomopt',
      optimize: true,
      includeForces: true,
    }),
  onDFTLog: (handler) => {
    if (typeof handler !== 'function') return () => {};
    const listener = (_event, msg) => {
      try { handler(msg); } catch (_) {}
    };
    ipcRenderer.on('dft-log', listener);
    return () => ipcRenderer.removeListener('dft-log', listener);
  },
});
