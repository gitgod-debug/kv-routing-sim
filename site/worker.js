// Web worker for run.html: loads Python through Pyodide, installs the kv-routing-sim wheel that is
// published next to this page, and runs one replay per message.
"use strict";

const PYODIDE_VERSION = "0.29.5";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const WORK_DIR = "/work";
const PROGRESS_INTERVAL_MS = 150;

importScripts(`${PYODIDE_URL}pyodide.js`);

let runtime = null;

function post(message) {
  self.postMessage(message);
}

async function boot() {
  post({ type: "status", text: "Downloading Python, about 10 MB the first time…" });
  const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });
  post({ type: "status", text: "Installing kv-routing-sim…" });
  await pyodide.loadPackage("micropip");
  const manifest = await (await fetch("wheel.json", { cache: "no-cache" })).json();
  const micropip = pyodide.pyimport("micropip");
  await micropip.install(new URL(manifest.wheel, self.location.href).href);
  pyodide.FS.mkdirTree(WORK_DIR);
  return { pyodide, entry: pyodide.pyimport("kv_routing_sim.browser") };
}

async function writeInput(pyodide, file) {
  const name = file.name.replace(/[^A-Za-z0-9._-]/g, "_");
  const path = `${WORK_DIR}/${name}`;
  pyodide.FS.writeFile(path, new Uint8Array(await file.arrayBuffer()));
  return { path, name: file.name };
}

function throttledProgress() {
  let last = 0;
  return (stage, done, total) => {
    const now = Date.now();
    if (done === 0 || done === total || now - last >= PROGRESS_INTERVAL_MS) {
      last = now;
      post({ type: "progress", stage, done, total });
    }
  };
}

function lastLine(text) {
  const lines = String(text).trim().split("\n");
  return lines[lines.length - 1];
}

self.onmessage = async (event) => {
  const { settings, traceFile, profileFile } = event.data;
  try {
    runtime = runtime || (await boot());
    const { pyodide, entry } = runtime;
    if (traceFile) Object.assign(settings.traffic, await writeInput(pyodide, traceFile));
    if (profileFile) Object.assign(settings.profile, await writeInput(pyodide, profileFile));
    post({
      type: "status",
      text: traceFile ? "Reading the log and simulating…" : "Generating traffic and simulating…",
    });
    const output = entry.run_json(JSON.stringify(settings), throttledProgress(), WORK_DIR);
    const result = JSON.parse(output);
    post(result.error ? { type: "error", message: result.error } : { type: "result", result });
  } catch (err) {
    post({ type: "error", message: `Unexpected error: ${lastLine(err && err.message ? err.message : err)}` });
  }
};
