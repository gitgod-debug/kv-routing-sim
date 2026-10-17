// Controller for run.html: reads the form, runs the simulator in a web worker, renders the report.
(function () {
  "use strict";

  const POLICIES = [
    { name: "round-robin", label: "Round-robin", standard: true },
    { name: "least-loaded", label: "Least loaded", standard: true },
    { name: "power-of-two", label: "Power of two choices", standard: true },
    { name: "consistent-hash-prefix", label: "Consistent hash on the prefix", standard: true },
    { name: "radix-affinity", label: "Prefix affinity", standard: true },
    { name: "radix-affinity-balanced", label: "Prefix affinity with a load weight", standard: true },
    { name: "session-sticky", label: "Session sticky", standard: true },
    { name: "lmetric", label: "LMetric, ported", standard: false },
    { name: "smetric-local", label: "SMetric's session rule, ported", standard: false },
  ];
  const STAGES = {
    sweep: "Simulating",
    spread: "Sizing on each calibration repeat",
    edge: "Checking the margin to saturation",
  };

  const $ = (id) => document.getElementById(id);
  let worker = null;
  let lastResult = null;
  let startedAt = 0;
  let activeSettings = null;

  function init() {
    $("policies").innerHTML = POLICIES.map((p) =>
      `<label class="check"><input type="checkbox" id="policy-${p.name}"${p.standard ? " checked" : ""}> ` +
      `${p.label}</label>`).join("");
    document.querySelectorAll('input[name="traffic"], input[name="profile"]')
      .forEach((input) => input.addEventListener("change", syncChoices));
    $("form").addEventListener("submit", (event) => { event.preventDefault(); start(); });
    $("cancel").addEventListener("click", cancel);
    $("download-json").addEventListener("click", () =>
      download("kvsim-report.json", JSON.stringify(lastResult.report, null, 2), "application/json"));
    $("download-md").addEventListener("click", () =>
      download("kvsim-report.md", lastResult.markdown, "text/markdown"));
    syncChoices();
  }

  function choice(name) {
    return document.querySelector(`input[name="${name}"]:checked`).value;
  }

  function syncChoices() {
    const fromFile = choice("traffic") === "file";
    $("generate-fields").hidden = fromFile;
    $("file-fields").hidden = !fromFile;
    $("file-hint").hidden = !fromFile;
    $("profile-field").hidden = choice("profile") !== "file";
  }

  function number(id) {
    const value = $(id).value.trim();
    return value === "" ? null : Number(value);
  }

  function readSettings() {
    const traffic = choice("traffic") === "file"
      ? { source: "file", format: $("trace-format").value }
      : {
        source: "generate", sessions: number("sessions"), turns: number("turns"),
        rate: number("rate"), sharing: number("sharing"), system_prompts: number("system-prompts"),
        seed: number("seed"),
      };
    const chosen = POLICIES.filter((p) => $(`policy-${p.name}`).checked).map((p) => p.name);
    const standard = POLICIES.filter((p) => p.standard).map((p) => p.name);
    const allStandard = chosen.length === standard.length && standard.every((n) => chosen.includes(n));
    return {
      traffic,
      profile: { source: choice("profile") },
      policies: allStandard ? "all" : chosen.join(","),
      replicas: $("replicas").value.trim(),
      seeds: number("seeds"),
      ttft_slo_ms: number("ttft"),
      tpot_slo_ms: number("tpot") ?? 0,
      gpu_price_per_hour_usd: number("price"),
      edge_check: $("edge").checked,
      spread: $("spread").checked,
    };
  }

  function validate(settings) {
    if (settings.traffic.source === "file" && !$("trace-file").files.length) {
      return "Choose a log file, or switch to generated traffic.";
    }
    if (settings.profile.source === "file" && !$("profile-file").files.length) {
      return "Choose a profile file, or use the measured T4.";
    }
    if (!settings.policies) return "Choose at least one routing policy.";
    return null;
  }

  function start() {
    const settings = readSettings();
    const problem = validate(settings);
    showError(problem);
    if (problem) return;
    activeSettings = settings;
    setRunning(true);
    startedAt = performance.now();
    worker = worker || createWorker();
    worker.postMessage({
      settings,
      traceFile: settings.traffic.source === "file" ? $("trace-file").files[0] : null,
      profileFile: settings.profile.source === "file" ? $("profile-file").files[0] : null,
    });
  }

  function createWorker() {
    const w = new Worker("worker.js");
    w.addEventListener("message", (event) => handleMessage(event.data));
    w.addEventListener("error", (event) => {
      finish();
      showError(`The simulator could not start: ${event.message || "unknown error"}.`);
      worker = null;
    });
    return w;
  }

  function handleMessage(message) {
    if (message.type === "status") {
      setStatus(message.text);
    } else if (message.type === "progress") {
      const bar = $("progress");
      bar.hidden = false;
      bar.max = Math.max(message.total, 1);
      bar.value = message.done;
      setStatus(`${STAGES[message.stage] || message.stage}: ${message.done} of ${message.total} runs`);
    } else if (message.type === "result") {
      finish();
      lastResult = message.result;
      setStatus(`Finished in ${elapsed()}.`);
      renderResult(message.result, activeSettings);
    } else if (message.type === "error") {
      finish();
      showError(message.message);
    }
  }

  function cancel() {
    if (worker) worker.terminate();
    worker = null;
    finish();
    setStatus("Cancelled. The next run reloads Python, which your browser has cached.");
  }

  function setRunning(running) {
    $("run").disabled = running;
    $("cancel").hidden = !running;
    $("progress").hidden = !running;
    $("progress").value = 0;
    if (running) setStatus("Starting…");
  }

  function finish() {
    setRunning(false);
  }

  function setStatus(text) {
    $("status").textContent = text;
  }

  function showError(text) {
    $("error").hidden = !text;
    $("error").textContent = text || "";
  }

  function elapsed() {
    const seconds = Math.round((performance.now() - startedAt) / 1000);
    return seconds < 60 ? `${seconds} s` : `${Math.floor(seconds / 60)} min ${seconds % 60} s`;
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
  }

  function labelOf(name) {
    const policy = POLICIES.find((p) => p.name === name);
    return policy ? policy.label : name;
  }

  function replicas(n) {
    return `${n} replica${n === 1 ? "" : "s"}`;
  }

  function rupees(value) {
    return value === null || value === undefined ? "—" : `₹${value.toFixed(2)}`;
  }

  function milliseconds(value) {
    if (value === null || value === undefined) return "—";
    return value >= 10000 ? `${(value / 1000).toFixed(1)} s` : `${Math.round(value).toLocaleString()} ms`;
  }

  function renderResult(result, settings) {
    const report = result.report;
    const rec = report.recommendation;
    const chosen = rec.recommended;
    const swept = Math.max(...report.points.map((p) => p.n_replicas));
    $("results").hidden = false;

    if (chosen && chosen.meets_slo) {
      $("answer").innerHTML = `Use <strong>${escapeHtml(labelOf(chosen.policy))}</strong> on ` +
        `<strong>${replicas(chosen.n_replicas)}</strong>: ${rupees(chosen.inr_per_1k_requests)} per ` +
        `1,000 requests, p95 first token ${milliseconds(chosen.p95_ttft_ms)}.`;
    } else if (chosen) {
      $("answer").innerHTML = `No configuration meets the target on up to ${replicas(swept)}. ` +
        `The lowest p95 first-token latency is ${escapeHtml(labelOf(chosen.policy))} on ` +
        `${replicas(chosen.n_replicas)}. Try more replicas, a lighter load, or a looser target.`;
    } else {
      $("answer").textContent = "No configuration could be simulated.";
    }

    const info = result.notes.filter((n) => !n.warning).map((n) => n.text);
    const cautions = result.notes.filter((n) => n.warning).map((n) => n.text);
    if (rec.WARNING) cautions.push(rec.WARNING);
    if (settings.profile.source === "shipped") {
      cautions.push("Every figure is for a Tesla T4 serving Qwen3-4B. For your hardware, run " +
        "kvsim calibrate against your vLLM server and load the profile it writes.");
    } else if (settings.gpu_price_per_hour_usd === null) {
      cautions.push("Rupee figures use the T4's price. Enter your GPU's price per hour for your costs.");
    }
    if (report.replica_spread_note) cautions.push(report.replica_spread_note);
    $("notes").innerHTML = info.map((n) => `<p class="muted">${escapeHtml(n)}</p>`).join("") +
      (cautions.length
        ? `<div class="note caveats">${cautions.map((n) => `<p>${escapeHtml(n)}</p>`).join("")}</div>`
        : "");

    $("summary").innerHTML = summaryTable(report.points, chosen, swept);
    $("target").textContent = targetSentence(settings);
    $("spread-section").innerHTML = spreadTable(report.replica_spread);
    $("table").innerHTML = fullTable(report.points, chosen);
    $("command").textContent = result.command;
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function targetSentence(settings) {
    const parts = [`p95 time to first token at most ${settings.ttft_slo_ms.toLocaleString()} ms`];
    if (settings.tpot_slo_ms > 0) {
      parts.push(`p95 time per output token at most ${settings.tpot_slo_ms.toLocaleString()} ms`);
    }
    return `Target: ${parts.join(" and ")}. Costs are rupees per 1,000 successful requests.`;
  }

  function summaryTable(points, chosen, swept) {
    const byPolicy = new Map();
    for (const p of points) {
      if (!byPolicy.has(p.policy)) byPolicy.set(p.policy, null);
      const best = byPolicy.get(p.policy);
      if (p.meets_slo && (!best || p.n_replicas < best.n_replicas)) byPolicy.set(p.policy, p);
    }
    let rows = "<thead><tr><th>Policy</th><th class=\"num\">Replicas</th><th class=\"num\">₹ per 1k</th>" +
      "<th class=\"num\">p95 first token</th><th class=\"num\">p95 per token</th>" +
      "<th class=\"num opt\">Hit rate</th></tr></thead><tbody>";
    for (const [policy, best] of byPolicy) {
      const isChosen = chosen && best && chosen.policy === policy && chosen.n_replicas === best.n_replicas;
      rows += `<tr${isChosen ? " class=\"chosen\"" : ""}><td>${escapeHtml(labelOf(policy))}</td>`;
      rows += best
        ? `<td class="num">${best.n_replicas}</td><td class="num">${rupees(best.inr_per_1k_requests)}</td>` +
          `<td class="num">${milliseconds(best.p95_ttft_ms)}</td>` +
          `<td class="num">${milliseconds(best.p95_tpot_ms)}</td><td class="num opt">${best.hit_rate.toFixed(3)}</td>`
        : `<td class="num">more than ${swept}</td><td class="num">—</td><td class="num">—</td>` +
          "<td class=\"num\">—</td><td class=\"num opt\">—</td>";
      rows += "</tr>";
    }
    return rows + "</tbody>";
  }

  function fullTable(points, chosen) {
    let rows = "<thead><tr><th>Policy</th><th class=\"num\">Replicas</th><th class=\"num\">Hit rate</th>" +
      "<th class=\"num opt\">p50 first token</th><th class=\"num\">p95 first token</th>" +
      "<th class=\"num\">p95 per token</th><th class=\"num\">₹ per 1k</th><th class=\"num\">Imbalance</th>" +
      "<th>Meets target</th></tr></thead><tbody>";
    for (const p of points) {
      const isChosen = chosen && chosen.policy === p.policy && chosen.n_replicas === p.n_replicas;
      rows += `<tr${isChosen ? " class=\"chosen\"" : ""}><td>${escapeHtml(labelOf(p.policy))}</td>` +
        `<td class="num">${p.n_replicas}</td><td class="num">${p.hit_rate.toFixed(3)}</td>` +
        `<td class="num opt">${milliseconds(p.p50_ttft_ms)}</td><td class="num">${milliseconds(p.p95_ttft_ms)}</td>` +
        `<td class="num">${milliseconds(p.p95_tpot_ms)}</td><td class="num">${rupees(p.inr_per_1k_requests)}</td>` +
        `<td class="num">${p.load_imbalance.toFixed(2)}</td>` +
        `<td>${p.meets_slo ? '<span class="ok">yes</span>' : '<span class="no">no</span>'}</td></tr>`;
    }
    return rows + "</tbody>";
  }

  function spreadTable(spread) {
    if (!spread || !spread.replicas) return "";
    const count = (n) => (n === null ? `more than ${spread.swept_up_to}` : String(n));
    let html = "<h2>How firm the replica counts are</h2>" +
      `<p>The calibration measured every timing cell ${spread.repeats} times. Sizing on each repeat ` +
      "alone gives:</p><div class=\"scroll\"><table><thead><tr><th>Policy</th>" +
      "<th class=\"num\">All repeats</th><th>Each repeat alone</th></tr></thead><tbody>";
    for (const [policy, row] of Object.entries(spread.replicas)) {
      html += `<tr><td>${escapeHtml(labelOf(policy))}</td><td class="num">${count(row.pooled)}</td>` +
        `<td>${row.repeats.map(count).join(" · ")}</td></tr>`;
    }
    return html + "</tbody></table></div>";
  }

  function download(filename, text, type) {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  init();
})();
