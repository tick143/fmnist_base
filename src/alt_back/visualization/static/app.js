const svgNS = "http://www.w3.org/2000/svg";

const DATASET_KEYS = [
  "threshold",
  "feature_min",
  "feature_max",
  "noise_std",
  "num_train",
  "num_test",
  "batch_size",
  "seed",
];

const TRAINER_NUMERIC_KEYS = [
  "release_rate",
  "reward_gain",
  "base_release",
  "decay",
  "temperature",
  "efficiency_bonus",
  "column_competition",
  "noise_std",
  "mass_budget",
  "target_gain",
  "affinity_strength",
  "affinity_decay",
  "affinity_temperature",
  "sign_consistency_strength",
  "sign_consistency_momentum",
  "spike_threshold",
  "spike_temperature",
  "snapshot_interval",
  "evaluate_interval",
];

const colors = {
  positive: "#f97316",
  negative: "#38bdf8",
  edgeBase: "rgba(148, 163, 184, 0.35)",
  nodeBase: "rgba(15, 23, 42, 0.9)",
  nodeSpike: h => `rgba(16, 185, 129, ${Math.min(0.15 + h * 0.85, 1).toFixed(2)})`,
};

const state = {
  topology: null,
  layers: [],
  edges: {},
  connections: [],
  autoTimer: null,
  autoInterval: 900,
  busy: false,
  config: null,
  configPath: null,
  configMode: "defaults",
  settingsCollapsed: false,
};

const elements = {
  svg: document.getElementById("network"),
  stepBtn: document.getElementById("step-btn"),
  autoBtn: document.getElementById("auto-btn"),
  resetBtn: document.getElementById("reset-btn"),
  reloadBtn: document.getElementById("reload-btn"),
  applyBtn: document.getElementById("apply-btn"),
  seedInput: document.getElementById("seed-input"),
  status: document.getElementById("status"),
  configSource: document.getElementById("config-source"),
  autoSpeed: document.getElementById("auto-speed"),
  autoSpeedValue: document.getElementById("auto-speed-value"),
  metricStep: document.getElementById("metric-step"),
  metricLoss: document.getElementById("metric-loss"),
  metricAcc: document.getElementById("metric-acc"),
  metricEval: document.getElementById("metric-eval"),
  extrasList: document.getElementById("extras-list"),
  inputsView: document.getElementById("inputs-view"),
  predictionsView: document.getElementById("predictions-view"),
  spikesView: document.getElementById("spikes-view"),
  weightsContainer: document.getElementById("weights-container"),
  dataset: {},
  trainer: {},
  settingsSection: document.querySelector(".settings"),
  settingsToggle: document.querySelector(".settings-toggle"),
  settingsContent: document.querySelector(".settings-content"),
  tabButtons: Array.from(document.querySelectorAll(".tab-button")),
  tabPanels: Array.from(document.querySelectorAll(".tab-panel")),
};

elements.useTargetBonus = document.getElementById("trainer-use_target_bonus");
elements.signedWeights = document.getElementById("trainer-signed_weights");
elements.hiddenLayout = document.getElementById("trainer-hidden_layers");

function buildInputMap(target, keys) {
  keys.forEach(key => {
    const element = document.getElementById(`${target}-${key}`);
    if (element) {
      elements[target][key] = element;
    }
  });
}

buildInputMap("dataset", DATASET_KEYS);
buildInputMap("trainer", TRAINER_NUMERIC_KEYS);

if (elements.autoSpeed) {
  elements.autoSpeed.value = String(state.autoInterval);
  updateAutoSpeedDisplay(state.autoInterval);
  elements.autoSpeed.addEventListener("input", handleAutoSpeedChange);
}

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const payload = await res.json();
      if (payload?.detail) {
        detail = payload.detail;
      }
    } catch (_) {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

function setStatus(message, tone = "info") {
  elements.status.textContent = message;
  elements.status.dataset.tone = tone;
}

function updateAutoSpeedDisplay(value) {
  if (elements.autoSpeedValue) {
    elements.autoSpeedValue.textContent = `${value} ms`;
  }
}

function updateConfigSource() {
  let text = "Source: defaults";
  if (state.configMode === "file" && state.configPath) {
    text = `Source: ${state.configPath}`;
  } else if (state.configMode === "override") {
    text = "Source: live overrides (unsaved)";
  }
  elements.configSource.textContent = text;
}

function updateReloadAvailability() {
  elements.reloadBtn.disabled = !state.configPath;
}

function setBusy(flag) {
  state.busy = flag;
  elements.stepBtn.disabled = flag;
  elements.resetBtn.disabled = flag;
  if (!state.autoTimer) {
    elements.autoBtn.disabled = flag;
  }
}

function stopAuto() {
  if (state.autoTimer) {
    clearInterval(state.autoTimer);
    state.autoTimer = null;
    elements.autoBtn.textContent = "Auto Step";
    elements.autoBtn.dataset.running = "false";
    elements.autoBtn.disabled = false;
  }
}

function clearOutputs() {
  elements.metricStep.textContent = "-";
  elements.metricLoss.textContent = "-";
  elements.metricAcc.textContent = "-";
  elements.metricEval.textContent = "-";
  elements.extrasList.innerHTML = "";
  elements.inputsView.textContent = "(run a step)";
  elements.predictionsView.textContent = "-";
  elements.spikesView.textContent = "-";
  elements.weightsContainer.innerHTML = "";
  Object.values(state.edges).forEach(group => {
    Object.values(group).forEach(path => {
      path.classList.remove("highlight");
      path.setAttribute("stroke-width", "1");
      path.setAttribute("stroke", colors.edgeBase);
    });
  });
  state.layers.forEach((nodes, idx) => {
    if (idx === 0 || idx === state.layers.length - 1) {
      return;
    }
    nodes.forEach(node => {
      node.circle.style.fill = colors.nodeBase;
    });
  });
}

function layerPositions(count, height, padding) {
  if (count === 0) {
    return [];
  }
  const availableHeight = height - padding * 2;
  return Array.from({ length: count }, (_, idx) => {
    if (count === 1) {
      return padding + availableHeight / 2;
    }
    return padding + (idx * availableHeight) / (count - 1);
  });
}

function buildNetwork(topology) {
  state.topology = topology;
  state.connections = topology.connections || [];
  const layerSizes = topology.layer_sizes || [];

  const svg = elements.svg;
  svg.innerHTML = "";
  state.layers = layerSizes.map(() => []);
  state.edges = {};

  if (layerSizes.length === 0) {
    return;
  }

  const viewBox = svg.getAttribute("viewBox")?.split(" ").map(Number);
  const viewWidth = viewBox && viewBox.length === 4 ? viewBox[2] : 800;
  const viewHeight = viewBox && viewBox.length === 4 ? viewBox[3] : 480;
  const xPadding = 80;
  const yPadding = 40;
  const totalLayers = layerSizes.length;
  const availableWidth = viewWidth - xPadding * 2;

  const layerX = layerSizes.map((_, idx) => {
    if (totalLayers === 1) {
      return xPadding + availableWidth / 2;
    }
    return xPadding + (idx * availableWidth) / (totalLayers - 1);
  });

  layerSizes.forEach((size, layerIdx) => {
    const ys = layerPositions(size, viewHeight, yPadding);
    ys.forEach((y, neuronIdx) => {
      const x = layerX[layerIdx];
      const circle = document.createElementNS(svgNS, "circle");
      circle.classList.add("node");
      circle.setAttribute("cx", x);
      circle.setAttribute("cy", y);
      circle.setAttribute("r", layerIdx === 0 || layerIdx === totalLayers - 1 ? 18 : 20);
      circle.style.fill = colors.nodeBase;
      svg.appendChild(circle);

      const text = document.createElementNS(svgNS, "text");
      text.setAttribute("x", x);
      text.setAttribute("y", y + 4);
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("fill", "#cbd5f5");
      text.setAttribute("font-size", "11");
      let prefix;
      if (layerIdx === 0) {
        prefix = "I";
      } else if (layerIdx === totalLayers - 1) {
        prefix = "O";
      } else {
        prefix = `H${layerIdx - 1}`;
      }
      text.textContent = `${prefix}${neuronIdx}`;
      svg.appendChild(text);

      state.layers[layerIdx].push({ circle, index: neuronIdx, x, y });
    });
  });

  state.connections.forEach(connection => {
    const fromNodes = state.layers[connection.from] || [];
    const toNodes = state.layers[connection.to] || [];
    const group = {};
    fromNodes.forEach((fromNode, sourceIdx) => {
      toNodes.forEach((toNode, targetIdx) => {
        const path = document.createElementNS(svgNS, "path");
        path.classList.add("edge");
        path.setAttribute("stroke", colors.edgeBase);
        const controlX = (fromNode.x + toNode.x) / 2;
        const d = `M ${fromNode.x} ${fromNode.y} C ${controlX} ${fromNode.y}, ${controlX} ${toNode.y}, ${toNode.x} ${toNode.y}`;
        path.setAttribute("d", d);
        svg.insertBefore(path, svg.firstChild);
        group[`${targetIdx}-${sourceIdx}`] = path;
      });
    });
    state.edges[connection.name] = group;
  });
}

function updateEdgeGroup(connectionName, matrix, deltas) {
  const edgeGroup = state.edges[connectionName];
  if (!edgeGroup || !Array.isArray(matrix) || matrix.length === 0) {
    return;
  }

  const maxMagnitude = matrix.reduce(
    (acc, row) => row.reduce((inner, value) => Math.max(inner, Math.abs(value)), acc),
    0,
  ) || 1e-6;

  matrix.forEach((row, rowIdx) => {
    row.forEach((value, colIdx) => {
      const edge = edgeGroup[`${rowIdx}-${colIdx}`];
      if (!edge) {
        return;
      }
      const width = 1 + (Math.abs(value) / maxMagnitude) * 5;
      edge.setAttribute("stroke-width", width.toFixed(2));
      edge.setAttribute("stroke", value >= 0 ? colors.positive : colors.negative);

      let delta = 0;
      if (Array.isArray(deltas) && Array.isArray(deltas[rowIdx])) {
        delta = deltas[rowIdx][colIdx] ?? 0;
      }
      if (Math.abs(delta) > 1e-4) {
        edge.classList.add("highlight");
      } else {
        edge.classList.remove("highlight");
      }
    });
  });
}

function updateHiddenNodes(spikeRatesByLayer) {
  state.layers.forEach((nodes, layerIdx) => {
    if (layerIdx === 0 || layerIdx === state.layers.length - 1) {
      return;
    }
    nodes.forEach(node => {
      node.circle.style.fill = colors.nodeBase;
    });
  });

  if (!Array.isArray(spikeRatesByLayer) || spikeRatesByLayer.length === 0) {
    return;
  }

  spikeRatesByLayer.forEach((layerRates, hiddenIdx) => {
    const nodes = state.layers[hiddenIdx + 1] || [];
    if (!Array.isArray(layerRates) || layerRates.length === 0) {
      return;
    }
    const neuronCount = nodes.length;
    if (neuronCount === 0) {
      return;
    }
    const totals = new Array(neuronCount).fill(0);
    layerRates.forEach(sample => {
      sample.forEach((value, neuronIdx) => {
        if (neuronIdx < neuronCount) {
          totals[neuronIdx] += value;
        }
      });
    });
    const samples = Math.max(layerRates.length, 1);
    nodes.forEach((node, neuronIdx) => {
      const avg = Math.max(0, Math.min(1, totals[neuronIdx] / samples));
      node.circle.style.fill = colors.nodeSpike(avg);
    });
  });
}

function buildTable(title, weightMatrix, bias) {
  const block = document.createElement("div");
  block.className = "table-block";
  const heading = document.createElement("h2");
  heading.textContent = title;
  block.appendChild(heading);

  if (!Array.isArray(weightMatrix) || weightMatrix.length === 0) {
    const placeholder = document.createElement("p");
    placeholder.className = "placeholder";
    placeholder.textContent = "No weights available";
    block.appendChild(placeholder);
    return block;
  }

  const table = document.createElement("table");
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  headerRow.appendChild(document.createElement("th"));
  weightMatrix[0].forEach((_, idx) => {
    const th = document.createElement("th");
    th.textContent = `In ${idx}`;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  weightMatrix.forEach((row, rowIdx) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = `Out ${rowIdx}`;
    tr.appendChild(th);
    row.forEach(value => {
      const td = document.createElement("td");
      td.textContent = Number(value).toFixed(3);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  block.appendChild(table);

  if (Array.isArray(bias) && bias.length > 0) {
    const biasRow = document.createElement("p");
    biasRow.className = "bias-row";
    biasRow.textContent = `Bias: ${bias.map(v => Number(v).toFixed(3)).join(", ")}`;
    block.appendChild(biasRow);
  }

  return block;
}

function updateTables(weights) {
  elements.weightsContainer.innerHTML = "";
  if (!weights || !state.topology) {
    return;
  }

  state.topology.connections.forEach(connection => {
    const node = weights[connection.name];
    if (!node) {
      return;
    }
    const block = buildTable(connection.label, node.weight, node.bias);
    elements.weightsContainer.appendChild(block);
  });
}

function formatBatch(inputs, preds, targets) {
  if (!Array.isArray(inputs) || inputs.length === 0) {
    return "(no batch)";
  }
  const lines = [];
  const limit = Math.min(inputs.length, 6);
  for (let i = 0; i < limit; i += 1) {
    const values = inputs[i].map(v => Number(v).toFixed(2)).join(", ");
    lines.push(`${i}: [${values}] → y=${targets[i]} | ŷ=${preds[i]}`);
  }
  if (inputs.length > limit) {
    lines.push(`… ${inputs.length - limit} more`);
  }
  return lines.join("\n");
}

function formatSpikes(spikeRatesByLayer) {
  if (!Array.isArray(spikeRatesByLayer) || spikeRatesByLayer.length === 0) {
    return "-";
  }
  const sections = [];
  spikeRatesByLayer.forEach((layerRates, layerIdx) => {
    if (!Array.isArray(layerRates) || layerRates.length === 0) {
      return;
    }
    const neuronCount = layerRates[0]?.length ?? 0;
    const totals = new Array(neuronCount).fill(0);
    layerRates.forEach(sample => {
      sample.forEach((value, idx) => {
        totals[idx] += value;
      });
    });
    const samples = Math.max(layerRates.length, 1);
    const averages = totals.map(total => (total / samples).toFixed(3));
    sections.push(`Layer ${layerIdx}: ${averages.join(", ")}`);
  });
  return sections.join("\n\n");
}

function updateExtras(extras) {
  elements.extrasList.innerHTML = "";
  Object.entries(extras ?? {}).forEach(([key, value]) => {
    const li = document.createElement("li");
    const display = typeof value === "number" ? value.toFixed(3) : value;
    li.textContent = `${key}: ${display}`;
    elements.extrasList.appendChild(li);
  });
}

function updateViews(data) {
  elements.metricStep.textContent = data.step;
  elements.metricLoss.textContent = data.loss.toFixed(4);
  elements.metricAcc.textContent = `${data.batch_accuracy.toFixed(2)}%`;
  if (data.eval) {
    elements.metricEval.textContent = `${data.eval.accuracy.toFixed(2)}%`;
  }
  updateExtras(data.extras);

  if (state.topology) {
    state.topology.connections.forEach(connection => {
      const weightNode = data.weights?.[connection.name]?.weight;
      const deltaNode = data.weight_deltas?.[connection.name]?.weight;
      updateEdgeGroup(connection.name, weightNode, deltaNode);
    });
  }

  updateHiddenNodes(data.hidden_spike_rates);
  updateTables(data.weights);

  elements.inputsView.textContent = formatBatch(data.inputs, data.predictions, data.targets);
  elements.predictionsView.textContent = `Predictions: ${data.predictions.join(", ")}\nTargets: ${data.targets.join(", ")}`;
  elements.spikesView.textContent = formatSpikes(data.hidden_spike_rates);
}

async function runStep() {
  if (state.busy) {
    return;
  }
  setBusy(true);
  setStatus("Running step…");
  try {
    const data = await fetchJSON("/api/step", { method: "POST" });
    updateViews(data);
    const tone = data.snapshot_captured ? "success" : "info";
    const snapshotNote = data.snapshot_captured ? "" : " (visuals unchanged)";
    const evalNote = data.evaluation_captured ? "" : " – eval deferred";
    setStatus(`Completed step ${data.step}${snapshotNote}${evalNote}`, tone);
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`, "error");
    stopAuto();
  } finally {
    setBusy(false);
  }
}

async function reset() {
  if (state.busy) {
    return;
  }
  stopAuto();
  setBusy(true);
  const seedVal = elements.seedInput.value;
  setStatus("Resetting…");
  try {
    const payload = seedVal ? { seed: Number(seedVal) } : {};
    await fetchJSON("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    clearOutputs();
    buildNetwork(state.topology || { layer_sizes: [], connections: [] });
    setStatus("Reset complete", "success");
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`, "error");
  } finally {
    setBusy(false);
  }
}

function toggleAuto() {
  if (state.autoTimer) {
    stopAuto();
    setStatus("Auto stepping stopped");
    return;
  }
  elements.autoBtn.textContent = "Stop";
  elements.autoBtn.dataset.running = "true";
  elements.autoBtn.disabled = false;
  setStatus("Auto stepping…");
  state.autoTimer = setInterval(runStep, state.autoInterval);
}

function handleAutoSpeedChange(event) {
  const value = Number.parseInt(event.target.value, 10);
  if (Number.isNaN(value)) {
    return;
  }
  state.autoInterval = value;
  updateAutoSpeedDisplay(value);
  if (state.autoTimer) {
    clearInterval(state.autoTimer);
    state.autoTimer = setInterval(runStep, state.autoInterval);
  }
}

function populateSettings(config) {
  DATASET_KEYS.forEach(key => {
    const element = elements.dataset[key];
    if (element) {
      const value = config.dataset?.[key];
      element.value = value ?? "";
    }
  });

  TRAINER_NUMERIC_KEYS.forEach(key => {
    const element = elements.trainer[key];
    if (element) {
      const value = config[key];
      element.value = value ?? "";
    }
  });

  if (elements.hiddenLayout) {
    const layout = config.hidden_layers ?? [];
    elements.hiddenLayout.value = Array.isArray(layout) ? layout.join(", ") : "";
  }

  if (elements.useTargetBonus) {
    const enabled = config.use_target_bonus !== false;
    elements.useTargetBonus.checked = enabled;
  }

  if (elements.signedWeights) {
    const enabled = config.signed_weights !== false;
    elements.signedWeights.checked = enabled;
  }
}

function parseHiddenLayout(raw) {
  if (!raw) {
    return undefined;
  }
  const parts = raw.split(",").map(part => part.trim()).filter(Boolean);
  if (parts.length === 0) {
    return undefined;
  }
  const values = parts.map(part => {
    const parsed = Number.parseInt(part, 10);
    if (Number.isNaN(parsed)) {
      throw new Error(`Invalid hidden layer size: ${part}`);
    }
    return parsed;
  });
  if (values.length === 0) {
    return undefined;
  }
  return values;
}

function parseInput(element) {
  if (!element) {
    return undefined;
  }
  const raw = element.value.trim();
  if (raw === "") {
    return undefined;
  }
  const type = element.dataset.type || "float";
  if (type === "int") {
    const parsed = Number.parseInt(raw, 10);
    return Number.isNaN(parsed) ? undefined : parsed;
  }
  if (type === "list") {
    return parseHiddenLayout(raw);
  }
  const parsed = Number.parseFloat(raw);
  return Number.isNaN(parsed) ? undefined : parsed;
}

function gatherSettings() {
  const payload = { dataset: {} };
  DATASET_KEYS.forEach(key => {
    const value = parseInput(elements.dataset[key]);
    if (value !== undefined) {
      payload.dataset[key] = value;
    }
  });
  if (Object.keys(payload.dataset).length === 0) {
    delete payload.dataset;
  }

  TRAINER_NUMERIC_KEYS.forEach(key => {
    const value = parseInput(elements.trainer[key]);
    if (value !== undefined) {
      payload[key] = value;
    }
  });

  const hiddenLayout = parseInput(elements.hiddenLayout);
  if (Array.isArray(hiddenLayout)) {
    payload.hidden_layers = hiddenLayout;
  }

  if (elements.useTargetBonus) {
    payload.use_target_bonus = elements.useTargetBonus.checked;
  }

  if (elements.signedWeights) {
    payload.signed_weights = elements.signedWeights.checked;
  }

  return payload;
}

async function applySettings() {
  stopAuto();
  let payload;
  try {
    payload = gatherSettings();
  } catch (err) {
    setStatus(`Invalid settings: ${err.message}`, "error");
    return;
  }
  if (Object.keys(payload).length === 0) {
    setStatus("No changes to apply", "info");
    return;
  }
  elements.applyBtn.disabled = true;
  elements.reloadBtn.disabled = true;
  setStatus("Applying settings…");
  try {
    const response = await fetchJSON("/api/reconfigure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.config = response.config;
    state.configMode = "override";
    buildNetwork(response.topology);
    populateSettings(state.config);
    clearOutputs();
    updateConfigSource();
    setStatus("Settings applied. Network reset.", "success");
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`, "error");
  } finally {
    elements.applyBtn.disabled = false;
    updateReloadAvailability();
  }
}

async function reloadConfig() {
  if (!state.configPath) {
    setStatus("Server was started without a YAML file.", "error");
    return;
  }
  stopAuto();
  elements.applyBtn.disabled = true;
  elements.reloadBtn.disabled = true;
  setStatus("Reloading YAML…");
  try {
    const response = await fetchJSON("/api/reload", { method: "POST" });
    state.config = response.config;
    state.configMode = "file";
    buildNetwork(response.topology);
    populateSettings(state.config);
    clearOutputs();
    updateConfigSource();
    setStatus("Reloaded config from disk.", "success");
  } catch (err) {
    console.error(err);
    setStatus(`Error: ${err.message}`, "error");
  } finally {
    elements.applyBtn.disabled = false;
    updateReloadAvailability();
  }
}

function handleTabClick(event) {
  const target = event.currentTarget;
  const tab = target.dataset.tab;
  elements.tabButtons.forEach(button => {
    button.classList.toggle("active", button === target);
  });
  elements.tabPanels.forEach(panel => {
    panel.classList.toggle("hidden", panel.id !== `tab-${tab}`);
  });
}

function toggleSettings() {
  state.settingsCollapsed = !state.settingsCollapsed;
  elements.settingsSection.classList.toggle("collapsed", state.settingsCollapsed);
}

async function bootstrap() {
  setStatus("Loading configuration…");
  try {
    const configResponse = await fetchJSON("/api/config");
    state.config = configResponse.config;
    state.configPath = configResponse.config_path || null;
    state.configMode = configResponse.config_path ? "file" : "defaults";
    populateSettings(state.config);
    updateConfigSource();
    updateReloadAvailability();

    const topology = await fetchJSON("/api/topology");
    buildNetwork(topology);

    const existing = await fetchJSON("/api/state");
    if (existing?.step !== null && existing?.step !== undefined) {
      updateViews(existing);
    } else {
      clearOutputs();
    }
    setStatus("Ready");
  } catch (err) {
    console.error(err);
    setStatus(`Failed to initialise: ${err.message}`, "error");
  }
}

elements.stepBtn.addEventListener("click", runStep);
elements.resetBtn.addEventListener("click", reset);
elements.autoBtn.addEventListener("click", toggleAuto);
elements.applyBtn.addEventListener("click", applySettings);
elements.reloadBtn.addEventListener("click", reloadConfig);
elements.settingsToggle.addEventListener("click", toggleSettings);
elements.tabButtons.forEach(button => button.addEventListener("click", handleTabClick));

window.addEventListener("beforeunload", () => {
  stopAuto();
});

bootstrap();
