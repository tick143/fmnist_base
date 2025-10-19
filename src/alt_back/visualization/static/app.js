const svgNS = "http://www.w3.org/2000/svg";
const state = {
  topology: null,
  edges: { encoder: {}, decoder: {} },
  nodes: { input: [], hidden: [], output: [] },
  autoTimer: null,
  busy: false,
};

const elements = {
  svg: document.getElementById("network"),
  stepBtn: document.getElementById("step-btn"),
  autoBtn: document.getElementById("auto-btn"),
  resetBtn: document.getElementById("reset-btn"),
  seedInput: document.getElementById("seed-input"),
  status: document.getElementById("status"),
  metricStep: document.getElementById("metric-step"),
  metricLoss: document.getElementById("metric-loss"),
  metricAcc: document.getElementById("metric-acc"),
  metricEval: document.getElementById("metric-eval"),
  extrasList: document.getElementById("extras-list"),
  encoderTable: document.getElementById("encoder-table"),
  decoderTable: document.getElementById("decoder-table"),
  inputsView: document.getElementById("inputs-view"),
  predictionsView: document.getElementById("predictions-view"),
  spikesView: document.getElementById("spikes-view"),
};

const colors = {
  positive: "#f97316",
  negative: "#38bdf8",
  nodeBase: "rgba(15, 23, 42, 0.9)",
  nodeActive: "rgba(14, 165, 233, 0.8)",
  nodeSpike: h =>
    `rgba(16, 185, 129, ${Math.min(0.15 + h * 0.85, 1).toFixed(2)})`,
};

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

function setBusy(flag) {
  state.busy = flag;
  elements.stepBtn.disabled = flag;
  elements.resetBtn.disabled = flag;
  elements.autoBtn.disabled = flag && !state.autoTimer;
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

function buildNetwork(topology) {
  state.topology = topology;
  const svg = elements.svg;
  svg.innerHTML = "";
  state.edges = { encoder: {}, decoder: {} };
  state.nodes = { input: [], hidden: [], output: [] };

  const layerX = [110, 380, 650];
  const height = 440;

  function layerPositions(count) {
    const positions = [];
    for (let i = 0; i < count; i += 1) {
      const y = ((i + 1) / (count + 1)) * height + 20;
      positions.push(y);
    }
    return positions;
  }

  const inputYs = layerPositions(topology.input_neurons);
  const hiddenYs = layerPositions(topology.hidden_neurons);
  const outputYs = layerPositions(topology.output_neurons);

  function createNode(layer, index, x, y) {
    const circle = document.createElementNS(svgNS, "circle");
    circle.classList.add("node");
    circle.dataset.layer = layer;
    circle.dataset.index = index;
    circle.setAttribute("cx", x);
    circle.setAttribute("cy", y);
    circle.setAttribute("r", layer === "hidden" ? 20 : 18);
    svg.appendChild(circle);

    const label = document.createElementNS(svgNS, "text");
    label.setAttribute("x", x);
    label.setAttribute("y", y + 4);
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("fill", "#cbd5f5");
    label.setAttribute("font-size", "12");
    label.textContent = layer[0].toUpperCase() + index;
    svg.appendChild(label);

    state.nodes[layer].push({ circle, index });
    return { x, y };
  }

  const inputPositions = inputYs.map((y, i) =>
    createNode("input", i, layerX[0], y),
  );
  const hiddenPositions = hiddenYs.map((y, i) =>
    createNode("hidden", i, layerX[1], y),
  );
  const outputPositions = outputYs.map((y, i) =>
    createNode("output", i, layerX[2], y),
  );

  function createEdge(layer, source, target) {
    const path = document.createElementNS(svgNS, "path");
    path.classList.add("edge");
    const d = `M ${source.x} ${source.y} C ${(source.x + target.x) / 2} ${
      source.y
    }, ${(source.x + target.x) / 2} ${target.y}, ${target.x} ${target.y}`;
    path.setAttribute("d", d);
    svg.insertBefore(path, svg.firstChild);
    return path;
  }

  for (let h = 0; h < hiddenPositions.length; h += 1) {
    for (let i = 0; i < inputPositions.length; i += 1) {
      const edge = createEdge("encoder", inputPositions[i], hiddenPositions[h]);
      state.edges.encoder[`${h}-${i}`] = edge;
    }
  }

  for (let o = 0; o < outputPositions.length; o += 1) {
    for (let h = 0; h < hiddenPositions.length; h += 1) {
      const edge = createEdge("decoder", hiddenPositions[h], outputPositions[o]);
      state.edges.decoder[`${o}-${h}`] = edge;
    }
  }
}

function updateEdgeGroup(layerKey, matrix, deltas) {
  if (!matrix) {
    return;
  }
  const entries = matrix.flat();
  const maxMagnitude = Math.max(
    entries.reduce((acc, value) => Math.max(acc, Math.abs(value)), 0),
    1e-6,
  );

  for (let row = 0; row < matrix.length; row += 1) {
    for (let col = 0; col < matrix[row].length; col += 1) {
      const edge = state.edges[layerKey][`${row}-${col}`];
      if (!edge) {
        continue;
      }
      const value = matrix[row][col];
      const width = 1 + (Math.abs(value) / maxMagnitude) * 5;
      edge.setAttribute("stroke-width", width.toFixed(2));
      edge.setAttribute("stroke", value >= 0 ? colors.positive : colors.negative);

      const delta =
        deltas && deltas[row] ? deltas[row][col] ?? 0 : 0;
      if (Math.abs(delta) > 1e-4) {
        edge.classList.add("highlight");
      } else {
        edge.classList.remove("highlight");
      }
    }
  }
}

function updateHiddenNodes(spikeRates) {
  if (!Array.isArray(spikeRates) || spikeRates.length === 0) {
    return;
  }
  const neuronCount = spikeRates[0].length;
  const averages = new Array(neuronCount).fill(0);
  spikeRates.forEach(row => {
    row.forEach((value, idx) => {
      averages[idx] += value;
    });
  });
  for (let i = 0; i < neuronCount; i += 1) {
    averages[i] /= spikeRates.length;
  }
  state.nodes.hidden.forEach(({ circle, index }) => {
    const avg = Math.max(0, Math.min(1, averages[index]));
    circle.style.fill = colors.nodeSpike(avg);
  });
}

function updateTables(weights) {
  if (!weights?.encoder || !weights?.decoder) {
    return;
  }

  const encoderHead = elements.encoderTable.querySelector("thead");
  const encoderBody = elements.encoderTable.querySelector("tbody");
  encoderHead.innerHTML = `<tr><th></th>${weights.encoder.weight[0]
    .map((_, idx) => `<th>I${idx}</th>`)
    .join("")}</tr>`;
  encoderBody.innerHTML = weights.encoder.weight
    .map(
      (row, idx) =>
        `<tr><th>H${idx}</th>${row
          .map(value => `<td>${value.toFixed(3)}</td>`)
          .join("")}</tr>`,
    )
    .join("");

  const decoderHead = elements.decoderTable.querySelector("thead");
  const decoderBody = elements.decoderTable.querySelector("tbody");
  decoderHead.innerHTML = `<tr><th></th>${weights.decoder.weight[0]
    .map((_, idx) => `<th>H${idx}</th>`)
    .join("")}</tr>`;
  decoderBody.innerHTML = weights.decoder.weight
    .map(
      (row, idx) =>
        `<tr><th>O${idx}</th>${row
          .map(value => `<td>${value.toFixed(3)}</td>`)
          .join("")}</tr>`,
    )
    .join("");
}

function formatBatch(inputs, preds, targets) {
  const lines = [];
  const count = Math.min(inputs.length, 6);
  for (let i = 0; i < count; i += 1) {
    const values = inputs[i].map(v => v.toFixed(2)).join(", ");
    lines.push(
      `${i}: [${values}] → y=${targets[i]} | ŷ=${preds[i]}`,
    );
  }
  if (inputs.length > count) {
    lines.push(`… ${inputs.length - count} more`);
  }
  return lines.join("\n");
}

function formatSpikes(spikeRates) {
  if (!spikeRates?.length) {
    return "-";
  }
  const averages = new Array(spikeRates[0].length).fill(0);
  spikeRates.forEach(row => {
    row.forEach((value, idx) => {
      averages[idx] += value;
    });
  });
  return averages
    .map((value, idx) => `H${idx}: ${(value / spikeRates.length).toFixed(3)}`)
    .join("\n");
}

function updateExtras(extras) {
  elements.extrasList.innerHTML = "";
  Object.entries(extras ?? {}).forEach(([key, value]) => {
    const li = document.createElement("li");
    li.textContent = `${key}: ${value.toFixed(3)}`;
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

  updateEdgeGroup("encoder", data.weights?.encoder?.weight, data.weight_deltas?.encoder?.weight);
  updateEdgeGroup("decoder", data.weights?.decoder?.weight, data.weight_deltas?.decoder?.weight);
  updateHiddenNodes(data.hidden_spike_rates);
  updateTables(data.weights);

  elements.inputsView.textContent = formatBatch(
    data.inputs,
    data.predictions,
    data.targets,
  );
  elements.predictionsView.textContent = `Predictions: ${data.predictions.join(
    ", ",
  )}\nTargets: ${data.targets.join(", ")}`;
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
    setStatus(`Completed step ${data.step}`, "success");
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
    elements.metricStep.textContent = "-";
    elements.metricLoss.textContent = "-";
    elements.metricAcc.textContent = "-";
    elements.metricEval.textContent = "-";
    elements.extrasList.innerHTML = "";
    elements.inputsView.textContent = "(run a step)";
    elements.predictionsView.textContent = "-";
    elements.spikesView.textContent = "-";
    buildNetwork(state.topology);
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
  setStatus("Auto stepping…");
  state.autoTimer = setInterval(runStep, 900);
}

async function bootstrap() {
  setStatus("Loading topology…");
  try {
    const topology = await fetchJSON("/api/topology");
    buildNetwork(topology);
    const existing = await fetchJSON("/api/state");
    if (existing?.step !== null && existing?.step !== undefined) {
      updateViews(existing);
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

window.addEventListener("beforeunload", () => {
  stopAuto();
});

bootstrap();
