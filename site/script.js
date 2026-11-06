const formatTime = (seconds) => {
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remainder = total % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
    : `${minutes}:${String(remainder).padStart(2, "0")}`;
};

const formatDuration = (seconds) => {
  const total = Math.round(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remainder = total % 60;
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return minutes > 0 ? `${minutes}m ${remainder}s` : `${remainder}s`;
};

function renderHighlights(data) {
  const highlights = data.highlights || [];
  const duration = Number(data.input_duration_seconds);
  const outputDuration = Number(data.output_duration_seconds);
  const timeline = document.querySelector("#timeline");
  const tableBody = document.querySelector("#highlights-table");

  document.querySelector("#input-duration").textContent = formatDuration(duration);
  document.querySelector("#output-duration").textContent = formatDuration(outputDuration);
  document.querySelector("#timeline-duration").textContent = formatTime(duration);
  document.querySelector("#timeline-end-label").textContent = formatTime(duration);
  timeline.replaceChildren();
  timeline.setAttribute(
    "aria-label",
    `${highlights.length} selected intervals across a ${formatDuration(duration)} input`
  );
  tableBody.replaceChildren();

  highlights.forEach((item, index) => {
    const start = Number(item.start_seconds);
    const end = Number(item.end_seconds);
    const bar = document.createElement("span");
    bar.className = "timeline-bar";
    bar.style.left = `${(start / duration) * 100}%`;
    bar.style.width = `${Math.max(((end - start) / duration) * 100, 0.35)}%`;
    bar.title = `${formatTime(start)}–${formatTime(end)} · mean score ${Number(item.mean_fused_score).toFixed(4)}`;
    timeline.append(bar);

    const row = document.createElement("tr");
    [
      `Moment ${String(index + 1).padStart(2, "0")}`,
      formatTime(start),
      formatTime(end),
      formatDuration(end - start),
      Number(item.mean_fused_score).toFixed(4)
    ].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    tableBody.append(row);
  });
}

function renderAblation(data) {
  const labels = {
    visual_only: "Visual only",
    audio_only: "Audio only",
    speech_only: "Speech only",
    full_fusion: "Fused"
  };
  const table = document.querySelector("#ablation-table");
  table.replaceChildren();
  Object.entries(labels).forEach(([key, label]) => {
    const metrics = data.variants?.[key]?.metrics;
    if (!metrics) {
      return;
    }
    const row = document.createElement("tr");
    [label, metrics.precision, metrics.recall, metrics.f1].forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = typeof value === "number" ? value.toFixed(4) : value;
      row.append(cell);
    });
    table.append(row);
  });
  if (table.children.length === 0) {
    const row = document.createElement("tr");
    row.innerHTML = '<td colspan="4" class="empty-state">Ablation metrics are unavailable.</td>';
    table.append(row);
  }
}

function renderHitRates(data) {
  const labels = ["visual_only", "audio_only", "speech_only", "full_fusion"];
  const displayNames = {
    visual_only: "visual-only",
    audio_only: "audio-only",
    speech_only: "speech-only",
    full_fusion: "fused"
  };
  const topKs = data.prediction_settings.top_k_values;
  const table = document.querySelector("#hit-rate-table");
  table.replaceChildren();

  topKs.forEach((topK) => {
    const row = document.createElement("tr");
    const budget = document.createElement("td");
    budget.textContent = topK;
    row.append(budget);
    labels.forEach((label) => {
      const cell = document.createElement("td");
      const value = data.variants[label].by_k[String(topK)].hit_rate;
      cell.textContent = value === null ? "—" : `${(value * 100).toFixed(1)}%`;
      row.append(cell);
    });
    table.append(row);
  });

  const firstK = String(topKs[0]);
  const lastK = String(topKs[topKs.length - 1]);
  const firstHits = labels.map((label) => data.variants[label].by_k[firstK].hit_rate);
  const firstF1 = labels.map((label) => data.variants[label].by_k[firstK].metrics.f1);
  const allFirstZero = firstHits.every((value) => value === 0) &&
    firstF1.every((value) => value === 0);
  const lastResults = labels.map((label) => ({
    label,
    hitRate: data.variants[label].by_k[lastK].hit_rate,
    f1: data.variants[label].by_k[lastK].metrics.f1
  }));
  const highestHit = [...lastResults].sort((a, b) => b.hitRate - a.hitRate)[0];
  const lowestHit = [...lastResults].sort((a, b) => a.hitRate - b.hitRate)[0];
  const highestF1 = [...lastResults].sort((a, b) => b.f1 - a.f1)[0];
  const sentences = [];
  if (allFirstZero) {
    sentences.push(`At k=${firstK}, all four variants have 0.0% hit-rate and F1 0.0000.`);
  }
  sentences.push(
    `At k=${lastK}, ${displayNames[highestHit.label]} has the highest hit-rate (${(highestHit.hitRate * 100).toFixed(1)}%) and ${displayNames[highestF1.label]} has the highest F1 (${highestF1.f1.toFixed(4)}); ${displayNames[lowestHit.label]} has the lowest hit-rate (${(lowestHit.hitRate * 100).toFixed(1)}%).`
  );
  sentences.push(
    `The best F1 at k=${lastK} remains low, and hit-rate does not increase monotonically for every variant as k grows.`
  );
  document.querySelector("#hit-rate-summary").textContent = sentences.join(" ");
}

async function loadJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load ${path} (${response.status}).`);
  }
  return response.json();
}

async function loadDemo() {
  try {
    renderHighlights(await loadJson("assets/cricket-full-match-demo.json"));
  } catch (error) {
    document.querySelector("#highlights-table").innerHTML =
      '<tr><td colspan="5" class="empty-state">Model output metadata could not be loaded.</td></tr>';
    document.querySelector("#data-source-note").textContent = error.message;
  }

  try {
    renderAblation(await loadJson("assets/ablation.json"));
  } catch {
    const row = document.createElement("tr");
    row.innerHTML = '<td colspan="4" class="empty-state">Run python run_ablation.py to generate this table.</td>';
    document.querySelector("#ablation-table").replaceChildren(row);
  }

  try {
    renderHitRates(await loadJson("assets/ablation_k.json"));
  } catch {
    const row = document.createElement("tr");
    row.innerHTML = '<td colspan="5" class="empty-state">Run python run_ablation_k.py to generate hit-rate results.</td>';
    document.querySelector("#hit-rate-table").replaceChildren(row);
  }
}

document.querySelector("#theme-toggle").addEventListener("click", (event) => {
  const root = document.documentElement;
  const isDark = root.dataset.theme !== "dark";
  root.dataset.theme = isDark ? "dark" : "light";
  event.currentTarget.setAttribute("aria-pressed", String(isDark));
  document.querySelector("#theme-label").textContent = isDark ? "Light mode" : "Dark mode";
});

document.querySelectorAll(".copy-button").forEach((button) => {
  button.addEventListener("click", async () => {
    const command = document.querySelector(`#${button.dataset.copy}`).innerText;
    try {
      await navigator.clipboard.writeText(command);
      button.textContent = "Copied";
    } catch (error) {
      button.textContent = "Copy unavailable";
      button.title = error.message;
    }
    window.setTimeout(() => {
      button.textContent = "Copy";
    }, 1400);
  });
});

loadDemo();
