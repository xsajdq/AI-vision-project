const API = {
  health: "/api/health",
  samples: "/api/samples",
  history: "/api/history",
  inspectImage: "/api/inspect/image",
  inspectAudio: "/api/inspect/audio",
};

function $(id) {
  return document.getElementById(id);
}

function drawGauge(canvas, score, threshold) {
  const ctx = canvas.getContext("2d");
  const cx = canvas.width / 2;
  const cy = canvas.height - 10;
  const radius = 85;
  const startAngle = Math.PI;
  const endAngle = 2 * Math.PI;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.lineWidth = 14;
  ctx.lineCap = "round";
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.beginPath();
  ctx.arc(cx, cy, radius, startAngle, endAngle);
  ctx.stroke();

  const maxScale = Math.max(threshold * 1.8, score * 1.1, 1e-6);
  const ratio = Math.min(score / maxScale, 1);
  const valueAngle = startAngle + ratio * Math.PI;
  const overThreshold = score > threshold;

  const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
  if (overThreshold) {
    gradient.addColorStop(0, "#ff9f5a");
    gradient.addColorStop(1, "#ff5470");
  } else {
    gradient.addColorStop(0, "#34d1c4");
    gradient.addColorStop(1, "#33d17e");
  }
  ctx.strokeStyle = gradient;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, startAngle, valueAngle);
  ctx.stroke();

  const threshRatio = Math.min(threshold / maxScale, 1);
  const threshAngle = startAngle + threshRatio * Math.PI;
  const mx1 = cx + (radius - 13) * Math.cos(threshAngle);
  const my1 = cy + (radius - 13) * Math.sin(threshAngle);
  const mx2 = cx + (radius + 13) * Math.cos(threshAngle);
  const my2 = cy + (radius + 13) * Math.sin(threshAngle);
  ctx.strokeStyle = "rgba(255,255,255,0.65)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(mx1, my1);
  ctx.lineTo(mx2, my2);
  ctx.stroke();

  ctx.fillStyle = "#e7eaf1";
  ctx.font = "600 15px -apple-system, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(score.toFixed(4), cx, cy - 6);
}

function formatTime(isoString) {
  const date = new Date(isoString);
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

async function refreshHealth() {
  const pill = $("status-pill");
  const text = $("status-text");
  try {
    const res = await fetch(API.health);
    const data = await res.json();
    if (data.vision_model_loaded && data.audio_model_loaded) {
      pill.className = "status status-ok";
      text.textContent = "Models ready";
    } else {
      pill.className = "status status-bad";
      const missing = [];
      if (!data.vision_model_loaded) missing.push("vision");
      if (!data.audio_model_loaded) missing.push("audio");
      text.textContent = `Missing model: ${missing.join(", ")} (run training scripts)`;
    }
  } catch (err) {
    pill.className = "status status-bad";
    text.textContent = "Backend unreachable";
  }
}

async function refreshHistory() {
  const body = $("history-body");
  try {
    const res = await fetch(API.history);
    const entries = await res.json();
    if (!entries.length) {
      body.innerHTML = '<tr class="empty-row"><td colspan="5">No inspections yet</td></tr>';
      return;
    }
    body.innerHTML = entries
      .map((entry) => {
        const bad = entry.verdict.includes("DEFECT") || entry.verdict.includes("ANOMALY");
        const icon = entry.kind === "vision" ? "🖼️" : "🎧";
        return `<tr>
          <td>${icon} ${entry.kind}</td>
          <td>${entry.filename}</td>
          <td><span class="badge ${bad ? "bad" : "ok"}">${entry.verdict}</span></td>
          <td>${entry.anomaly_score.toFixed(4)}</td>
          <td>${formatTime(entry.timestamp)}</td>
        </tr>`;
      })
      .join("");
  } catch (err) {
    // History is a nice-to-have; ignore transient failures silently.
  }
}

function showError(kind, message) {
  const box = $(`${kind}-error`);
  box.hidden = false;
  box.textContent = message;
}

function clearError(kind) {
  $(`${kind}-error`).hidden = true;
}

async function runInspection(kind, file) {
  const resultBox = $(`${kind}-result`);
  const loading = $(`${kind}-loading`);
  clearError(kind);
  resultBox.hidden = true;
  loading.hidden = false;

  const formData = new FormData();
  formData.append("file", file);
  const endpoint = kind === "vision" ? API.inspectImage : API.inspectAudio;

  try {
    const res = await fetch(endpoint, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `Request failed (${res.status})`);
    }

    const verdictBox = $(`${kind}-verdict`);
    const isBad = kind === "vision" ? data.is_defective : data.is_anomalous;
    verdictBox.textContent = data.verdict;
    verdictBox.className = `verdict ${isBad ? "bad" : "ok"}`;

    $(`${kind}-score`).textContent = data.anomaly_score.toFixed(5);
    $(`${kind}-threshold`).textContent = data.threshold.toFixed(5);

    const img = $(kind === "vision" ? "vision-heatmap" : "audio-spectrogram");
    img.src = `data:image/png;base64,${kind === "vision" ? data.heatmap_png_base64 : data.spectrogram_png_base64}`;

    drawGauge($(`${kind}-gauge`), data.anomaly_score, data.threshold);

    resultBox.hidden = false;
    refreshHistory();
  } catch (err) {
    showError(kind, err.message || "Something went wrong");
  } finally {
    loading.hidden = true;
  }
}

function setupDropzone(kind) {
  const zone = $(`${kind}-dropzone`);
  const input = $(`${kind}-input`);

  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") input.click();
  });

  input.addEventListener("change", () => {
    if (input.files.length) runInspection(kind, input.files[0]);
  });

  ["dragenter", "dragover"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.add("drag-over");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.remove("drag-over");
    })
  );
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) runInspection(kind, file);
  });
}

async function fetchFileAsBlob(url) {
  const res = await fetch(url);
  const blob = await res.blob();
  const name = url.split("/").pop();
  return new File([blob], name, { type: blob.type });
}

function sampleLabel(path, category) {
  const name = path.split("/").pop().replace(/\.(png|wav)$/i, "");
  return { name, category };
}

async function loadSampleButtons() {
  try {
    const res = await fetch(API.samples);
    const data = await res.json();

    const visionContainer = $("vision-sample-buttons");
    [...data.vision.normal, ...data.vision.defective].forEach((path) => {
      const isDefective = path.includes("/defective/");
      const btn = document.createElement("button");
      btn.className = `sample-btn${isDefective ? " defective" : ""}`;
      btn.textContent = path.split("/").pop();
      btn.addEventListener("click", async () => {
        const file = await fetchFileAsBlob(path);
        runInspection("vision", file);
      });
      visionContainer.appendChild(btn);
    });

    const audioContainer = $("audio-sample-buttons");
    [...data.audio.normal, ...data.audio.anomalous].forEach((path) => {
      const isAnomalous = path.includes("/anomalous/");
      const btn = document.createElement("button");
      btn.className = `sample-btn${isAnomalous ? " anomalous" : ""}`;
      btn.textContent = path.split("/").pop();
      btn.addEventListener("click", async () => {
        const file = await fetchFileAsBlob(path);
        runInspection("audio", file);
      });
      audioContainer.appendChild(btn);
    });
  } catch (err) {
    // No sample files generated yet - the drop zones still work standalone.
  }
}

setupDropzone("vision");
setupDropzone("audio");
loadSampleButtons();
refreshHealth();
refreshHistory();
