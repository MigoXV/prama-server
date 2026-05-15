const form = document.querySelector("#evaluationForm");
const submitButton = document.querySelector("#submitButton");
const jobBadge = document.querySelector("#jobBadge");
const jobId = document.querySelector("#jobId");
const progressBar = document.querySelector("#progressBar");
const processed = document.querySelector("#processed");
const evaluated = document.querySelector("#evaluated");
const runningWer = document.querySelector("#runningWer");
const runningCer = document.querySelector("#runningCer");
const finalRates = document.querySelector("#finalRates");
const currentId = document.querySelector("#currentId");
const reference = document.querySelector("#reference");
const hypothesis = document.querySelector("#hypothesis");
const errorBox = document.querySelector("#errorBox");
const resultRows = document.querySelector("#resultRows");
const reportSummary = document.querySelector("#reportSummary");
const reportUtterances = document.querySelector("#reportUtterances");

let eventSource = null;
let seenResultKeys = new Set();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resetView();
  setBusy(true);

  try {
    const response = await fetch("/api/evaluations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readForm()),
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }

    const created = await response.json();
    jobId.textContent = created.job_id;
    setBadge(created.status);
    connectEvents(created.job_id);
  } catch (error) {
    showError(error.message);
    setBusy(false);
  }
});

function readForm() {
  const data = new FormData(form);
  const limit = data.get("limit");
  const connectTimeout = data.get("connect_timeout_seconds");
  return {
    target: data.get("target"),
    dataset_path: data.get("dataset_path"),
    split: data.get("split"),
    limit: limit ? Number(limit) : null,
    language_code: data.get("language_code"),
    sample_rate: Number(data.get("sample_rate")),
    min_reference_words: Number(data.get("min_reference_words")),
    hotwords: String(data.get("hotwords") || "")
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean),
    connect_timeout_seconds: connectTimeout ? Number(connectTimeout) : null,
    request_timeout_seconds: Number(data.get("request_timeout_seconds")),
  };
}

function connectEvents(id) {
  if (eventSource) {
    eventSource.close();
  }

  eventSource = new EventSource(`/api/evaluations/${id}/events`);
  eventSource.addEventListener("inference_result", (event) => {
    applyProgress(JSON.parse(event.data));
  });
  eventSource.addEventListener("progress", (event) => {
    applyProgress(JSON.parse(event.data));
  });
  eventSource.addEventListener("error", (event) => {
    if (event.data) {
      showError(JSON.parse(event.data).message);
    }
    setBadge("failed");
    setBusy(false);
  });
  eventSource.addEventListener("done", (event) => {
    const snapshot = JSON.parse(event.data);
    setBadge(snapshot.status);
    if (snapshot.error) {
      showError(snapshot.error);
    }
    if (snapshot.result) {
      applyFinalResult(snapshot.result);
    }
    setBusy(false);
    eventSource.close();
  });
}

function applyProgress(progress) {
  setBadge(progress.status);
  const total = progress.total || 0;
  const done = progress.processed || 0;
  const percent = total > 0 ? Math.min((done / total) * 100, 100) : 0;

  progressBar.style.width = `${percent}%`;
  processed.textContent = `${done} / ${total}`;
  evaluated.textContent = progress.evaluated ?? 0;
  currentId.textContent = progress.current_id || "-";
  reference.textContent = progress.reference || "-";
  hypothesis.textContent = progress.hypothesis || "-";
  appendInferenceResult(progress);
  finalRates.textContent = progress.status === "completed" ? "已完成" : "推理中";

  if (progress.result) {
    runningWer.textContent = formatRate(progress.result.wer);
    runningCer.textContent = formatRate(progress.result.cer);
    finalRates.textContent = "已完成";
    renderWerReport(progress.result.wer_report);
  }
}

function appendInferenceResult(progress) {
  if (!progress.id && !progress.current_id) {
    return;
  }

  const index = progress.evaluated ?? resultRows.children.length + 1;
  const sampleId = progress.id || progress.current_id;
  const key = `${index}:${sampleId}`;
  if (seenResultKeys.has(key)) {
    return;
  }
  seenResultKeys.add(key);

  const row = document.createElement("tr");
  row.innerHTML = `
    <td>${escapeHtml(String(index))}</td>
    <td>${escapeHtml(sampleId || "-")}</td>
    <td>${escapeHtml(progress.reference || "-")}</td>
    <td>${escapeHtml(progress.hypothesis || "-")}</td>
  `;
  resultRows.appendChild(row);
}

function applyFinalResult(result) {
  runningWer.textContent = formatRate(result.wer);
  runningCer.textContent = formatRate(result.cer);
  finalRates.textContent = "已完成";
  renderWerReport(result.wer_report);
}

function renderWerReport(report) {
  if (!report) {
    reportSummary.replaceChildren();
    reportUtterances.replaceChildren();
    return;
  }

  const summary = report.summary || {};
  reportSummary.innerHTML = `
    <div><span>WER</span><strong>${formatRate(summary.wer)}</strong></div>
    <div><span>Accuracy</span><strong>${formatRate(summary.accuracy)}</strong></div>
    <div><span>Correct</span><strong>${formatNumber(summary.correct)}</strong></div>
    <div><span>Sub</span><strong>${formatNumber(summary.substitutions)}</strong></div>
    <div><span>Del</span><strong>${formatNumber(summary.deletions)}</strong></div>
    <div><span>Ins</span><strong>${formatNumber(summary.insertions)}</strong></div>
  `;

  reportUtterances.replaceChildren();
  for (const utterance of report.utterances || []) {
    reportUtterances.appendChild(createUtteranceReport(utterance));
  }
}

function createUtteranceReport(utterance) {
  const details = document.createElement("details");
  details.className = "utterance-report";

  const counts = countLabels(utterance.tokens || []);
  const summary = document.createElement("summary");
  summary.innerHTML = `
    <span>${escapeHtml(utterance.id || "-")}</span>
    <span>C ${counts.correct}</span>
    <span>S ${counts.substitution}</span>
    <span>D ${counts.deletion}</span>
    <span>I ${counts.insertion}</span>
  `;
  details.appendChild(summary);

  const tokens = document.createElement("div");
  tokens.className = "token-grid";
  for (const token of utterance.tokens || []) {
    const item = document.createElement("span");
    item.className = `token token-${token.label || "unknown"}`;
    item.title = `ref: ${token.ref || ""}\nhyp: ${token.hyp || ""}`;
    item.innerHTML = `
      <span>${escapeHtml(token.ref || "∅")}</span>
      <span>${escapeHtml(token.hyp || "∅")}</span>
    `;
    tokens.appendChild(item);
  }
  details.appendChild(tokens);
  return details;
}

function countLabels(tokens) {
  return tokens.reduce(
    (counts, token) => {
      const label = token.label || "unknown";
      counts[label] = (counts[label] || 0) + 1;
      return counts;
    },
    { correct: 0, substitution: 0, deletion: 0, insertion: 0 },
  );
}

function resetView() {
  if (eventSource) {
    eventSource.close();
  }
  errorBox.hidden = true;
  errorBox.textContent = "";
  jobId.textContent = "-";
  setBadge("queued");
  progressBar.style.width = "0%";
  processed.textContent = "0 / 0";
  evaluated.textContent = "0";
  runningWer.textContent = "-";
  runningCer.textContent = "-";
  finalRates.textContent = "-";
  currentId.textContent = "-";
  reference.textContent = "-";
  hypothesis.textContent = "-";
  resultRows.replaceChildren();
  seenResultKeys = new Set();
  reportSummary.replaceChildren();
  reportUtterances.replaceChildren();
}

function setBusy(isBusy) {
  submitButton.disabled = isBusy;
  submitButton.textContent = isBusy ? "评估中..." : "启动评估";
}

function setBadge(status) {
  const labelMap = {
    queued: "排队中",
    running: "运行中",
    started: "已开始",
    completed: "已完成",
    failed: "失败",
  };
  jobBadge.textContent = labelMap[status] || status || "未启动";
  jobBadge.dataset.status = status || "";
}

function showError(message) {
  errorBox.hidden = false;
  errorBox.textContent = message || "未知错误";
}

function formatRate(value) {
  if (value === null || value === undefined) {
    return "-";
  }
  return `${Number(value).toFixed(2)}%`;
}

function formatNumber(value) {
  if (value === null || value === undefined) {
    return "0";
  }
  return String(value);
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
