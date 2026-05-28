import {
  Activity,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  CircleDashed,
  Clipboard,
  Database,
  Download,
  EyeOff,
  FileText,
  Moon,
  Pause,
  Play,
  Server,
  Settings,
  Sun,
  TerminalSquare,
  TriangleAlert,
  Monitor,
  Radio,
} from "lucide-react";
import type { ChangeEvent, FormEvent, MouseEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { usePersistentState } from "./hooks/usePersistentState";
import { useThemeMode } from "./hooks/useThemeMode";
import {
  createEvaluation,
  recalculateEvaluationMetrics,
  subscribeEvaluationEvents,
} from "./services/evaluations";
import type {
  EvaluationFormState,
  EvaluationProgress,
  EvaluationRequest,
  EvaluationResult,
  EvaluationTask,
  InferenceRow,
  JobStatus,
  ThemeMode,
  VadReportSample,
  VadReportSegment,
  WerReport,
  WerSummary,
  WerToken,
  WerUtterance,
} from "./types";

const DEFAULT_FORM_STATE: EvaluationFormState = {
  task: "asr",
  target: "192.168.0.222:50011",
  dataset_path: "data-bin/audiofolder/asr-demo",
  split: "test",
  limit: "",
  language_code: "en-US",
  sample_rate: "16000",
  min_reference_words: "5",
  hotwords: "",
  hotword_bias: "0",
  connect_timeout_seconds: "10",
  request_timeout_seconds: "60",
  interim_results: true,
  remove_punctuation: false,
  mask_frame_seconds: "0.01",
  chunk_duration_seconds: "0.1",
  hit_threshold: "0.9",
  streaming: false,
};

const TASK_DEFAULTS: Record<EvaluationTask, Partial<EvaluationFormState>> = {
  asr: {
    target: "192.168.0.222:50011",
    dataset_path: "data-bin/audiofolder/asr-demo",
    min_reference_words: "5",
  },
  vad: {
    target: "192.168.0.222:50021",
    dataset_path: "data-bin/audiofolder/vad-demo",
    min_reference_words: "0",
  },
};

const STATUS_LABELS: Record<JobStatus | "idle" | "started", string> = {
  idle: "未启动",
  queued: "排队中",
  running: "运行中",
  started: "已开始",
  completed: "已完成",
  failed: "失败",
};

const MODULES = [
  { label: "在线评估", icon: Activity, active: true },
  { label: "任务队列", icon: TerminalSquare, active: false },
  { label: "数据集", icon: Database, active: false },
  { label: "报告", icon: FileText, active: false },
  { label: "设置", icon: Settings, active: false },
];

type AlignmentMetric = "wer" | "cer";
type ReportSortMode =
  | "index-asc"
  | "index-desc"
  | "wer-desc"
  | "wer-asc"
  | "cer-desc"
  | "cer-asc";

export default function App() {
  const { themeMode, setThemeMode } = useThemeMode();
  const [storedFormState, setFormState] = usePersistentState<EvaluationFormState>(
    "prama.evaluationForm",
    DEFAULT_FORM_STATE,
  );
  const formState = { ...DEFAULT_FORM_STATE, ...storedFormState };
  const [advancedOpen, setAdvancedOpen] = usePersistentState(
    "prama.advancedOpen",
    false,
  );
  const [status, setStatus] = useState<JobStatus | "idle" | "started">("idle");
  const [jobId, setJobId] = useState("");
  const [progress, setProgress] = useState<EvaluationProgress | null>(null);
  const [rows, setRows] = useState<InferenceRow[]>([]);
  const [finalResult, setFinalResult] = useState<EvaluationResult | null>(null);
  const [excludedSampleIds, setExcludedSampleIds] = useState<Set<string>>(
    () => new Set(),
  );
  const [errorMessage, setErrorMessage] = useState("");
  const [connectionWarning, setConnectionWarning] = useState("");
  const [busy, setBusy] = useState(false);
  const [recalculating, setRecalculating] = useState(false);
  const [activeTab, setActiveTab] = useState<"overview" | "results" | "report">(
    "overview",
  );
  const [activeAlignmentMetric, setActiveAlignmentMetric] =
    useState<AlignmentMetric>("wer");
  const [reportSort, setReportSort] = useState<ReportSortMode>("index-asc");
  const [wrapWerAlignment, setWrapWerAlignment] = useState(false);
  const [resultJumpIndex, setResultJumpIndex] = useState("");
  const [highlightedResultIndex, setHighlightedResultIndex] = useState<number | null>(
    null,
  );
  const resultRowRefs = useRef(new Map<number, HTMLTableRowElement>());
  const closeEventsRef = useRef<(() => void) | null>(null);
  const formRef = useRef<HTMLFormElement | null>(null);

  useEffect(() => {
    return () => closeEventsRef.current?.();
  }, []);

  useEffect(() => {
    function handleGlobalKeyDown(event: KeyboardEvent) {
      if (event.key !== "Enter" || event.repeat || busy) {
        return;
      }
      if (isEditableTarget(event.target)) {
        return;
      }
      event.preventDefault();
      formRef.current?.requestSubmit();
    }

    window.addEventListener("keydown", handleGlobalKeyDown);
    return () => window.removeEventListener("keydown", handleGlobalKeyDown);
  }, [busy]);

  const progressPercent = useMemo(() => {
    const total = progress?.total ?? 0;
    const processed = progress?.processed ?? 0;
    return total > 0 ? Math.min((processed / total) * 100, 100) : 0;
  }, [progress]);

  const werReport = finalResult?.wer_report;
  const cerReport = finalResult?.cer_report;
  const canExport = finalResult !== null;
  const isVad = formState.task === "vad";
  const canRecalculate = status === "completed" && finalResult !== null && !recalculating;

  function resetRunState() {
    closeEventsRef.current?.();
    closeEventsRef.current = null;
    setStatus("idle");
    setJobId("");
    setProgress(null);
    setRows([]);
    setFinalResult(null);
    setExcludedSampleIds(new Set());
    setErrorMessage("");
    setConnectionWarning("");
    setBusy(false);
    setRecalculating(false);
    setResultJumpIndex("");
    setHighlightedResultIndex(null);
    resultRowRefs.current.clear();
  }

  function handleTaskChange(task: EvaluationTask) {
    if (task === formState.task) {
      return;
    }
    resetRunState();
    setFormState((current) => ({
      ...current,
      ...TASK_DEFAULTS[task],
      task,
    }));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    closeEventsRef.current?.();
    closeEventsRef.current = null;
    setBusy(true);
    setErrorMessage("");
    setConnectionWarning("");
    setRows([]);
    setProgress(null);
    setFinalResult(null);
    setExcludedSampleIds(new Set());
    setStatus("queued");
    setJobId("");

    try {
      const request = buildRequest(formState);
      const created = await createEvaluation(request);
      setJobId(created.job_id);
      setStatus(created.status);
      closeEventsRef.current = subscribeEvaluationEvents(created.job_id, {
        onProgress: (nextProgress) => {
          setConnectionWarning("");
          setProgress(nextProgress);
          setStatus(nextProgress.status ?? "running");
          appendProgressRow(nextProgress);
          if (nextProgress.result) {
            setFinalResult(nextProgress.result);
          }
        },
        onPartialProgress: (nextProgress) => {
          setConnectionWarning("");
          setProgress(nextProgress);
          setStatus(nextProgress.status ?? "running");
        },
        onDone: (snapshot) => {
          setStatus(snapshot.status);
          setProgress(snapshot.progress);
          setFinalResult(snapshot.result);
          setExcludedSampleIds(
            new Set(snapshot.result?.excluded_sample_ids ?? []),
          );
          setErrorMessage(snapshot.error ?? "");
          setBusy(false);
          closeEventsRef.current = null;
          if (snapshot.progress) {
            appendProgressRow(snapshot.progress);
          }
        },
        onError: (message) => {
          setStatus("failed");
          setErrorMessage(message);
          setBusy(false);
        },
        onConnectionError: () => {
          setConnectionWarning("事件流连接暂时不可用");
        },
      });
    } catch (error) {
      setStatus("failed");
      setErrorMessage(error instanceof Error ? error.message : "评估任务创建失败");
      setBusy(false);
    }
  }

  function appendProgressRow(nextProgress: EvaluationProgress) {
    const sampleId = nextProgress.id || nextProgress.current_id;
    if (!sampleId) {
      return;
    }

    setRows((currentRows) => {
      const index = nextProgress.evaluated ?? currentRows.length + 1;
      if (currentRows.some((row) => row.index === index && row.sampleId === sampleId)) {
        return currentRows;
      }

      return [
        ...currentRows,
        {
          index,
          sampleId,
          reference: nextProgress.reference || "-",
          hypothesis: nextProgress.hypothesis || "-",
          audioUrl: nextProgress.audio_url,
          durationSeconds: nextProgress.duration_seconds,
        },
      ];
    });
  }

  function toggleExcludedSample(sampleId: string, excluded: boolean) {
    setExcludedSampleIds((current) => {
      const next = new Set(current);
      if (excluded) {
        next.add(sampleId);
      } else {
        next.delete(sampleId);
      }
      return next;
    });
  }

  async function handleRecalculateMetrics() {
    if (!jobId || !canRecalculate) {
      return;
    }
    setRecalculating(true);
    setErrorMessage("");
    try {
      const nextResult = await recalculateEvaluationMetrics(
        jobId,
        [...excludedSampleIds],
      );
      setFinalResult(nextResult);
      setExcludedSampleIds(new Set(nextResult.excluded_sample_ids ?? []));
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "重新计算评估指标失败",
      );
    } finally {
      setRecalculating(false);
    }
  }

  function updateField(field: keyof EvaluationFormState, value: string) {
    setFormState((current) => ({ ...current, [field]: value }));
  }

  function resetForm() {
    setFormState(DEFAULT_FORM_STATE);
  }

  async function copyText(value: string) {
    if (!value) {
      return;
    }
    await navigator.clipboard.writeText(value);
  }

  function downloadResult() {
    if (!finalResult) {
      return;
    }
    const blob = new Blob([JSON.stringify(finalResult, null, 2)], {
      type: "application/json;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${jobId || "prama-evaluation"}-result.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  function jumpToResultIndex() {
    const index = Number(resultJumpIndex);
    if (!Number.isInteger(index) || index < 1) {
      return;
    }
    const row = resultRowRefs.current.get(index);
    if (!row) {
      return;
    }
    row.scrollIntoView({ block: "center", behavior: "smooth" });
    setHighlightedResultIndex(index);
  }

  return (
    <div className="console-frame">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-symbol">
            <Server size={19} />
          </div>
          <div>
            <strong>Prama</strong>
            <span>评估控制台</span>
          </div>
        </div>

        <nav className="module-nav" aria-label="主导航">
          {MODULES.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.label}
                type="button"
                className={`module-item ${item.active ? "active" : ""}`}
                disabled={!item.active}
              >
                <Icon size={17} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <span className="sidebar-label">主题</span>
          <div className="theme-switcher" aria-label="主题切换">
            <ThemeButton
              label="系统"
              active={themeMode === "system"}
              mode="system"
              onClick={setThemeMode}
            />
            <ThemeButton
              label="白色"
              active={themeMode === "light"}
              mode="light"
              onClick={setThemeMode}
            />
            <ThemeButton
              label="黑色"
              active={themeMode === "dark"}
              mode="dark"
              onClick={setThemeMode}
            />
          </div>
        </div>
      </aside>

      <main className="workspace">
        <header className="workspace-header">
          <div>
            <p className="eyebrow">{isVad ? "VAD Evaluation" : "ASR Evaluation"}</p>
            <h1>{isVad ? "语音活动检测评估" : "语音识别评估"}</h1>
          </div>
          <div className="header-actions">
            <StatusPill status={status} />
            <button
              type="button"
              className="icon-action"
              title="复制任务 ID"
              aria-label="复制任务 ID"
              disabled={!jobId}
              onClick={() => void copyText(jobId)}
            >
              <Clipboard size={16} />
            </button>
            <button
              type="button"
              className="icon-action"
              title="下载结果 JSON"
              aria-label="下载结果 JSON"
              disabled={!canExport}
              onClick={downloadResult}
            >
              <Download size={16} />
            </button>
          </div>
        </header>

        <div className="page-tabs" role="tablist" aria-label="评估视图">
          <TabButton
            active={activeTab === "overview"}
            label="运行概览"
            meta={STATUS_LABELS[status]}
            onClick={() => setActiveTab("overview")}
          />
          <TabButton
            active={activeTab === "report"}
            label={isVad ? "VAD 指标" : "对齐报告"}
            meta={
              isVad
                ? `${formatNumber(finalResult?.sample_count)} 个样本`
                : `${werReport?.utterances.length ?? cerReport?.utterances.length ?? 0} 个样本`
            }
            onClick={() => setActiveTab("report")}
          />
          <TabButton
            active={activeTab === "results"}
            label="推理结果"
            meta={`${rows.length} 条`}
            onClick={() => setActiveTab("results")}
          />
        </div>

        <section className="work-grid">
          <form ref={formRef} className="panel evaluation-form" onSubmit={handleSubmit}>
            <div className="panel-heading">
              <div>
                <h2>评估参数</h2>
                <span>Job {jobId || "-"}</span>
              </div>
              <button type="button" className="ghost-button" onClick={resetForm}>
                重置
              </button>
            </div>

            <div className="field-grid">
              <TaskSelector
                value={formState.task}
                onChange={handleTaskChange}
              />
              <TextField
                label={`${isVad ? "VAD" : "ASR"} gRPC 地址`}
                value={formState.target}
                onChange={(value) => updateField("target", value)}
                required
              />
              <TextField
                label="数据集路径"
                value={formState.dataset_path}
                onChange={(value) => updateField("dataset_path", value)}
                required
              />
              <TextField
                label="Split"
                value={formState.split}
                onChange={(value) => updateField("split", value)}
                required
              />
              <TextField
                label="Limit"
                value={formState.limit}
                type="number"
                min="1"
                placeholder="不限制"
                onChange={(value) => updateField("limit", value)}
              />
            </div>

            <button
              type="button"
              className={`advanced-toggle ${advancedOpen ? "open" : ""}`}
              onClick={() => setAdvancedOpen((value) => !value)}
            >
              <span>高级参数</span>
              <ChevronDown size={16} />
            </button>

            {advancedOpen ? (
              <div className="field-grid advanced-grid">
                {!isVad ? (
                  <>
                    <TextField
                      label="语言"
                      value={formState.language_code}
                      onChange={(value) => updateField("language_code", value)}
                      required
                    />
                    <TextField
                      label="最少参考词数"
                      value={formState.min_reference_words}
                      type="number"
                      min="0"
                      onChange={(value) => updateField("min_reference_words", value)}
                      required
                    />
                    <TextField
                      label="热词"
                      value={formState.hotwords}
                      onChange={(value) => updateField("hotwords", value)}
                    />
                    <TextField
                      label="热词 Bias"
                      value={formState.hotword_bias}
                      type="number"
                      step="0.1"
                      onChange={(value) => updateField("hotword_bias", value)}
                    />
                  </>
                ) : null}
                <TextField
                  label="采样率"
                  value={formState.sample_rate}
                  type="number"
                  min="1"
                  onChange={(value) => updateField("sample_rate", value)}
                  required
                />
                {isVad ? (
                  <>
                    <TextField
                      label="VAD 帧长秒"
                      value={formState.mask_frame_seconds}
                      type="number"
                      min="0.001"
                      step="0.001"
                      onChange={(value) => updateField("mask_frame_seconds", value)}
                    />
                    <TextField
                      label="VAD 分块秒"
                      value={formState.chunk_duration_seconds}
                      type="number"
                      min="0.01"
                      step="0.01"
                      onChange={(value) => updateField("chunk_duration_seconds", value)}
                    />
                    <TextField
                      label="段命中阈值"
                      value={formState.hit_threshold}
                      type="number"
                      min="0"
                      max="1"
                      step="0.01"
                      onChange={(value) => updateField("hit_threshold", value)}
                    />
                  </>
                ) : null}
                <TextField
                  label="连接超时秒"
                  value={formState.connect_timeout_seconds}
                  type="number"
                  min="0.1"
                  step="0.1"
                  onChange={(value) => updateField("connect_timeout_seconds", value)}
                />
                <TextField
                  label="请求超时秒"
                  value={formState.request_timeout_seconds}
                  type="number"
                  min="0.1"
                  step="0.1"
                  onChange={(value) => updateField("request_timeout_seconds", value)}
                  required
                />
                {!isVad ? (
                  <>
                    <label className="check-field">
                      <input
                        type="checkbox"
                        checked={formState.interim_results}
                        onChange={(event) =>
                          setFormState((current) => ({
                            ...current,
                            interim_results: event.target.checked,
                          }))
                        }
                      />
                      <span>启用临时识别结果</span>
                    </label>
                    <label className="check-field">
                      <input
                        type="checkbox"
                        checked={formState.remove_punctuation}
                        onChange={(event) =>
                          setFormState((current) => ({
                            ...current,
                            remove_punctuation: event.target.checked,
                          }))
                        }
                      />
                      <span>评估时去掉标点</span>
                    </label>
                  </>
                ) : (
                  <label className="check-field">
                    <input
                      type="checkbox"
                      checked={formState.streaming}
                      onChange={(event) =>
                        setFormState((current) => ({
                          ...current,
                          streaming: event.target.checked,
                        }))
                      }
                    />
                    <span>使用 VAD 流式接口</span>
                  </label>
                )}
              </div>
            ) : null}

            <button type="submit" className="primary-action" disabled={busy}>
              {busy ? "评估中..." : "启动评估"}
            </button>
          </form>

          <section className="run-column">
            {activeTab === "overview" ? (
              <>
                <div className="panel progress-panel">
                  <div className="panel-heading">
                    <div>
                      <h2>任务进度</h2>
                      <span>{jobId || "等待启动"}</span>
                    </div>
                    <strong className="progress-percent">{progressPercent.toFixed(0)}%</strong>
                  </div>
                  <div className="progress-track" aria-label="评估进度">
                    <div className="progress-bar" style={{ width: `${progressPercent}%` }} />
                  </div>
                  <div className="metric-strip progress-metrics">
                    <Metric label="已处理" value={`${progress?.processed ?? 0} / ${progress?.total ?? 0}`} />
                    <Metric label="已评估" value={String(progress?.evaluated ?? rows.length)} />
                  </div>
                  {connectionWarning ? (
                    <div className="inline-warning">
                      <TriangleAlert size={15} />
                      {connectionWarning}
                    </div>
                  ) : null}
                  {errorMessage ? (
                    <div className="error-box">
                      <TriangleAlert size={16} />
                      <span>{errorMessage}</span>
                    </div>
                  ) : null}
                </div>
                {isVad ? <VadOverviewMetrics result={finalResult} /> : null}

                {!isVad ? (
                  <div className="panel sample-panel">
                    <div className="panel-heading compact-heading">
                      <div>
                        <h2>当前样本</h2>
                        <span>{progress?.current_id || progress?.id || "-"}</span>
                      </div>
                    </div>
                    <div className="sample-grid">
                      <TextBlock label="Reference" value={progress?.reference || "-"} />
                      <TextBlock label="Hypothesis" value={progress?.hypothesis || "-"} />
                    </div>
                  </div>
                ) : null}
              </>
            ) : null}

            {activeTab === "results" ? (
              <div className="panel result-panel">
                <div className="panel-heading compact-heading">
                  <div>
                    <h2>推理结果</h2>
                    <span>{rows.length} 条</span>
                  </div>
                  <div className="jump-control">
                    <label>
                      <span>跳转序号</span>
                      <input
                        type="number"
                        min="1"
                        value={resultJumpIndex}
                        placeholder="例如 12"
                        onChange={(event) => setResultJumpIndex(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") {
                            jumpToResultIndex();
                          }
                        }}
                      />
                    </label>
                    <button type="button" onClick={jumpToResultIndex}>
                      跳转
                    </button>
                  </div>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>屏蔽</th>
                        <th>序号</th>
                        <th>样本</th>
                        <th>{isVad ? "Reference Segments" : "Reference"}</th>
                        <th>{isVad ? "Prediction Segments" : "Hypothesis"}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.length ? (
                        rows.map((row) => (
                          <tr
                            key={`${row.index}:${row.sampleId}`}
                            className={
                              highlightedResultIndex === row.index
                                ? "highlighted-row"
                                : ""
                            }
                            ref={(element) => {
                              if (element) {
                                resultRowRefs.current.set(row.index, element);
                              } else {
                                resultRowRefs.current.delete(row.index);
                              }
                            }}
                          >
                            <td>
                              <ExcludeCheckbox
                                checked={excludedSampleIds.has(row.sampleId)}
                                onChange={(checked) =>
                                  toggleExcludedSample(row.sampleId, checked)
                                }
                              />
                            </td>
                            <td>{row.index}</td>
                            <td>{row.sampleId}</td>
                            <td>{row.reference}</td>
                            <td>{row.hypothesis}</td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={5} className="empty-cell">
                            暂无推理结果
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}

            {activeTab === "report" ? (
              isVad ? (
                <VadReportPanel
                  result={finalResult}
                  excludedSampleIds={excludedSampleIds}
                  onExcludedChange={toggleExcludedSample}
                  canRecalculate={canRecalculate}
                  recalculating={recalculating}
                  onRecalculate={() => void handleRecalculateMetrics()}
                />
              ) : (
                <AsrAlignmentReportPanel
                  werReport={werReport}
                  cerReport={cerReport}
                  activeMetric={activeAlignmentMetric}
                  onActiveMetricChange={setActiveAlignmentMetric}
                  result={finalResult}
                  sortMode={reportSort}
                  onSortModeChange={setReportSort}
                  wrapAlignment={wrapWerAlignment}
                  onWrapAlignmentChange={setWrapWerAlignment}
                  excludedSampleIds={excludedSampleIds}
                  onExcludedChange={toggleExcludedSample}
                  canRecalculate={canRecalculate}
                  recalculating={recalculating}
                  onRecalculate={() => void handleRecalculateMetrics()}
                />
              )
            ) : null}
          </section>
        </section>
      </main>
    </div>
  );
}

function TabButton({
  active,
  label,
  meta,
  onClick,
}: {
  active: boolean;
  label: string;
  meta: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={`page-tab ${active ? "active" : ""}`}
      role="tab"
      aria-selected={active}
      onClick={onClick}
    >
      <span>{label}</span>
      <small>{meta}</small>
    </button>
  );
}

function ThemeButton({
  label,
  mode,
  active,
  onClick,
}: {
  label: string;
  mode: ThemeMode;
  active: boolean;
  onClick: (mode: ThemeMode) => void;
}) {
  const Icon = mode === "system" ? Monitor : mode === "light" ? Sun : Moon;
  return (
    <button
      type="button"
      className={`theme-button ${active ? "active" : ""}`}
      title={label}
      aria-label={label}
      onClick={() => onClick(mode)}
    >
      <Icon size={15} />
      <span>{label}</span>
    </button>
  );
}

function StatusPill({ status }: { status: JobStatus | "idle" | "started" }) {
  const Icon =
    status === "completed"
      ? CheckCircle2
      : status === "failed"
        ? TriangleAlert
        : status === "running" || status === "started"
          ? CircleDashed
          : BarChart3;
  return (
    <div className={`status-pill status-${status}`}>
      <Icon size={15} />
      {STATUS_LABELS[status]}
    </div>
  );
}

function TextField({
  label,
  value,
  onChange,
  type = "text",
  required = false,
  placeholder,
  min,
  step,
  max,
  disabled = false,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  type?: string;
  required?: boolean;
  placeholder?: string;
  min?: string;
  step?: string;
  max?: string;
  disabled?: boolean;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <input
        value={value}
        type={type}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        required={required}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

function TaskSelector({
  value,
  onChange,
}: {
  value: EvaluationTask;
  onChange: (task: EvaluationTask) => void;
}) {
  return (
    <div className="task-selector" role="radiogroup" aria-label="评估类型">
      <button
        type="button"
        className={value === "asr" ? "active" : ""}
        onClick={() => onChange("asr")}
      >
        <Activity size={16} />
        <span>ASR</span>
      </button>
      <button
        type="button"
        className={value === "vad" ? "active" : ""}
        onClick={() => onChange("vad")}
      >
        <Radio size={16} />
        <span>VAD</span>
      </button>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function TextBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-block">
      <span>{label}</span>
      <p>{value}</p>
    </div>
  );
}

function ExcludeCheckbox({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <button
      type="button"
      className={`exclude-toggle ${checked ? "excluded" : ""}`}
      title={checked ? "取消屏蔽该样本" : "屏蔽该样本的推理结果"}
      aria-label={checked ? "取消屏蔽该样本" : "屏蔽该样本的推理结果"}
      aria-pressed={checked}
      onClick={(event) => {
        event.preventDefault();
        event.stopPropagation();
        onChange(!checked);
      }}
    >
      <EyeOff size={15} />
    </button>
  );
}

function AudioPlayer({
  src,
  durationSeconds,
}: {
  src?: string;
  durationSeconds?: number;
}) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [actualDuration, setActualDuration] = useState(durationSeconds ?? 0);
  const duration = actualDuration || durationSeconds || 0;
  const progress = duration > 0 ? Math.min((currentTime / duration) * 100, 100) : 0;

  if (!src) {
    return <span className="audio-empty">无音频</span>;
  }

  function togglePlay(event: MouseEvent<HTMLButtonElement>) {
    event.preventDefault();
    event.stopPropagation();
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    if (audio.paused) {
      void audio.play();
    } else {
      audio.pause();
    }
  }

  function handleSeek(event: ChangeEvent<HTMLInputElement>) {
    event.stopPropagation();
    const audio = audioRef.current;
    if (!audio || duration <= 0) {
      return;
    }
    const nextTime = (Number(event.target.value) / 100) * duration;
    audio.currentTime = nextTime;
    setCurrentTime(nextTime);
  }

  return (
    <div className="audio-player" onClick={(event) => event.stopPropagation()}>
      <audio
        ref={audioRef}
        preload="metadata"
        src={src}
        onLoadedMetadata={(event) => {
          const nextDuration = event.currentTarget.duration;
          if (Number.isFinite(nextDuration)) {
            setActualDuration(nextDuration);
          }
        }}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
      />
      <button
        type="button"
        className="audio-play-button"
        title={playing ? "暂停音频" : "播放音频"}
        aria-label={playing ? "暂停音频" : "播放音频"}
        onClick={togglePlay}
      >
        {playing ? <Pause size={14} /> : <Play size={14} />}
      </button>
      <input
        className="audio-progress"
        type="range"
        min="0"
        max="100"
        step="0.1"
        value={progress}
        aria-label="音频播放进度"
        onClick={(event) => event.stopPropagation()}
        onChange={handleSeek}
      />
      <small>
        {formatSeconds(currentTime)} / {formatSeconds(duration)}
      </small>
    </div>
  );
}

function SampleCountStrip({
  result,
  fallbackCount,
}: {
  result: EvaluationResult | null;
  fallbackCount: number;
}) {
  const included = result?.included_sample_count ?? fallbackCount;
  const excluded = result?.excluded_sample_count ?? 0;
  return (
    <div className="metric-strip sample-count-strip">
      <Metric label="参与样本" value={formatNumber(included)} />
      <Metric label="屏蔽样本" value={formatNumber(excluded)} />
    </div>
  );
}

function VadOverviewMetrics({ result }: { result: EvaluationResult | null }) {
  if (!result) {
    return null;
  }

  return <VadMetricGroups metrics={result} />;
}

function VadMetricGroups({ metrics }: { metrics: EvaluationResult }) {
  const frame = metrics.frame;
  const segment = metrics.segment;

  return (
    <div className="vad-metric-groups">
      <section className="vad-metric-section">
        <div className="vad-metric-title">
          <span>帧级指标</span>
          <strong>{formatRate(frame?.frame_f1 ?? metrics.frame_f1)}</strong>
        </div>
        <div className="report-summary vad-summary vad-frame-summary">
          <Metric
            label="Accuracy"
            value={formatRate(frame?.frame_accuracy ?? metrics.frame_accuracy)}
          />
          <Metric
            label="Recall"
            value={formatRate(frame?.frame_recall ?? metrics.frame_recall)}
          />
          <Metric
            label="Precision"
            value={formatRate(frame?.frame_precision ?? metrics.frame_precision)}
          />
          <Metric label="F1" value={formatRate(frame?.frame_f1 ?? metrics.frame_f1)} />
        </div>
      </section>

      <section className="vad-metric-section">
        <div className="vad-metric-title">
          <span>段级指标</span>
          <strong>{formatRate(segment?.segment_f1)}</strong>
        </div>
        <div className="report-summary vad-summary">
          <Metric
            label="Recall"
            value={formatRate(segment?.segment_recall ?? metrics.segment_recall)}
          />
          <Metric
            label="Precision"
            value={formatRate(segment?.segment_precision ?? metrics.segment_precision)}
          />
          <Metric label="F1" value={formatRate(segment?.segment_f1)} />
        </div>
        <div className="metric-strip vad-segment-counts">
          <Metric
            label="Reference Segments"
            value={formatNumber(
              segment?.reference_segment_count ?? metrics.reference_segment_count,
            )}
          />
          <Metric
            label="Prediction Segments"
            value={formatNumber(
              segment?.prediction_segment_count ?? metrics.prediction_segment_count,
            )}
          />
        </div>
      </section>
    </div>
  );
}

function AsrAlignmentReportPanel({
  werReport,
  cerReport,
  activeMetric,
  onActiveMetricChange,
  result,
  sortMode,
  onSortModeChange,
  wrapAlignment,
  onWrapAlignmentChange,
  excludedSampleIds,
  onExcludedChange,
  canRecalculate,
  recalculating,
  onRecalculate,
}: {
  werReport: WerReport | undefined;
  cerReport: WerReport | undefined;
  activeMetric: AlignmentMetric;
  onActiveMetricChange: (metric: AlignmentMetric) => void;
  result: EvaluationResult | null;
  sortMode: ReportSortMode;
  onSortModeChange: (sortMode: ReportSortMode) => void;
  wrapAlignment: boolean;
  onWrapAlignmentChange: (wrapAlignment: boolean) => void;
  excludedSampleIds: Set<string>;
  onExcludedChange: (sampleId: string, excluded: boolean) => void;
  canRecalculate: boolean;
  recalculating: boolean;
  onRecalculate: () => void;
}) {
  const activeReport = activeMetric === "wer" ? werReport : cerReport;
  const activeLabel: "WER" | "CER" = activeMetric === "wer" ? "WER" : "CER";
  const sampleCount = werReport?.utterances.length ?? cerReport?.utterances.length ?? 0;
  const utterances = useMemo(
    () =>
      sortAlignmentUtterances(activeReport?.utterances ?? [], sortMode, {
        wer: werReport,
        cer: cerReport,
      }),
    [activeReport?.utterances, cerReport, sortMode, werReport],
  );
  return (
    <div className="panel report-panel">
      <div className="panel-heading compact-heading">
        <div>
          <h2>对齐报告</h2>
          <span>{sampleCount} 个样本</span>
        </div>
        <div className="report-controls">
          <button
            type="button"
            className="ghost-button"
            disabled={!canRecalculate}
            onClick={onRecalculate}
          >
            {recalculating ? "重算中..." : "重新计算评估指标"}
          </button>
          <label className="wrap-control">
            <input
              type="checkbox"
              checked={wrapAlignment}
              onChange={(event) => onWrapAlignmentChange(event.target.checked)}
            />
            <span>自动换行</span>
          </label>
          <label className="sort-control">
            <span>排序</span>
            <select
              value={sortMode}
              onChange={(event) =>
                onSortModeChange(event.target.value as ReportSortMode)
              }
            >
              <option value="index-asc">索引升序</option>
              <option value="index-desc">索引降序</option>
              <option value="wer-desc">WER 降序</option>
              <option value="wer-asc">WER 升序</option>
              <option value="cer-desc">CER 降序</option>
              <option value="cer-asc">CER 升序</option>
            </select>
          </label>
        </div>
      </div>

      {werReport?.summary || cerReport?.summary ? (
        <>
          <div className="asr-summary-grid">
            <AsrMetricSummaryCard
              label="WER"
              summary={werReport?.summary}
              fallbackRate={result?.wer}
            />
            <AsrMetricSummaryCard
              label="CER"
              summary={cerReport?.summary}
              fallbackRate={result?.cer}
            />
          </div>
          <SampleCountStrip result={result} fallbackCount={sampleCount} />
          <div className="alignment-metric-tabs" role="tablist" aria-label="对齐指标">
            <button
              type="button"
              className={activeMetric === "wer" ? "active" : ""}
              aria-selected={activeMetric === "wer"}
              role="tab"
              onClick={() => onActiveMetricChange("wer")}
            >
              WER
            </button>
            <button
              type="button"
              className={activeMetric === "cer" ? "active" : ""}
              aria-selected={activeMetric === "cer"}
              role="tab"
              onClick={() => onActiveMetricChange("cer")}
            >
              CER
            </button>
          </div>
          <div className="utterance-list">
            {utterances.map((utterance) => (
              <details className="utterance" key={utterance.id}>
                <summary>
                  <span className="utterance-title">
                    <ExcludeCheckbox
                      checked={excludedSampleIds.has(utterance.id)}
                      onChange={(checked) => onExcludedChange(utterance.id, checked)}
                    />
                    <strong>#{utterance.index ?? "-"}</strong>
                    <span>{utterance.id || "-"}</span>
                  </span>
                  <TokenCounts
                    metricLabel={activeLabel}
                    summary={utterance.summary}
                    tokens={utterance.tokens}
                  />
                </summary>
                <div className="utterance-audio-row">
                  <AudioPlayer
                    src={utterance.audio_url}
                    durationSeconds={utterance.duration_seconds}
                  />
                </div>
                <WerAlignmentRows tokens={utterance.tokens} wrap={wrapAlignment} />
              </details>
            ))}
          </div>
        </>
      ) : (
        <div className="empty-state">评估完成后生成对齐报告</div>
      )}
    </div>
  );
}

function VadReportPanel({
  result,
  excludedSampleIds,
  onExcludedChange,
  canRecalculate,
  recalculating,
  onRecalculate,
}: {
  result: EvaluationResult | null;
  excludedSampleIds: Set<string>;
  onExcludedChange: (sampleId: string, excluded: boolean) => void;
  canRecalculate: boolean;
  recalculating: boolean;
  onRecalculate: () => void;
}) {
  const samples = result?.vad_report?.samples ?? [];
  return (
    <div className="panel report-panel">
      <div className="panel-heading compact-heading">
        <div>
          <h2>VAD 报告</h2>
          <span>{formatNumber(result?.sample_count)} 个样本</span>
        </div>
        <div className="report-controls vad-report-controls">
          <button
            type="button"
            className="ghost-button"
            disabled={!canRecalculate}
            onClick={onRecalculate}
          >
            {recalculating ? "重算中..." : "重新计算评估指标"}
          </button>
          <div className="vad-legend report-legend">
            <LegendItem className="hit" label="命中" />
            <LegendItem className="miss" label="漏检" />
            <LegendItem className="false_alarm" label="虚警" />
            <LegendItem className="correct_reject" label="静音正确" />
          </div>
        </div>
      </div>
      {result ? (
        <>
          <SampleCountStrip result={result} fallbackCount={samples.length} />
          <VadMaskReport
            samples={samples}
            excludedSampleIds={excludedSampleIds}
            onExcludedChange={onExcludedChange}
          />
        </>
      ) : (
        <div className="empty-state">评估完成后生成 VAD 指标</div>
      )}
    </div>
  );
}

function AsrMetricSummaryCard({
  label,
  summary,
  fallbackRate,
}: {
  label: "WER" | "CER";
  summary?: WerSummary;
  fallbackRate?: number;
}) {
  return (
    <section className="asr-summary-card">
      <div className="asr-summary-title">
        <span>{label}</span>
        <strong>{formatRate(summary?.wer ?? fallbackRate)}</strong>
      </div>
      <div className="report-summary">
        <Metric label="Correct" value={formatNumber(summary?.correct)} />
        <Metric label="Sub" value={formatNumber(summary?.substitutions)} />
        <Metric label="Del" value={formatNumber(summary?.deletions)} />
        <Metric label="Ins" value={formatNumber(summary?.insertions)} />
      </div>
    </section>
  );
}

function VadMaskReport({
  samples,
  excludedSampleIds,
  onExcludedChange,
}: {
  samples: VadReportSample[];
  excludedSampleIds: Set<string>;
  onExcludedChange: (sampleId: string, excluded: boolean) => void;
}) {
  if (!samples.length) {
    return <div className="empty-state">评估完成后生成 mask 对齐报告</div>;
  }

  return (
    <div className="vad-report-list">
      {samples.map((sample) => (
        <details className="vad-sample" key={sample.id} open={samples.length === 1}>
          <summary>
            <span>
              <ExcludeCheckbox
                checked={excludedSampleIds.has(sample.id)}
                onChange={(checked) => onExcludedChange(sample.id, checked)}
              />
              #{sample.index ?? "-"} {sample.id}
            </span>
            <AudioPlayer
              src={sample.audio_url}
              durationSeconds={sample.duration_seconds}
            />
            <small>{formatSeconds(sample.duration_seconds)}</small>
          </summary>
          {sample.metrics ? (
            <div className="sample-vad-metrics">
              <VadMetricGroups metrics={sample.metrics} />
            </div>
          ) : null}
          <div className="mask-stack">
            <MaskTrack
              label="Reference"
              duration={sample.duration_seconds}
              segments={sample.reference_segments}
            />
            <MaskTrack
              label="Prediction"
              duration={sample.duration_seconds}
              segments={sample.prediction_segments}
            />
          </div>
          <div className="region-track" aria-label="VAD 对齐区域">
            {sample.regions.map((region) => (
              <span
                className={`region region-${region.label}`}
                key={`${sample.id}:${region.start_frame}:${region.end_frame}:${region.label}`}
                style={{
                  left: `${toPercent(region.start, sample.duration_seconds)}%`,
                  width: `${toPercent(region.duration, sample.duration_seconds)}%`,
                }}
                title={`${regionLabel(region.label)} ${formatSeconds(region.start)} - ${formatSeconds(region.end)}`}
              />
            ))}
          </div>
        </details>
      ))}
    </div>
  );
}

function MaskTrack({
  label,
  duration,
  segments,
}: {
  label: string;
  duration: number;
  segments: VadReportSegment[];
}) {
  return (
    <div className="mask-row">
      <span>{label}</span>
      <div className="mask-track">
        {segments.map((segment) => (
          <span
            className={`mask-segment mask-${segment.status}`}
            key={`${label}:${segment.start_frame}:${segment.end_frame}:${segment.status}`}
            style={{
              left: `${toPercent(segment.start, duration)}%`,
              width: `${toPercent(segment.duration, duration)}%`,
            }}
            title={`${segmentStatusLabel(segment.status)} ${formatSeconds(segment.start)} - ${formatSeconds(segment.end)}`}
          />
        ))}
      </div>
    </div>
  );
}

function LegendItem({ className, label }: { className: string; label: string }) {
  return (
    <span>
      <i className={`legend-dot ${className}`} />
      {label}
    </span>
  );
}

function TokenCounts({
  metricLabel,
  summary,
  tokens,
}: {
  metricLabel: "WER" | "CER";
  summary?: WerSummary;
  tokens: WerToken[];
}) {
  const counts = tokens.reduce(
    (current, token) => {
      current[token.label] = (current[token.label] ?? 0) + 1;
      return current;
    },
    { correct: 0, substitution: 0, deletion: 0, insertion: 0 } as Record<string, number>,
  );

  return (
    <span className="token-counts">
      {summary ? `${metricLabel} ${formatRate(summary.wer)} · ` : ""}
      C {summary?.correct ?? counts.correct ?? 0} · S{" "}
      {summary?.substitutions ?? counts.substitution ?? 0} · D{" "}
      {summary?.deletions ?? counts.deletion ?? 0} · I{" "}
      {summary?.insertions ?? counts.insertion ?? 0}
    </span>
  );
}

function WerAlignmentRows({ tokens, wrap }: { tokens: WerToken[]; wrap: boolean }) {
  const gridStyle = {
    gridTemplateColumns: `42px repeat(${Math.max(tokens.length, 1)}, max-content)`,
  };

  if (wrap) {
    return (
      <div className="wer-alignment wrap">
        <div className="wer-wrap-stack">
          {chunkWerTokens(tokens).map((chunk, chunkIndex) => (
            <div
              className="wer-alignment-grid"
              style={{
                gridTemplateColumns: `42px repeat(${Math.max(
                  chunk.length,
                  1,
                )}, max-content)`,
              }}
              key={`chunk:${chunkIndex}`}
            >
              <span className="wer-row-label">REF</span>
              {chunk.map((token, index) => (
                <span
                  className={`wer-word ${getWerWordClass(token.label, "ref")}`}
                  title={`ref: ${token.ref || "*"}\nhyp: ${token.hyp || "*"}`}
                  key={`ref:${chunkIndex}:${index}:${token.ref ?? ""}:${token.hyp ?? ""}`}
                >
                  {token.ref || "*"}
                </span>
              ))}
              <span className="wer-row-label">HYP</span>
              {chunk.map((token, index) => (
                <span
                  className={`wer-word ${getWerWordClass(token.label, "hyp")}`}
                  title={`ref: ${token.ref || "*"}\nhyp: ${token.hyp || "*"}`}
                  key={`hyp:${chunkIndex}:${index}:${token.ref ?? ""}:${token.hyp ?? ""}`}
                >
                  {token.hyp || "*"}
                </span>
              ))}
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="wer-alignment">
      <div className="wer-alignment-grid" style={gridStyle}>
        <span className="wer-row-label">REF</span>
        {tokens.map((token, index) => (
          <span
            className={`wer-word ${getWerWordClass(token.label, "ref")}`}
            title={`ref: ${token.ref || "*"}\nhyp: ${token.hyp || "*"}`}
            key={`ref:${index}:${token.ref ?? ""}:${token.hyp ?? ""}`}
          >
            {token.ref || "*"}
          </span>
        ))}
        <span className="wer-row-label">HYP</span>
        {tokens.map((token, index) => (
          <span
            className={`wer-word ${getWerWordClass(token.label, "hyp")}`}
            title={`ref: ${token.ref || "*"}\nhyp: ${token.hyp || "*"}`}
            key={`hyp:${index}:${token.ref ?? ""}:${token.hyp ?? ""}`}
          >
            {token.hyp || "*"}
          </span>
        ))}
      </div>
    </div>
  );
}

function chunkWerTokens(tokens: WerToken[]): WerToken[][] {
  const chunkSize = 12;
  const chunks: WerToken[][] = [];
  for (let index = 0; index < tokens.length; index += chunkSize) {
    chunks.push(tokens.slice(index, index + chunkSize));
  }
  return chunks;
}

function getWerWordClass(label: string, row: "ref" | "hyp"): string {
  if (label === "substitution") {
    return "wer-word-substitution";
  }
  if (label === "deletion" && row === "ref") {
    return "wer-word-deletion";
  }
  if (label === "insertion" && row === "hyp") {
    return "wer-word-insertion";
  }
  if ((label === "deletion" && row === "hyp") || (label === "insertion" && row === "ref")) {
    return "wer-word-placeholder";
  }
  return "wer-word-correct";
}

function buildRequest(state: EvaluationFormState): EvaluationRequest {
  return {
    task: state.task,
    target: state.target,
    dataset_path: state.dataset_path,
    split: state.split,
    limit: toOptionalNumber(state.limit),
    language_code: state.language_code,
    sample_rate: toNumber(state.sample_rate, 16000),
    min_reference_words: toNumber(state.min_reference_words, 5),
    hotwords: state.hotwords
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean),
    hotword_bias: toNumber(state.hotword_bias, 0),
    connect_timeout_seconds: toOptionalNumber(state.connect_timeout_seconds),
    request_timeout_seconds: toNumber(state.request_timeout_seconds, 60),
    interim_results: state.interim_results ?? true,
    remove_punctuation: state.remove_punctuation ?? false,
    mask_frame_seconds: toNumber(state.mask_frame_seconds, 0.01),
    chunk_duration_seconds: toNumber(state.chunk_duration_seconds, 0.1),
    hit_threshold: toNumber(state.hit_threshold, 0.9),
    streaming: state.streaming ?? false,
  };
}

function sortAlignmentUtterances(
  utterances: WerUtterance[],
  sortMode: ReportSortMode,
  reports: Record<AlignmentMetric, WerReport | undefined>,
): WerUtterance[] {
  return [...utterances].sort((left, right) => {
    if (sortMode === "index-desc") {
      return getUtteranceIndex(right) - getUtteranceIndex(left);
    }
    if (
      sortMode === "wer-desc" ||
      sortMode === "wer-asc" ||
      sortMode === "cer-desc" ||
      sortMode === "cer-asc"
    ) {
      const metric: AlignmentMetric = sortMode.startsWith("cer") ? "cer" : "wer";
      const leftRate = getUtteranceRate(left, reports[metric]);
      const rightRate = getUtteranceRate(right, reports[metric]);
      if (leftRate !== rightRate) {
        return sortMode.endsWith("desc") ? rightRate - leftRate : leftRate - rightRate;
      }
    }
    return getUtteranceIndex(left) - getUtteranceIndex(right);
  });
}

function getUtteranceRate(
  utterance: WerUtterance,
  report: WerReport | undefined,
): number {
  const metricUtterance = report?.utterances.find(
    (item) => item.id === utterance.id,
  );
  const summary = metricUtterance?.summary ?? utterance.summary;
  if (typeof summary?.wer === "number") {
    return summary.wer;
  }
  const tokens = metricUtterance?.tokens ?? utterance.tokens;
  const referenceWords = tokens.filter(
    (token) => token.label !== "insertion",
  ).length;
  if (referenceWords === 0) {
    return 0;
  }
  const errors = tokens.filter((token) => token.label !== "correct").length;
  return (errors / referenceWords) * 100;
}

function getUtteranceIndex(utterance: WerUtterance): number {
  return utterance.index ?? Number.MAX_SAFE_INTEGER;
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tagName = target.tagName.toLowerCase();
  return (
    target.isContentEditable ||
    tagName === "input" ||
    tagName === "textarea" ||
    tagName === "select" ||
    tagName === "button"
  );
}

function toOptionalNumber(value: string): number | null {
  return value.trim() ? Number(value) : null;
}

function toNumber(value: string, fallback: number): number {
  return value.trim() ? Number(value) : fallback;
}

function formatRate(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "-";
  }
  const percentage = Math.abs(value) <= 1 ? value * 100 : value;
  return `${percentage.toFixed(2)}%`;
}

function formatNumber(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? String(value) : "0";
}

function formatSeconds(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value.toFixed(2)}s`
    : "-";
}

function toPercent(value: number, total: number): number {
  if (!Number.isFinite(value) || !Number.isFinite(total) || total <= 0) {
    return 0;
  }
  return Math.min(Math.max((value / total) * 100, 0), 100);
}

function segmentStatusLabel(status: string): string {
  if (status === "hit") {
    return "命中";
  }
  if (status === "miss") {
    return "漏检";
  }
  if (status === "false_alarm") {
    return "虚警";
  }
  return status;
}

function regionLabel(label: string): string {
  if (label === "hit") {
    return "命中";
  }
  if (label === "miss") {
    return "漏检";
  }
  if (label === "false_alarm") {
    return "虚警";
  }
  if (label === "correct_reject") {
    return "静音正确";
  }
  return label;
}
