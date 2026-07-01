import {
  Activity,
  BarChart3,
  CheckCircle2,
  CircleDashed,
  Clipboard,
  Download,
  Upload,
  Moon,
  Pause,
  Play,
  Server,
  Settings,
  Sun,
  TriangleAlert,
  Monitor,
} from "lucide-react";
import type { ChangeEvent, FormEvent, MouseEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Button,
  Field,
  GhostButton,
  MetricTile,
  ServerDirectoryBrowserDialog,
  SidebarPane,
  StatusChip,
  SvaraThemeProvider,
  WorkbenchShell,
  WorkspacePane,
} from "svara-ui";
import { usePersistentState } from "./hooks/usePersistentState";
import { useThemeMode } from "./hooks/useThemeMode";
import packageJson from "../package.json";
import {
  createEvaluation,
  getEvaluation,
  listServerDirectory,
  subscribeEvaluationEvents,
  testEngineConnectivity,
  uploadDatasetFiles,
} from "./services/evaluations";
import type {
  EvaluationFormState,
  EvaluationProgress,
  EvaluationRequest,
  EvaluationResult,
  EvaluationSnapshot,
  EvaluationTask,
  DenoiseReportSample,
  JobStatus,
  KeywordAudioReportSample,
  KeywordReportSample,
  LidReportSample,
  SqaScore,
  SqaSummary,
  ThemeMode,
  VadReportRegion,
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
  inference_concurrency: "0",
  asr_inference_concurrency: "0",
  vad_inference_concurrency: "0",
  lid_inference_concurrency: "0",
  enable_mos: false,
  mos_target: "",
  enable_snr: false,
  snr_target: "",
  sqa_inference_concurrency: "0",
  lid_confidence_threshold: "0",
  remove_punctuation: false,
  mask_frame_seconds: "0.01",
  chunk_duration_seconds: "0.1",
  speech_padding_seconds: "0",
  hit_threshold: "0.9",
  streaming: false,
};
const KEYWORD_REPORT_INITIAL_VISIBLE = 100;
const KEYWORD_REPORT_LOAD_STEP = 100;

const APP_VERSION = packageJson.version;

const LAST_EVALUATION_JOB_KEY = "prama.lastEvaluationJobId";
const VAD_TIMELINE_LABEL_WIDTH = 88;
const VAD_TIMELINE_PIXELS_PER_SECOND = 24;
const VAD_TIMELINE_MIN_WIDTH = 760;

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
  lid: {
    target: "192.168.0.222:50026",
    dataset_path: "data-bin/audiofolder/lid-demo",
    min_reference_words: "0",
  },
  keyword: {
    target: "192.168.0.222:50011",
    dataset_path: "data-bin/audiofolder/keyword-demo",
    min_reference_words: "0",
  },
  denoise: {
    target: "192.168.0.222:50027",
    dataset_path: "data-bin/audiofolder/denoise-demo",
    min_reference_words: "0",
  },
};

type TaskRememberedFields = Pick<EvaluationFormState, "target" | "dataset_path">;

const TASK_REMEMBERED_DEFAULTS: Record<EvaluationTask, TaskRememberedFields> = {
  asr: {
    target: "192.168.0.222:50011",
    dataset_path: "data-bin/audiofolder/asr-demo",
  },
  vad: {
    target: "192.168.0.222:50021",
    dataset_path: "data-bin/audiofolder/vad-demo",
  },
  lid: {
    target: "192.168.0.222:50026",
    dataset_path: "data-bin/audiofolder/lid-demo",
  },
  keyword: {
    target: "192.168.0.222:50011",
    dataset_path: "data-bin/audiofolder/keyword-demo",
  },
  denoise: {
    target: "192.168.0.222:50027",
    dataset_path: "data-bin/audiofolder/denoise-demo",
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

type ConsoleModule = "evaluation" | "settings";

const MODULES: Array<{
  id: ConsoleModule;
  label: string;
  icon: typeof Activity;
}> = [
  { id: "evaluation", label: "在线评估", icon: Activity },
  { id: "settings", label: "设置", icon: Settings },
];

type AlignmentMetric = "wer" | "cer";
type ReportSortMode =
  | "index-asc"
  | "index-desc"
  | "wer-desc"
  | "wer-asc"
  | "cer-desc"
  | "cer-asc";
type ConnectivityState = "idle" | "testing" | "ok" | "failed";
type ConnectivityStatus = {
  state: ConnectivityState;
  message: string;
};
type EvaluationRunState = {
  status: JobStatus | "idle" | "started";
  jobId: string;
  progress: EvaluationProgress | null;
  finalResult: EvaluationResult | null;
  errorMessage: string;
  connectionWarning: string;
  busy: boolean;
};
type TaskEventClosers = Record<EvaluationTask, (() => void) | null>;
const EMPTY_RUN_STATE: EvaluationRunState = {
  status: "idle",
  jobId: "",
  progress: null,
  finalResult: null,
  errorMessage: "",
  connectionWarning: "",
  busy: false,
};

function createRunState(): EvaluationRunState {
  return { ...EMPTY_RUN_STATE };
}

function createTaskRunStates(): Record<EvaluationTask, EvaluationRunState> {
  return {
    asr: createRunState(),
    vad: createRunState(),
    lid: createRunState(),
    keyword: createRunState(),
    denoise: createRunState(),
  };
}

function createTaskEventClosers(): TaskEventClosers {
  return {
    asr: null,
    vad: null,
    lid: null,
    keyword: null,
    denoise: null,
  };
}

export default function App() {
  const { themeMode, effectiveTheme, setThemeMode } = useThemeMode();
  const [storedFormState, setFormState] = usePersistentState<EvaluationFormState>(
    "prama.evaluationForm",
    DEFAULT_FORM_STATE,
  );
  const formState = normalizeFormState(storedFormState);
  const [taskRememberedFields, setTaskRememberedFields] = usePersistentState<
    Record<EvaluationTask, TaskRememberedFields>
  >("prama.taskRememberedFields", TASK_REMEMBERED_DEFAULTS);
  const [runStates, setRunStates] = useState<Record<EvaluationTask, EvaluationRunState>>(
    () => createTaskRunStates(),
  );
  const [rememberedJobId, setRememberedJobId] = usePersistentState<string>(
    LAST_EVALUATION_JOB_KEY,
    "",
  );
  const [connectivityStatus, setConnectivityStatus] = useState<
    Record<string, ConnectivityStatus>
  >({});
  const [datasetUploading, setDatasetUploading] = useState(false);
  const [activeModule, setActiveModule] = useState<ConsoleModule>("evaluation");
  const [activeTab, setActiveTab] = useState<"overview" | "report">("overview");
  const [activeAlignmentMetric, setActiveAlignmentMetric] =
    useState<AlignmentMetric>("wer");
  const [reportSort, setReportSort] = useState<ReportSortMode>("index-asc");
  const [wrapWerAlignment, setWrapWerAlignment] = useState(false);
  const evaluationConnectivityKey = `evaluation:${formState.task}`;
  const [directoryBrowserOpen, setDirectoryBrowserOpen] = useState(false);
  const [directoryBrowserPath, setDirectoryBrowserPath] = useState(
    formState.dataset_path,
  );
  const eventClosersRef = useRef<TaskEventClosers>(createTaskEventClosers());
  const datasetUploadInputRef = useRef<HTMLInputElement | null>(null);
  const formRef = useRef<HTMLFormElement | null>(null);
  const activeRunState = runStates[formState.task] ?? EMPTY_RUN_STATE;
  const status = activeRunState.status;
  const jobId = activeRunState.jobId;
  const progress = activeRunState.progress;
  const finalResult = activeRunState.finalResult;
  const errorMessage = activeRunState.errorMessage;
  const connectionWarning = activeRunState.connectionWarning;
  const busy = activeRunState.busy;

  function updateTaskRunState(
    task: EvaluationTask,
    update:
      | Partial<EvaluationRunState>
      | ((current: EvaluationRunState) => EvaluationRunState),
  ) {
    setRunStates((current) => {
      const currentTaskState = current[task] ?? createRunState();
      const nextTaskState =
        typeof update === "function"
          ? update(currentTaskState)
          : { ...currentTaskState, ...update };
      return {
        ...current,
        [task]: nextTaskState,
      };
    });
  }

  useEffect(() => {
    return () => {
      Object.values(eventClosersRef.current).forEach((closeEvents) => {
        closeEvents?.();
      });
    };
  }, []);

  useEffect(() => {
    if (
      activeModule !== "evaluation" ||
      jobId ||
      activeRunState.busy ||
      !rememberedJobId
    ) {
      return;
    }

    let canceled = false;
    getEvaluation(rememberedJobId)
      .then((snapshot) => {
        if (canceled) {
          return;
        }
        applyEvaluationSnapshot(snapshot);
        if (snapshot.status === "queued" || snapshot.status === "running") {
          subscribeToEvaluation(snapshot.job_id, snapshot.request.task);
        }
      })
      .catch((error) => {
        if (canceled) {
          return;
        }
        const message =
          error instanceof Error ? error.message : "最近评估任务恢复失败";
        if (message.includes("评估任务不存在")) {
          setRememberedJobId("");
          return;
        }
        updateTaskRunState(formState.task, {
          connectionWarning: message,
        });
      });

    return () => {
      canceled = true;
    };
  }, [activeModule, rememberedJobId, jobId, activeRunState.busy, formState.task]);

  useEffect(() => {
    datasetUploadInputRef.current?.setAttribute("webkitdirectory", "");
  }, [activeModule]);

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
  const lidReport = finalResult?.lid_report;
  const denoiseReport = finalResult?.denoise_report;
  const keywordReport = finalResult?.keyword_report;
  const canExport = finalResult !== null;
  const displayTask = formState.task;
  const isVad = displayTask === "vad";
  const isLid = displayTask === "lid";
  const isDenoise = displayTask === "denoise";
  const isKeyword = displayTask === "keyword";

  function handleTaskChange(task: EvaluationTask) {
    if (task === formState.task) {
      return;
    }
    const nextTaskRemembered =
      taskRememberedFields[task] ?? TASK_REMEMBERED_DEFAULTS[task];
    setTaskRememberedFields((current) => ({
      ...current,
      [formState.task]: pickTaskRememberedFields(formState),
    }));
    setFormState((current) => ({
      ...current,
      ...TASK_DEFAULTS[task],
      ...nextTaskRemembered,
      task,
    }));
    setDirectoryBrowserPath(nextTaskRemembered.dataset_path);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const requestTask = formState.task;
    updateTaskRunState(requestTask, {
      busy: true,
      errorMessage: "",
      connectionWarning: "",
    });

    try {
      const request = buildRequest(formState);
      const created = await createEvaluation(request);
      closeTaskEvents(request.task);
      setRememberedJobId(created.job_id);
      updateTaskRunState(request.task, {
        status: created.status,
        jobId: created.job_id,
        progress: null,
        finalResult: null,
        errorMessage: "",
        connectionWarning: "",
        busy: true,
      });
      subscribeToEvaluation(created.job_id, request.task);
    } catch (error) {
      updateTaskRunState(requestTask, (current) => ({
        ...current,
        errorMessage: error instanceof Error ? error.message : "评估任务创建失败",
        busy: false,
      }));
    }
  }

  async function handleTestConnectivity(key: string, target: string) {
    const normalizedTarget = target.trim();
    if (!normalizedTarget) {
      setConnectivityStatus((current) => ({
        ...current,
        [key]: { state: "failed", message: "引擎地址不能为空" },
      }));
      return;
    }
    setConnectivityStatus((current) => ({
      ...current,
      [key]: { state: "testing", message: "测试中" },
    }));
    try {
      const result = await testEngineConnectivity(
        normalizedTarget,
        toOptionalNumber(formState.connect_timeout_seconds),
      );
      setConnectivityStatus((current) => ({
        ...current,
        [key]: {
          state: result.ok ? "ok" : "failed",
          message: result.message,
        },
      }));
    } catch (error) {
      setConnectivityStatus((current) => ({
        ...current,
        [key]: {
          state: "failed",
          message: error instanceof Error ? error.message : "连接测试失败",
        },
      }));
    }
  }

  function applyEvaluationSnapshot(snapshot: EvaluationSnapshot) {
    setRememberedJobId(snapshot.job_id);
    updateTaskRunState(snapshot.request.task, {
      status: snapshot.status,
      jobId: snapshot.job_id,
      progress: snapshot.progress,
      finalResult: snapshot.result,
      errorMessage: snapshot.error ?? "",
      busy: snapshot.status === "queued" || snapshot.status === "running",
    });
  }

  function subscribeToEvaluation(nextJobId: string, task: EvaluationTask) {
    closeTaskEvents(task);
    const closeEvents = subscribeEvaluationEvents(nextJobId, {
      onProgress: (nextProgress) => {
        updateTaskRunState(task, (current) => ({
          ...current,
          connectionWarning: "",
          progress: nextProgress,
          status: nextProgress.status ?? "running",
          finalResult: nextProgress.result ?? current.finalResult,
        }));
      },
      onPartialProgress: (nextProgress) => {
        updateTaskRunState(task, (current) => ({
          ...current,
          connectionWarning: "",
          progress: nextProgress,
          status: nextProgress.status ?? "running",
        }));
      },
      onDone: (snapshot) => {
        applyEvaluationSnapshot(snapshot);
        if (eventClosersRef.current[task] === closeEvents) {
          eventClosersRef.current[task] = null;
        }
      },
      onError: (message) => {
        updateTaskRunState(task, (current) => ({
          ...current,
          status: "failed",
          errorMessage: message,
          busy: false,
        }));
      },
      onConnectionError: () => {
        updateTaskRunState(task, (current) => ({
          ...current,
          connectionWarning: "事件流连接暂时不可用",
        }));
      },
    });
    eventClosersRef.current[task] = closeEvents;
  }

  function closeTaskEvents(task: EvaluationTask) {
    eventClosersRef.current[task]?.();
    eventClosersRef.current[task] = null;
  }

  function updateField(field: keyof EvaluationFormState, value: string) {
    setFormState((current) => ({ ...current, [field]: value }));
    if (field === "target" || field === "dataset_path") {
      setTaskRememberedFields((current) => ({
        ...current,
        [formState.task]: {
          ...(current[formState.task] ?? TASK_REMEMBERED_DEFAULTS[formState.task]),
          [field]: value,
        },
      }));
    }
  }

  function setBooleanField(field: keyof EvaluationFormState, value: boolean) {
    setFormState((current) => ({ ...current, [field]: value }));
  }

  async function handleImportDataset(files: File[]) {
    const result = await uploadDatasetFiles(files);
    updateField("dataset_path", result.dataset_path);
    setDirectoryBrowserPath(result.dataset_path);
    return {
      importedCount: result.imported_count,
      skippedCount: result.skipped_count,
      message: result.message,
    };
  }

  async function handleDatasetUploadChange(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    if (files.length === 0) {
      return;
    }
    setDatasetUploading(true);
    updateTaskRunState(formState.task, { errorMessage: "" });
    try {
      await handleImportDataset(files);
    } catch (error) {
      updateTaskRunState(formState.task, {
        errorMessage: error instanceof Error ? error.message : "数据集上传失败",
      });
    } finally {
      setDatasetUploading(false);
      event.target.value = "";
    }
  }

  function resetForm() {
    setFormState(DEFAULT_FORM_STATE);
    setTaskRememberedFields(TASK_REMEMBERED_DEFAULTS);
    setDirectoryBrowserPath(DEFAULT_FORM_STATE.dataset_path);
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

  return (
    <SvaraThemeProvider
      mode={effectiveTheme}
      applyBackground={false}
      className="app-theme-root"
    >
      <WorkbenchShell className="console-frame" sidebarWidth={232} variant="bare">
        <SidebarPane className="sidebar" variant="bare">
        <div className="sidebar-brand">
          <div className="brand-symbol">
            <Server size={19} />
          </div>
          <div>
            <div className="brand-title-row">
              <strong>Prama</strong>
              <small>v{APP_VERSION}</small>
            </div>
            <span>评估控制台</span>
          </div>
        </div>

        <nav className="module-nav" aria-label="主导航">
          {MODULES.map((item) => {
            const Icon = item.icon;
            return (
              <div className="module-group" key={item.id}>
                <button
                  type="button"
                  className={`module-item ${activeModule === item.id ? "active" : ""}`}
                  onClick={() => setActiveModule(item.id)}
                >
                  <Icon size={17} />
                  <span>{item.label}</span>
                </button>
                {item.id === "evaluation" ? (
                  <div className="task-nav" aria-label="评估类型">
                    <button
                      type="button"
                      className={formState.task === "asr" ? "active" : ""}
                      onClick={() => {
                        setActiveModule("evaluation");
                        handleTaskChange("asr");
                      }}
                    >
                      ASR
                    </button>
                    <button
                      type="button"
                      className={formState.task === "vad" ? "active" : ""}
                      onClick={() => {
                        setActiveModule("evaluation");
                        handleTaskChange("vad");
                      }}
                    >
                      VAD
                    </button>
                    <button
                      type="button"
                      className={formState.task === "lid" ? "active" : ""}
                      onClick={() => {
                        setActiveModule("evaluation");
                        handleTaskChange("lid");
                      }}
                    >
                      LID
                    </button>
                    <button
                      type="button"
                      className={formState.task === "keyword" ? "active" : ""}
                      onClick={() => {
                        setActiveModule("evaluation");
                        handleTaskChange("keyword");
                      }}
                    >
                      Keyword
                    </button>
                    <button
                      type="button"
                      className={formState.task === "denoise" ? "active" : ""}
                      onClick={() => {
                        setActiveModule("evaluation");
                        handleTaskChange("denoise");
                      }}
                    >
                      SE
                    </button>
                  </div>
                ) : null}
              </div>
            );
          })}
        </nav>

        </SidebarPane>

        <WorkspacePane
          className={`workspace ${
            activeModule === "evaluation" ? "" : "workspace-compact"
          }`}
          variant="bare"
        >
        {activeModule === "evaluation" ? (
          <header className="workspace-header">
            <div>
              <p className="eyebrow">{evaluationTaskEnglishLabel(formState.task)}</p>
              <h1>{evaluationTaskTitle(formState.task)}</h1>
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
        ) : null}

        {activeModule === "evaluation" ? (
          <div className="page-tabs" role="tablist" aria-label="评估视图">
            <TabButton
              active={activeTab === "overview"}
              label="运行概览"
              meta={STATUS_LABELS[status]}
              onClick={() => setActiveTab("overview")}
            />
            <TabButton
              active={activeTab === "report"}
              label={
                isVad
                  ? "VAD 指标"
                  : isLid
                    ? "LID 报告"
                    : isKeyword
                      ? "关键词报告"
                      : isDenoise
                        ? "SE 报告"
                        : "对齐报告"
              }
              meta={
                isVad
                  ? `${formatNumber(finalResult?.sample_count)} 个样本`
                  : isLid
                    ? `${lidReport?.samples.length ?? 0} 个样本`
                    : isKeyword
                      ? `${keywordReport?.samples.length ?? 0} 个样本`
                      : isDenoise
                        ? `${denoiseReport?.samples.length ?? 0} 个样本`
                        : `${werReport?.utterances.length ?? cerReport?.utterances.length ?? 0} 个样本`
              }
              onClick={() => setActiveTab("report")}
            />
          </div>
        ) : null}

        <section
          className={`work-grid ${
            activeModule === "evaluation" ? "" : "single-column"
          }`}
        >
          {activeModule === "evaluation" ? (
            <form
              ref={formRef}
              className="panel evaluation-form svara-surface svara-surface-padded"
              onSubmit={handleSubmit}
            >
              <div className="panel-heading">
                <div>
                  <h2>{evaluationTaskShortLabel(formState.task)} 在线评估</h2>
                  <span>Job {jobId || "-"}</span>
                </div>
              </div>

              <div className="field-grid">
                <div className="engine-target-row">
                  <TextField
                    label={`${evaluationTaskShortLabel(formState.task)} 引擎地址`}
                    value={formState.target}
                    onChange={(value) => updateField("target", value)}
                    required
                  />
                  <ConnectivityButton
                    status={connectivityStatus[evaluationConnectivityKey]}
                    onClick={() =>
                      void handleTestConnectivity(
                        evaluationConnectivityKey,
                        formState.target,
                      )
                    }
                  />
                </div>
                <label className="field dataset-path-field">
                  <span>数据集路径</span>
                  <div className="dataset-path-controls">
                    <input
                      value={formState.dataset_path}
                      required
                      disabled={busy || datasetUploading}
                      onChange={(event) => updateField("dataset_path", event.target.value)}
                    />
                    <input
                      ref={datasetUploadInputRef}
                      type="file"
                      hidden
                      multiple
                      accept=".wav,.mp3,.flac,.ogg,.json,.jsonl,.csv,.parquet,.txt"
                      onChange={handleDatasetUploadChange}
                    />
                    <GhostButton
                      className="dataset-action-button dataset-upload-button"
                      disabled={busy || datasetUploading}
                      onClick={() => datasetUploadInputRef.current?.click()}
                    >
                      <Upload size={14} />
                      <span>{datasetUploading ? "上传中" : "上传"}</span>
                    </GhostButton>
                    <GhostButton
                      className="dataset-action-button dataset-browser-button"
                      disabled={busy || datasetUploading}
                      onClick={() => {
                        setDirectoryBrowserPath(formState.dataset_path);
                        setDirectoryBrowserOpen(true);
                      }}
                    >
                      浏览
                    </GhostButton>
                  </div>
                </label>
                <div className="dataset-options-grid">
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
              </div>

              <div className="form-action-bar">
                <Button type="submit" className="primary-action" disabled={busy || datasetUploading} stretch>
                  {busy ? "评估中..." : "启动评估"}
                </Button>
              </div>
            </form>
          ) : null}

          {activeModule === "settings" ? (
            <section className="panel configuration-panel settings-panel">
              <div className="panel-heading">
                <div>
                  <h2>设置</h2>
                  <span>ASR、VAD、LID 与 SE 的高级评估参数</span>
                </div>
                <button type="button" className="ghost-button" onClick={resetForm}>
                  重置
                </button>
              </div>
              <div className="settings-stack">
                <section className="settings-section">
                  <div className="settings-section-heading">
                    <h3>界面设置</h3>
                  </div>
                  <div className="theme-switcher settings-theme-switcher" aria-label="主题切换">
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
                </section>

                <section className="settings-section">
                  <div className="settings-section-heading">
                    <h3>通用设置</h3>
                  </div>
                  <div className="field-grid advanced-grid">
                    <TextField
                      label="采样率"
                      value={formState.sample_rate}
                      type="number"
                      min="1"
                      onChange={(value) => updateField("sample_rate", value)}
                      required
                    />
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
                  </div>
                </section>

                <section className="settings-section">
                  <div className="settings-section-heading">
                    <h3>SQA 语音质量评估</h3>
                  </div>
                  <div className="field-grid advanced-grid">
                    <TextField
                      label="MOS/SNR 推理并发数"
                      value={formState.sqa_inference_concurrency}
                      type="number"
                      min="0"
                      step="1"
                      onChange={(value) => updateField("sqa_inference_concurrency", value)}
                      required
                    />
                  </div>
                  <div className="sqa-fixed-list">
                    <div className="sqa-fixed-row">
                      <label className="check-field">
                        <input
                          type="checkbox"
                          checked={formState.enable_mos}
                          onChange={(event) =>
                            setBooleanField("enable_mos", event.target.checked)
                          }
                        />
                        <span>MOS</span>
                      </label>
                      <TextField
                        label="MOS 地址"
                        value={formState.mos_target}
                        onChange={(value) => updateField("mos_target", value)}
                      />
                      <ConnectivityButton
                        status={connectivityStatus.mos}
                        onClick={() =>
                          void handleTestConnectivity("mos", formState.mos_target)
                        }
                      />
                    </div>
                    <div className="sqa-fixed-row">
                      <label className="check-field">
                        <input
                          type="checkbox"
                          checked={formState.enable_snr}
                          onChange={(event) =>
                            setBooleanField("enable_snr", event.target.checked)
                          }
                        />
                        <span>SNR</span>
                      </label>
                      <TextField
                        label="SNR 地址"
                        value={formState.snr_target}
                        onChange={(value) => updateField("snr_target", value)}
                      />
                      <ConnectivityButton
                        status={connectivityStatus.snr}
                        onClick={() =>
                          void handleTestConnectivity("snr", formState.snr_target)
                        }
                      />
                    </div>
                  </div>
                </section>

                <section className="settings-section">
                  <div className="settings-section-heading">
                    <h3>ASR 高级设置</h3>
                  </div>
                  <div className="field-grid advanced-grid">
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
                    <TextField
                      label="ASR 推理并发数"
                      value={formState.asr_inference_concurrency}
                      type="number"
                      min="0"
                      step="1"
                      onChange={(value) =>
                        updateField("asr_inference_concurrency", value)
                      }
                      required
                    />
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
                  </div>
                </section>

                <section className="settings-section">
                  <div className="settings-section-heading">
                    <h3>VAD 高级设置</h3>
                  </div>
                  <div className="field-grid advanced-grid">
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
                      label="语音扩展秒"
                      value={formState.speech_padding_seconds ?? "0"}
                      type="number"
                      min="0"
                      step="0.01"
                      onChange={(value) => updateField("speech_padding_seconds", value)}
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
                    <TextField
                      label="VAD 推理并发数"
                      value={formState.vad_inference_concurrency}
                      type="number"
                      min="0"
                      step="1"
                      onChange={(value) =>
                        updateField("vad_inference_concurrency", value)
                      }
                      required
                    />
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
                  </div>
                </section>
                <section className="settings-section">
                  <div className="settings-section-heading">
                    <h3>LID 高级设置</h3>
                  </div>
                  <div className="field-grid advanced-grid">
                    <TextField
                      label="LID 推理并发数"
                      value={formState.lid_inference_concurrency}
                      type="number"
                      min="0"
                      step="1"
                      onChange={(value) =>
                        updateField("lid_inference_concurrency", value)
                      }
                      required
                    />
                    <TextField
                      label="LID 置信度阈值"
                      value={formState.lid_confidence_threshold}
                      type="number"
                      min="0"
                      max="1"
                      step="0.01"
                      onChange={(value) =>
                        updateField("lid_confidence_threshold", value)
                      }
                      required
                    />
                  </div>
                </section>
              </div>
            </section>
          ) : null}

          {activeModule === "evaluation" ? (
          <section
            className={`run-column ${
              activeTab === "report" ? "report-column" : "overview-column"
            }`}
          >
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
                    <Metric label="已评估" value={String(progress?.evaluated ?? 0)} />
                  </div>
                  <PerformanceMetrics result={finalResult} />
                  <SqaSummaryMetrics summary={finalResult?.sqa_summary} />
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
                {isLid ? <LidOverviewMetrics result={finalResult} /> : null}
                {isKeyword ? <KeywordOverviewMetrics result={finalResult} /> : null}
                {isDenoise ? <DenoiseOverviewMetrics result={finalResult} /> : null}
                {!isVad && !isLid && !isKeyword && !isDenoise ? (
                  <AsrOverviewMetrics result={finalResult} />
                ) : null}

                {!isVad && !isDenoise ? (
                  <div className="panel sample-panel">
                    <div className="panel-heading compact-heading">
                      <div>
                        <h2>当前样本</h2>
                        <span>{progress?.current_id || progress?.id || "-"}</span>
                      </div>
                    </div>
                    <div className="sample-grid">
                      <TextBlock
                        label={
                          isLid
                            ? "Reference Language"
                            : isKeyword
                              ? "Expected"
                              : "Reference"
                        }
                        value={progress?.reference || "-"}
                      />
                      <TextBlock
                        label={
                          isLid || isKeyword ? "Prediction" : "Hypothesis"
                        }
                        value={progress?.hypothesis || "-"}
                      />
                    </div>
                  </div>
                ) : null}
              </>
            ) : null}

            {activeTab === "report" ? (
              isVad ? (
                <VadReportPanel
                  result={finalResult}
                />
              ) : isLid ? (
                <LidReportPanel
                  result={finalResult}
                />
              ) : isDenoise ? (
                <DenoiseReportPanel
                  result={finalResult}
                />
              ) : isKeyword ? (
                <KeywordReportPanel
                  result={finalResult}
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
                />
              )
            ) : null}
          </section>
          ) : null}
        </section>
        <ServerDirectoryBrowserDialog
          isOpen={directoryBrowserOpen}
          initialPath={directoryBrowserPath}
          listDirectory={listServerDirectory}
          onClose={() => setDirectoryBrowserOpen(false)}
          onSelect={(path) => {
            updateField("dataset_path", path);
            setDirectoryBrowserPath(path);
            setDirectoryBrowserOpen(false);
          }}
        />
        </WorkspacePane>
      </WorkbenchShell>
    </SvaraThemeProvider>
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
    <StatusChip className={`status-pill status-${status}`}>
      <Icon size={15} />
      {STATUS_LABELS[status]}
    </StatusChip>
  );
}

function ConnectivityButton({
  status,
  onClick,
}: {
  status?: ConnectivityStatus;
  onClick: () => void;
}) {
  const state = status?.state ?? "idle";
  const label =
    state === "testing"
      ? "测试中"
      : state === "ok"
        ? "已连接"
        : state === "failed"
          ? "失败"
          : "测试";
  return (
    <button
      type="button"
      className={`connectivity-button ${state}`}
      title={status?.message || "测试 gRPC 连通性"}
      disabled={state === "testing"}
      onClick={onClick}
    >
      {label}
    </button>
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
    <Field label={label} className="field">
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
    </Field>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <MetricTile className="metric" label={label} value={value} />;
}

function TextBlock({ label, value }: { label: string; value: string }) {
  return (
    <div className="text-block">
      <span>{label}</span>
      <p>{value}</p>
    </div>
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
  const total = result?.total_sample_count;
  return (
    <div className="metric-strip sample-count-strip">
      <Metric label="参与样本" value={formatNumber(included)} />
      {typeof total === "number" && total !== included ? (
        <Metric label="总样本" value={formatNumber(total)} />
      ) : null}
    </div>
  );
}

function PerformanceMetrics({ result }: { result: EvaluationResult | null }) {
  if (!result) {
    return null;
  }

  return (
    <div className="metric-strip performance-metrics">
      <Metric label="音频时长" value={formatSeconds(result.audio_duration_seconds)} />
      <Metric label="处理耗时" value={formatSeconds(result.processing_elapsed_seconds)} />
      <Metric label="倍时" value={formatRealtimeFactor(result.realtime_factor)} />
    </div>
  );
}

function CompactReportMeta({
  result,
  fallbackCount,
  className = "",
}: {
  result: EvaluationResult | null;
  fallbackCount: number;
  className?: string;
}) {
  const included = result?.included_sample_count ?? fallbackCount;
  const total = result?.total_sample_count;
  const sampleText =
    typeof total === "number" && total !== included
      ? `${formatNumber(included)} / ${formatNumber(total)}`
      : formatNumber(included);
  const items = [
    { label: "参与样本", value: sampleText },
    { label: "音频时长", value: formatSeconds(result?.audio_duration_seconds) },
    { label: "处理耗时", value: formatSeconds(result?.processing_elapsed_seconds) },
    { label: "倍时", value: formatRealtimeFactor(result?.realtime_factor) },
  ];

  return (
    <div
      className={`compact-report-meta ${className}`.trim()}
      aria-label="评估运行元信息"
    >
      {items.map((item) => (
        <span key={item.label}>
          <em>{item.label}</em>
          <strong>{item.value}</strong>
        </span>
      ))}
    </div>
  );
}

function SqaSummaryMetrics({ summary }: { summary?: SqaSummary[] }) {
  if (!summary?.length) {
    return null;
  }

  return (
    <div className="metric-strip sqa-summary-metrics">
      {summary.map((item) => (
        <Metric
          key={`${item.engine_name}-${item.target}`}
          label={item.engine_name}
          value={formatSqaScore(item.mean_score)}
        />
      ))}
    </div>
  );
}

function SqaScoreChips({ scores }: { scores?: SqaScore[] }) {
  if (!scores?.length) {
    return null;
  }

  return (
    <span className="sqa-score-chips">
      {scores.map((item) => {
        const failed = item.score === null || item.score === undefined || item.error;
        const title = failed
          ? `${item.engine_name}: ${item.error || "无有效分数"}`
          : `${item.engine_name}: ${item.target}`;
        return (
          <span
            className={`sqa-score-chip ${failed ? "failed" : ""}`}
            key={`${item.engine_name}-${item.target}`}
            title={title}
          >
            <em>{item.engine_name}</em>
            <strong>{formatSqaScore(item.score)}</strong>
          </span>
        );
      })}
    </span>
  );
}

function AsrOverviewMetrics({ result }: { result: EvaluationResult | null }) {
  if (!result) {
    return null;
  }
  const wordAccuracy =
    result.word_accuracy ??
    result.accuracy ??
    result.wer_report?.summary?.accuracy;
  const characterAccuracy =
    result.character_accuracy ??
    result.cer_report?.summary?.accuracy;

  return (
    <div className="panel asr-overview-panel">
      <div className="metric-strip asr-overview-metrics">
        <Metric label="词正确率" value={formatRate(wordAccuracy)} />
        <Metric label="字正确率" value={formatRate(characterAccuracy)} />
        <Metric label="WER" value={formatRate(result.wer)} />
        <Metric label="CER" value={formatRate(result.cer)} />
      </div>
      <SampleCountStrip
        result={result}
        fallbackCount={
          result.wer_report?.utterances.length ??
          result.cer_report?.utterances.length ??
          0
        }
      />
    </div>
  );
}

function VadOverviewMetrics({ result }: { result: EvaluationResult | null }) {
  if (!result) {
    return null;
  }

  return <VadMetricGroups metrics={result} />;
}

function LidOverviewMetrics({ result }: { result: EvaluationResult | null }) {
  if (!result) {
    return null;
  }

  return (
    <div className="metric-strip lid-overview-metrics">
      <Metric label="准确率" value={formatRate(result.macro_precision ?? result.precision)} />
      <Metric label="召回率" value={formatRate(result.macro_recall ?? result.recall)} />
    </div>
  );
}

function KeywordOverviewMetrics({ result }: { result: EvaluationResult | null }) {
  if (!result) {
    return null;
  }

  return (
    <div className="panel keyword-overview-panel">
      <div className="metric-strip keyword-overview-metrics">
        <Metric label="Accuracy" value={formatRate(result.accuracy)} />
        <Metric label="Precision" value={formatRate(result.precision)} />
        <Metric label="Recall" value={formatRate(result.recall)} />
        <Metric label="F1" value={formatRate(result.f1)} />
        <Metric label="Miss" value={formatNumber(result.miss_count)} />
        <Metric label="False Alarm" value={formatNumber(result.false_alarm_count)} />
      </div>
      <SampleCountStrip
        result={result}
        fallbackCount={result.keyword_report?.samples.length ?? 0}
      />
    </div>
  );
}

function DenoiseOverviewMetrics({ result }: { result: EvaluationResult | null }) {
  if (!result) {
    return null;
  }

  return (
    <div className="panel denoise-overview-panel">
      <div className="metric-strip denoise-overview-metrics">
        <Metric label="SNR Δ" value={formatSignedScore(result.mean_snr_delta)} />
        <Metric label="MOS Δ" value={formatSignedScore(result.mean_mos_delta)} />
        <Metric label="SNR Samples" value={formatNumber(result.scored_snr_sample_count)} />
        <Metric label="MOS Samples" value={formatNumber(result.scored_mos_sample_count)} />
        <Metric label="Failed" value={formatNumber(result.failed_sample_count)} />
      </div>
      <SampleCountStrip
        result={result}
        fallbackCount={result.denoise_report?.samples.length ?? 0}
      />
    </div>
  );
}

function VadMetricGroups({ metrics }: { metrics: EvaluationResult }) {
  const frame = metrics.frame;

  return (
    <div className="vad-metric-groups">
      <section className="vad-metric-section">
        <div className="vad-metric-title">
          <span>帧级指标</span>
          <strong>{formatRate(frame?.frame_precision ?? metrics.frame_precision)}</strong>
        </div>
        <div className="report-summary vad-summary vad-frame-summary">
          <Metric
            label="召回率"
            value={formatRate(frame?.frame_recall ?? metrics.frame_recall)}
          />
          <Metric
            label="准确率"
            value={formatRate(frame?.frame_precision ?? metrics.frame_precision)}
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
    <div className="panel report-panel asr-report-panel compact-report-panel">
      <div className="panel-heading compact-heading">
        <div>
          <h2>对齐报告</h2>
          <span>{sampleCount} 个样本</span>
        </div>
        <div className="report-controls">
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
              accuracy={result?.word_accuracy ?? result?.accuracy}
              accuracyLabel="词正确率"
            />
            <AsrMetricSummaryCard
              label="CER"
              summary={cerReport?.summary}
              fallbackRate={result?.cer}
              accuracy={result?.character_accuracy}
              accuracyLabel="字正确率"
            />
          </div>
          <CompactReportMeta result={result} fallbackCount={sampleCount} />
          <SqaSummaryMetrics summary={result?.sqa_summary} />
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
            {utterances.map((utterance, index) => (
              <details
                className="utterance"
                key={utterance.id}
                open={index < 3}
              >
                <summary>
                  <span className="utterance-title">
                    <strong>#{utterance.index ?? "-"}</strong>
                    <span>{utterance.id || "-"}</span>
                  </span>
                  <TokenCounts
                    metricLabel={activeLabel}
                    summary={utterance.summary}
                    tokens={utterance.tokens}
                  />
                  <SqaScoreChips scores={utterance.sqa_scores} />
                </summary>
                <div className="utterance-body">
                  <div className="utterance-audio-row">
                    <AudioPlayer
                      src={utterance.audio_url}
                      durationSeconds={utterance.duration_seconds}
                    />
                  </div>
                  <WerAlignmentRows tokens={utterance.tokens} wrap={wrapAlignment} />
                </div>
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
}: {
  result: EvaluationResult | null;
}) {
  const samples = result?.vad_report?.samples ?? [];
  return (
    <div className="panel report-panel vad-report-panel compact-report-panel">
      <div className="panel-heading compact-heading">
        <div>
          <h2>VAD 报告</h2>
          <span>{formatNumber(result?.sample_count)} 个样本</span>
        </div>
        <div className="report-controls vad-report-controls">
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
          <CompactReportMeta result={result} fallbackCount={samples.length} />
          <SqaSummaryMetrics summary={result.sqa_summary} />
          <VadMaskReport samples={samples} />
        </>
      ) : (
        <div className="empty-state">评估完成后生成 VAD 指标</div>
      )}
    </div>
  );
}

function LidReportPanel({
  result,
}: {
  result: EvaluationResult | null;
}) {
  const samples = result?.lid_report?.samples ?? [];
  return (
    <div className="panel report-panel lid-report-panel compact-report-panel">
      <div className="panel-heading compact-heading">
        <div>
          <h2>LID 报告</h2>
          <span>{formatNumber(result?.sample_count)} 个样本</span>
        </div>
      </div>
      {result ? (
        <>
          <div className="report-summary lid-summary">
            <Metric
              label="准确率"
              value={formatRate(result.macro_precision ?? result.precision)}
            />
            <Metric
              label="召回率"
              value={formatRate(result.macro_recall ?? result.recall)}
            />
          </div>
          <CompactReportMeta result={result} fallbackCount={samples.length} />
          <SqaSummaryMetrics summary={result.sqa_summary} />
          <LidMetricsDetails result={result} />
          <LidSampleList samples={samples} />
        </>
      ) : (
        <div className="empty-state">评估完成后生成 LID 报告</div>
      )}
    </div>
  );
}

function KeywordReportPanel({
  result,
}: {
  result: EvaluationResult | null;
}) {
  const samples = result?.keyword_report?.samples ?? [];
  const audioSamples = result?.keyword_audio_report?.samples ?? [];
  const [viewMode, setViewMode] = useState<"keyword" | "audio">("keyword");
  return (
    <div className="panel report-panel keyword-report-panel compact-report-panel">
      <div className="panel-heading compact-heading">
        <div>
          <h2>关键词报告</h2>
          <span>
            {formatNumber(result?.sample_count)} 个关键词 /{" "}
            {formatNumber(result?.audio_sample_count ?? audioSamples.length)} 条语音
          </span>
        </div>
      </div>
      {result ? (
        <>
          <div className="report-summary keyword-summary">
            <Metric label="Accuracy" value={formatRate(result.accuracy)} />
            <Metric label="Precision" value={formatRate(result.precision)} />
            <Metric label="Recall" value={formatRate(result.recall)} />
            <Metric label="F1" value={formatRate(result.f1)} />
            <Metric label="Hit" value={formatNumber(result.hit_count)} />
            <Metric label="Miss" value={formatNumber(result.miss_count)} />
            <Metric
              label="False Alarm"
              value={formatNumber(result.false_alarm_count)}
            />
            <Metric
              label="Correct Reject"
              value={formatNumber(result.correct_reject_count)}
            />
          </div>
          <CompactReportMeta result={result} fallbackCount={samples.length} />
          <SqaSummaryMetrics summary={result.sqa_summary} />
          <div className="keyword-view-tabs" role="tablist" aria-label="关键词报告视图">
            <button
              type="button"
              className={viewMode === "keyword" ? "active" : ""}
              onClick={() => setViewMode("keyword")}
            >
              按关键词
            </button>
            <button
              type="button"
              className={viewMode === "audio" ? "active" : ""}
              onClick={() => setViewMode("audio")}
            >
              按语音
            </button>
          </div>
          {viewMode === "keyword" ? (
            <KeywordSampleList samples={samples} />
          ) : (
            <KeywordAudioSampleList samples={audioSamples} />
          )}
        </>
      ) : (
        <div className="empty-state">评估完成后生成关键词报告</div>
      )}
    </div>
  );
}

function KeywordAudioSampleList({ samples }: { samples: KeywordAudioReportSample[] }) {
  const [visibleCount, setVisibleCount] = useState(KEYWORD_REPORT_INITIAL_VISIBLE);
  useEffect(() => {
    setVisibleCount(KEYWORD_REPORT_INITIAL_VISIBLE);
  }, [samples]);

  if (!samples.length) {
    return <div className="empty-state">评估完成后生成语音聚合结果</div>;
  }

  const visibleSamples = samples.slice(0, visibleCount);
  const hasMore = visibleCount < samples.length;

  return (
    <>
      <KeywordListToolbar
        total={samples.length}
        visible={visibleSamples.length}
        onCollapse={
          visibleSamples.length > KEYWORD_REPORT_INITIAL_VISIBLE
            ? () => setVisibleCount(KEYWORD_REPORT_INITIAL_VISIBLE)
            : undefined
        }
      />
      <div className="keyword-sample-list">
        {visibleSamples.map((sample) => (
          <section className="keyword-sample keyword-audio-sample" key={sample.id}>
            <div className="keyword-sample-title">
              <span className="keyword-sample-name">
                <strong>#{sample.index ?? "-"}</strong>
                <span title={sample.id}>{sample.id}</span>
              </span>
              <span className="keyword-status">
                {formatNumber(sample.keywords.length)} 个关键词
              </span>
            </div>
            <AudioPlayer
              src={sample.audio_url}
              durationSeconds={sample.duration_seconds}
            />
            <SqaScoreChips scores={sample.sqa_scores} />
            <div className="keyword-token-list">
              {sample.keywords.map((keyword) => (
                <div
                  className={`keyword-token ${keyword.correct ? "correct" : "incorrect"}`}
                  key={`${keyword.id}-${keyword.keyword}`}
                >
                  <span title={keyword.id}>{keyword.keyword}</span>
                  <small>
                    {keyword.expected_hit ? "Expected Hit" : "Expected No Hit"} /{" "}
                    {keyword.predicted_hit ? "Predicted Hit" : "Predicted No Hit"}
                  </small>
                </div>
              ))}
            </div>
            <div className="keyword-transcript-grid">
              <TextBlock label="Transcript" value={sample.transcript || "-"} />
              <TextBlock label="Match Text" value={sample.match_text || "-"} />
            </div>
          </section>
        ))}
        {hasMore ? (
          <KeywordLoadMoreButton
            onClick={() =>
              setVisibleCount((current) =>
                Math.min(current + KEYWORD_REPORT_LOAD_STEP, samples.length),
              )
            }
          />
        ) : null}
      </div>
    </>
  );
}

function KeywordSampleList({ samples }: { samples: KeywordReportSample[] }) {
  const [visibleCount, setVisibleCount] = useState(KEYWORD_REPORT_INITIAL_VISIBLE);
  useEffect(() => {
    setVisibleCount(KEYWORD_REPORT_INITIAL_VISIBLE);
  }, [samples]);

  if (!samples.length) {
    return <div className="empty-state">评估完成后生成关键词结果</div>;
  }

  const visibleSamples = samples.slice(0, visibleCount);
  const hasMore = visibleCount < samples.length;

  return (
    <>
      <KeywordListToolbar
        total={samples.length}
        visible={visibleSamples.length}
        onCollapse={
          visibleSamples.length > KEYWORD_REPORT_INITIAL_VISIBLE
            ? () => setVisibleCount(KEYWORD_REPORT_INITIAL_VISIBLE)
            : undefined
        }
      />
      <div className="keyword-sample-list">
        {visibleSamples.map((sample) => (
          <section
            className={`keyword-sample ${sample.correct ? "correct" : "incorrect"}`}
            key={sample.id}
          >
            <div className="keyword-sample-title">
              <span className="keyword-sample-name">
                <strong>#{sample.index ?? "-"}</strong>
                <span title={sample.id}>{sample.id}</span>
              </span>
              <span className={`keyword-status ${sample.correct ? "correct" : "incorrect"}`}>
                {sample.correct ? "正确" : "错误"}
              </span>
            </div>
            <AudioPlayer
              src={sample.audio_url}
              durationSeconds={sample.duration_seconds}
            />
            <div className="keyword-sample-metrics">
              <Metric label="Keyword" value={sample.keyword || "-"} />
              <Metric label="Expected" value={sample.expected_hit ? "Hit" : "No Hit"} />
              <Metric label="Prediction" value={sample.predicted_hit ? "Hit" : "No Hit"} />
            </div>
            <SqaScoreChips scores={sample.sqa_scores} />
            <div className="keyword-transcript-grid">
              <TextBlock label="Transcript" value={sample.transcript || "-"} />
              <TextBlock label="Match Text" value={sample.match_text || "-"} />
            </div>
          </section>
        ))}
        {hasMore ? (
          <KeywordLoadMoreButton
            onClick={() =>
              setVisibleCount((current) =>
                Math.min(current + KEYWORD_REPORT_LOAD_STEP, samples.length),
              )
            }
          />
        ) : null}
      </div>
    </>
  );
}

function KeywordListToolbar({
  total,
  visible,
  onCollapse,
}: {
  total: number;
  visible: number;
  onCollapse?: () => void;
}) {
  return (
    <div className="keyword-list-toolbar">
      <span>
        已显示 {formatNumber(visible)} / {formatNumber(total)}
      </span>
      {onCollapse ? (
        <button type="button" onClick={onCollapse}>
          收起
        </button>
      ) : null}
    </div>
  );
}

function KeywordLoadMoreButton({ onClick }: { onClick: () => void }) {
  return (
    <button type="button" className="keyword-load-more" onClick={onClick}>
      加载更多
    </button>
  );
}

function DenoiseReportPanel({
  result,
}: {
  result: EvaluationResult | null;
}) {
  const samples = result?.denoise_report?.samples ?? [];
  return (
    <div className="panel report-panel denoise-report-panel compact-report-panel">
      <div className="panel-heading compact-heading">
        <div>
          <h2>SE 报告</h2>
          <span>{formatNumber(result?.sample_count)} 个样本</span>
        </div>
      </div>
      {result ? (
        <>
          <div className="report-summary denoise-summary">
            <Metric label="SNR Δ" value={formatSignedScore(result.mean_snr_delta)} />
            <Metric label="MOS Δ" value={formatSignedScore(result.mean_mos_delta)} />
            <Metric label="SNR Before" value={formatSqaScore(result.mean_original_snr)} />
            <Metric label="SNR After" value={formatSqaScore(result.mean_denoised_snr)} />
            <Metric label="MOS Before" value={formatSqaScore(result.mean_original_mos)} />
            <Metric label="MOS After" value={formatSqaScore(result.mean_denoised_mos)} />
          </div>
          <CompactReportMeta result={result} fallbackCount={samples.length} />
          <DenoiseSampleList samples={samples} />
        </>
      ) : (
        <div className="empty-state">评估完成后生成 SE 报告</div>
      )}
    </div>
  );
}

function DenoiseSampleList({ samples }: {
  samples: DenoiseReportSample[];
}) {
  if (!samples.length) {
    return <div className="empty-state">评估完成后生成 SE 结果</div>;
  }

  return (
    <div className="denoise-sample-list">
      {samples.map((sample) => (
        <section
          className={`denoise-sample ${sample.error ? "failed" : ""}`}
          key={sample.id}
        >
          <div className="denoise-sample-title">
            <span className="denoise-sample-name">
              <strong>#{sample.index ?? "-"}</strong>
              <span title={sample.id}>{sample.id}</span>
            </span>
            {sample.error ? (
              <span className="denoise-error" title={sample.error}>
                {sample.error}
              </span>
            ) : null}
          </div>
          <div className="denoise-audio-grid">
            <div>
              <span>原始</span>
              <AudioPlayer
                src={sample.audio_url}
                durationSeconds={sample.duration_seconds}
              />
            </div>
            <div>
              <span>SE</span>
              <AudioPlayer
                src={sample.denoised_audio_url || undefined}
                durationSeconds={sample.duration_seconds}
              />
            </div>
          </div>
          <DenoiseSampleMetrics sample={sample} />
        </section>
      ))}
    </div>
  );
}

function DenoiseSampleMetrics({ sample }: { sample: DenoiseReportSample }) {
  const rows = [
    {
      label: "SNR",
      before: sample.original_snr,
      after: sample.denoised_snr,
      delta: sample.snr_delta,
    },
    {
      label: "MOS",
      before: sample.original_mos,
      after: sample.denoised_mos,
      delta: sample.mos_delta,
    },
  ].filter(
    (row) =>
      hasFiniteNumber(row.before) ||
      hasFiniteNumber(row.after) ||
      hasFiniteNumber(row.delta),
  );

  if (!rows.length) {
    return <div className="denoise-metric-empty">暂无质量指标</div>;
  }

  return (
    <div className="denoise-metric-table" role="table" aria-label="SE 质量指标">
      <div className="denoise-metric-row denoise-metric-header" role="row">
        <span role="columnheader">指标</span>
        <span role="columnheader">Before</span>
        <span role="columnheader">After</span>
        <span role="columnheader">Δ</span>
      </div>
      {rows.map((row) => (
        <div className="denoise-metric-row" role="row" key={row.label}>
          <strong role="rowheader">{row.label}</strong>
          <span role="cell">{formatSqaScore(row.before)}</span>
          <span role="cell">{formatSqaScore(row.after)}</span>
          <span className="delta" role="cell">
            {formatSignedScore(row.delta)}
          </span>
        </div>
      ))}
    </div>
  );
}

function LidMetricsDetails({ result }: { result: EvaluationResult }) {
  const recalls = getKnownLidLanguageRecalls(result);
  const hasRecalls = recalls.length > 0;
  const matrix = result.lid_confusion_matrix;
  const hasMatrix =
    (matrix?.rows ?? []).length > 0 && (matrix?.predicted_languages ?? []).length > 0;
  const overallCorrect = result.overall_correct_count ?? result.correct_count;
  const errorCount = getLidErrorCount(result);

  if (!hasRecalls && !hasMatrix) {
    return null;
  }

  return (
    <details className="lid-metrics-details">
      <summary>
        <span>类别指标与混淆矩阵</span>
        <small>
          {recalls.length} 类 / 正确 {formatNumber(overallCorrect)} / 错误{" "}
          {formatNumber(errorCount)}
        </small>
      </summary>
      <div className="lid-metrics-scroll">
        <LidMetricsTables result={result} />
      </div>
    </details>
  );
}

function LidMetricsTables({ result }: { result: EvaluationResult }) {
  const recalls = getKnownLidLanguageRecalls(result);
  const matrix = result.lid_confusion_matrix;
  const matrixRows = matrix?.rows ?? [];
  const predictedLanguages = matrix?.predicted_languages ?? [];
  const hasMatrix = matrixRows.length > 0 && predictedLanguages.length > 0;

  if (!recalls.length && !hasMatrix) {
    return null;
  }

  return (
    <div className="lid-metrics-grid">
      {recalls.length ? (
        <section className="metric-table-section">
          <div className="metric-table-heading">
            <h3>类别指标</h3>
          </div>
          <div className="table-wrap lid-recall-table">
            <table>
              <thead>
                <tr>
                  <th>真实标签</th>
                  <th>正确数</th>
                  <th>真实总数</th>
                  <th>预测总数</th>
                  <th>准确率</th>
                  <th>召回率</th>
                </tr>
              </thead>
              <tbody>
                {recalls.map((item) => (
                  <tr
                    className={item.recall < 0.9 ? "low-recall" : ""}
                    key={item.language}
                  >
                    <td title={item.language}>{item.language || "-"}</td>
                    <td>{formatNumber(item.correct_count)}</td>
                    <td>{formatNumber(item.sample_count)}</td>
                    <td>{formatNumber(item.predicted_count)}</td>
                    <td>{formatRate(item.precision)}</td>
                    <td>{formatRate(item.recall)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
      {hasMatrix ? (
        <section className="metric-table-section">
          <div className="metric-table-heading">
            <h3>混淆矩阵</h3>
          </div>
          <div className="table-wrap lid-confusion-table">
            <table>
              <thead>
                <tr>
                  <th title="真实标签 / 预测标签">真实/预测</th>
                  {predictedLanguages.map((language) => (
                    <th key={language} title={language}>
                      {language || "-"}
                    </th>
                  ))}
                  <th>总数</th>
                  <th>召回率</th>
                </tr>
              </thead>
              <tbody>
                {matrixRows.map((row) => (
                  <tr key={row.reference_language}>
                    <th title={row.reference_language}>{row.reference_language || "-"}</th>
                    {predictedLanguages.map((language) => (
                      <td
                        className={lidConfusionCellClass(
                          row.reference_language,
                          language,
                          row.counts[language] ?? 0,
                          row.total,
                        )}
                        key={`${row.reference_language}:${language}`}
                        title={`${row.reference_language || "-"} -> ${language || "-"}: ${formatNumber(row.counts[language] ?? 0)}`}
                      >
                        {formatNumber(row.counts[language] ?? 0)}
                      </td>
                    ))}
                    <td>{formatNumber(row.total)}</td>
                    <td>{formatRate(getLidMatrixRowRecall(result, row.reference_language))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}
    </div>
  );
}

function lidConfusionCellClass(
  referenceLanguage: string,
  predictedLanguage: string,
  count: number,
  total: number,
): string {
  const ratio = total > 0 ? count / total : 0;
  const level = count > 0 ? Math.max(1, Math.min(5, Math.ceil(ratio * 5))) : 0;
  const status = referenceLanguage === predictedLanguage ? "diagonal" : "error";
  return `confusion-cell ${status} heat-${level}`;
}

function getLidErrorCount(result: EvaluationResult): number | undefined {
  const sampleCount = toDisplayNumber(result.sample_count);
  const correctCount = toDisplayNumber(result.overall_correct_count ?? result.correct_count);
  if (sampleCount === undefined || correctCount === undefined) {
    return undefined;
  }
  return Math.max(0, sampleCount - correctCount);
}

function getKnownLidLanguageRecalls(result: EvaluationResult) {
  return (result.lid_language_recalls ?? []).filter(
    (item) => item.language !== "<others>",
  );
}

function getLidMatrixRowRecall(
  result: EvaluationResult,
  referenceLanguage: string,
): number | undefined {
  if (referenceLanguage === "<others>") {
    return undefined;
  }
  const recall = result.lid_language_recalls?.find(
    (item) => item.language === referenceLanguage,
  )?.recall;
  if (typeof recall === "number") {
    return recall;
  }
  const row = result.lid_confusion_matrix?.rows.find(
    (item) => item.reference_language === referenceLanguage,
  );
  if (!row || row.total <= 0) {
    return undefined;
  }
  return (row.counts[referenceLanguage] ?? 0) / row.total;
}

function toDisplayNumber(value: unknown): number | undefined {
  const numeric = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numeric) ? numeric : undefined;
}

function LidSampleList({ samples }: {
  samples: LidReportSample[];
}) {
  if (!samples.length) {
    return <div className="empty-state">评估完成后生成 LID 结果</div>;
  }

  return (
    <div className="lid-sample-list">
      {samples.map((sample) => (
        <section
          className={`lid-sample ${sample.correct ? "correct" : "incorrect"}`}
          key={sample.id}
        >
          <div className="lid-sample-title">
            <span className="lid-sample-name">
              <strong>#{sample.index ?? "-"}</strong>
              <span title={sample.id}>{sample.id}</span>
            </span>
            <div className="lid-result-line">
              <span>
                <em>真实</em>
                <strong>{sample.reference_language || "-"}</strong>
              </span>
              <span>
                <em>预测</em>
                <strong>{sample.predicted_language || "-"}</strong>
              </span>
              <span>
                <em>置信度</em>
                <strong>{formatConfidence(sample.confidence)}</strong>
              </span>
            </div>
            <SqaScoreChips scores={sample.sqa_scores} />
            <div className="lid-sample-actions">
              <b>{sample.correct ? "正确" : "错误"}</b>
              <AudioPlayer
                src={sample.audio_url}
                durationSeconds={sample.duration_seconds}
              />
            </div>
          </div>
        </section>
      ))}
    </div>
  );
}

function AsrMetricSummaryCard({
  label,
  summary,
  fallbackRate,
  accuracy,
  accuracyLabel,
}: {
  label: "WER" | "CER";
  summary?: WerSummary;
  fallbackRate?: number;
  accuracy?: number;
  accuracyLabel?: string;
}) {
  return (
    <section className="asr-summary-card">
      <div className="asr-summary-title">
        <span>{label}</span>
        <strong>{formatRate(summary?.wer ?? fallbackRate)}</strong>
      </div>
      <div className="report-summary">
        <Metric
          label={accuracyLabel ?? "Accuracy"}
          value={formatRate(summary?.accuracy ?? accuracy)}
        />
        <Metric label="Correct" value={formatNumber(summary?.correct)} />
        <Metric label="Sub" value={formatNumber(summary?.substitutions)} />
        <Metric label="Del" value={formatNumber(summary?.deletions)} />
        <Metric label="Ins" value={formatNumber(summary?.insertions)} />
      </div>
    </section>
  );
}

function VadMaskReport({ samples }: {
  samples: VadReportSample[];
}) {
  if (!samples.length) {
    return <div className="empty-state">评估完成后生成 mask 对齐报告</div>;
  }

  return (
    <div className="vad-report-list">
      {samples.map((sample, index) => {
        const timelineWidth = getVadTimelineWidth(sample.duration_seconds);
        return (
          <details className="vad-sample" key={sample.id} open={index < 3}>
            <summary>
              <span>
                #{sample.index ?? "-"} {sample.id}
              </span>
              <AudioPlayer
                src={sample.audio_url}
                durationSeconds={sample.duration_seconds}
              />
              <SqaScoreChips scores={sample.sqa_scores} />
              <small>{formatSeconds(sample.duration_seconds)}</small>
            </summary>
            <div className="vad-sample-body">
              {sample.metrics ? (
                <div className="sample-vad-metrics">
                  <VadMetricGroups metrics={sample.metrics} />
                </div>
              ) : null}
              <div className="vad-timeline-scroll" aria-label="VAD 时间轴">
                <div
                  className="vad-timeline-canvas"
                  style={{ width: timelineWidth }}
                >
                  <VadTimeRuler duration={sample.duration_seconds} />
                  <div className="mask-stack">
                    <MaskTrack
                      label="Reference"
                      duration={sample.duration_seconds}
                      segments={sample.reference_segments}
                      regions={sample.regions}
                      track="reference"
                    />
                    <MaskTrack
                      label="Prediction"
                      duration={sample.duration_seconds}
                      segments={sample.prediction_segments}
                      regions={sample.regions}
                      track="prediction"
                    />
                    <div className="mask-row region-row">
                      <span>Errors</span>
                      <div className="region-track" aria-label="VAD 对齐区域">
                        {sample.regions.map((region) => (
                          <span
                            className={`region region-${normalizeVadLabel(region.label)}`}
                            key={`${sample.id}:${region.start_frame}:${region.end_frame}:${region.label}`}
                            style={{
                              left: `${toPercent(
                                region.start,
                                sample.duration_seconds,
                              )}%`,
                              width: `${toPercent(
                                region.duration,
                                sample.duration_seconds,
                              )}%`,
                            }}
                            title={`${regionLabel(region.label)} ${formatSeconds(region.start)} - ${formatSeconds(region.end)}`}
                          />
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </details>
        );
      })}
    </div>
  );
}

function VadTimeRuler({ duration }: { duration: number }) {
  const ticks = buildVadRulerTicks(duration);
  return (
    <div className="vad-time-ruler">
      <span>Time</span>
      <div className="vad-ruler-track">
        {ticks.map((tick) => (
          <i
            key={tick}
            style={{ left: `${toPercent(tick, duration)}%` }}
            title={formatSeconds(tick)}
          >
            {formatSeconds(tick)}
          </i>
        ))}
      </div>
    </div>
  );
}

function MaskTrack({
  label,
  duration,
  segments,
  regions,
  track,
}: {
  label: string;
  duration: number;
  segments: VadReportSegment[];
  regions: VadReportRegion[];
  track: "reference" | "prediction";
}) {
  const visibleSegments =
    regions.length > 0 ? vadRegionsToTrackSegments(regions, track) : segments;

  return (
    <div className="mask-row">
      <span>{label}</span>
      <div className="mask-track">
        {visibleSegments.map((segment) => (
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

function vadRegionsToTrackSegments(
  regions: VadReportRegion[],
  track: "reference" | "prediction",
): VadReportSegment[] {
  return regions
    .filter((region) => {
      const label = normalizeVadLabel(region.label);
      return track === "reference"
        ? label === "hit" || label === "miss"
        : label === "hit" || label === "false_alarm";
    })
    .map((region) => ({
      start: region.start,
      end: region.end,
      duration: region.duration,
      start_frame: region.start_frame,
      end_frame: region.end_frame,
      status: normalizeVadLabel(region.label),
    }));
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
      const label = normalizeWerTokenLabel(token.label);
      current[label] = (current[label] ?? 0) + 1;
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
  const normalizedLabel = normalizeWerTokenLabel(label);
  if (normalizedLabel === "substitution") {
    return "wer-word-substitution";
  }
  if (normalizedLabel === "deletion" && row === "ref") {
    return "wer-word-deletion";
  }
  if (normalizedLabel === "insertion" && row === "hyp") {
    return "wer-word-insertion";
  }
  if (
    (normalizedLabel === "deletion" && row === "hyp") ||
    (normalizedLabel === "insertion" && row === "ref")
  ) {
    return "wer-word-placeholder";
  }
  return "wer-word-correct";
}

function normalizeWerTokenLabel(label: string): string {
  const normalized = label.trim().toLowerCase();
  if (["sub", "subst", "substitution", "s"].includes(normalized)) {
    return "substitution";
  }
  if (["del", "delete", "deletion", "d"].includes(normalized)) {
    return "deletion";
  }
  if (["ins", "insert", "insertion", "i"].includes(normalized)) {
    return "insertion";
  }
  if (["cor", "correct", "ok", "c"].includes(normalized)) {
    return "correct";
  }
  return normalized;
}

function buildRequest(rawState: EvaluationFormState): EvaluationRequest {
  const state = normalizeFormState(rawState);
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
    inference_concurrency: 0,
    asr_inference_concurrency: toNumber(state.asr_inference_concurrency, 0),
    vad_inference_concurrency: toNumber(state.vad_inference_concurrency, 0),
    lid_inference_concurrency: toNumber(state.lid_inference_concurrency, 0),
    enable_mos: state.enable_mos ?? false,
    mos_target: state.mos_target.trim(),
    enable_snr: state.enable_snr ?? false,
    snr_target: state.snr_target.trim(),
    sqa_inference_concurrency: toNumber(state.sqa_inference_concurrency, 0),
    lid_confidence_threshold: toNumber(state.lid_confidence_threshold, 0),
    remove_punctuation: state.remove_punctuation ?? false,
    mask_frame_seconds: toNumber(state.mask_frame_seconds, 0.01),
    chunk_duration_seconds: toNumber(state.chunk_duration_seconds, 0.1),
    speech_padding_seconds: toNumber(state.speech_padding_seconds ?? "0", 0),
    hit_threshold: toNumber(state.hit_threshold, 0.9),
    streaming: state.streaming ?? false,
  };
}

function normalizeFormState(state: EvaluationFormState): EvaluationFormState {
  const { enable_sqa: _enableSqa, sqa_engines: _sqaEngines, ...knownState } =
    state as EvaluationFormState & {
      enable_sqa?: unknown;
      sqa_engines?: unknown;
    };
  return {
    ...DEFAULT_FORM_STATE,
    ...knownState,
    enable_mos:
      typeof state.enable_mos === "boolean"
        ? state.enable_mos
        : DEFAULT_FORM_STATE.enable_mos,
    mos_target:
      typeof state.mos_target === "string"
        ? state.mos_target
        : DEFAULT_FORM_STATE.mos_target,
    enable_snr:
      typeof state.enable_snr === "boolean"
        ? state.enable_snr
        : DEFAULT_FORM_STATE.enable_snr,
    snr_target:
      typeof state.snr_target === "string"
        ? state.snr_target
        : DEFAULT_FORM_STATE.snr_target,
    sqa_inference_concurrency:
      typeof state.sqa_inference_concurrency === "string"
        ? state.sqa_inference_concurrency
        : DEFAULT_FORM_STATE.sqa_inference_concurrency,
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
    (token) => normalizeWerTokenLabel(token.label) !== "insertion",
  ).length;
  if (referenceWords === 0) {
    return 0;
  }
  const errors = tokens.filter(
    (token) => normalizeWerTokenLabel(token.label) !== "correct",
  ).length;
  return (errors / referenceWords) * 100;
}

function getUtteranceIndex(utterance: WerUtterance): number {
  return utterance.index ?? Number.MAX_SAFE_INTEGER;
}

function evaluationTaskShortLabel(task: EvaluationTask): string {
  if (task === "vad") {
    return "VAD";
  }
  if (task === "lid") {
    return "LID";
  }
  if (task === "keyword") {
    return "Keyword";
  }
  if (task === "denoise") {
    return "SE";
  }
  return "ASR";
}

function evaluationTaskEnglishLabel(task: EvaluationTask): string {
  if (task === "denoise") {
    return "SE Evaluation";
  }
  return `${evaluationTaskShortLabel(task)} Evaluation`;
}

function evaluationTaskTitle(task: EvaluationTask): string {
  if (task === "vad") {
    return "VAD 评估";
  }
  if (task === "lid") {
    return "LID 评估";
  }
  if (task === "keyword") {
    return "关键词评估";
  }
  if (task === "denoise") {
    return "SE 评估";
  }
  return "ASR 评估";
}

function pickTaskRememberedFields(
  state: EvaluationFormState,
): TaskRememberedFields {
  return {
    target: state.target,
    dataset_path: state.dataset_path,
  };
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

function formatRealtimeFactor(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value.toFixed(2)}x`
    : "-";
}

function formatConfidence(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(4)
    : "-";
}

function formatSqaScore(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(2)
    : "-";
}

function hasFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function formatSignedScore(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "-";
  }
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

function getVadTimelineWidth(duration: number): number {
  if (!Number.isFinite(duration) || duration <= 0) {
    return VAD_TIMELINE_MIN_WIDTH;
  }
  return Math.max(
    VAD_TIMELINE_MIN_WIDTH,
    Math.ceil(
      VAD_TIMELINE_LABEL_WIDTH + duration * VAD_TIMELINE_PIXELS_PER_SECOND,
    ),
  );
}

function buildVadRulerTicks(duration: number): number[] {
  if (!Number.isFinite(duration) || duration <= 0) {
    return [0];
  }
  const targetTickCount = Math.max(4, Math.ceil(duration / 12));
  const roughStep = duration / targetTickCount;
  const step = pickRulerStep(roughStep);
  const ticks: number[] = [];
  for (let tick = 0; tick < duration; tick += step) {
    ticks.push(Number(tick.toFixed(3)));
  }
  if (ticks[ticks.length - 1] !== duration) {
    ticks.push(duration);
  }
  return ticks;
}

function pickRulerStep(roughStep: number): number {
  const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  return steps.find((step) => step >= roughStep) ?? steps[steps.length - 1];
}

function toPercent(value: number, total: number): number {
  if (!Number.isFinite(value) || !Number.isFinite(total) || total <= 0) {
    return 0;
  }
  return Math.min(Math.max((value / total) * 100, 0), 100);
}

function segmentStatusLabel(status: string): string {
  const normalizedStatus = normalizeVadLabel(status);
  if (normalizedStatus === "hit") {
    return "命中";
  }
  if (normalizedStatus === "miss") {
    return "漏检";
  }
  if (normalizedStatus === "false_alarm") {
    return "虚警";
  }
  return status;
}

function regionLabel(label: string): string {
  const normalizedLabel = normalizeVadLabel(label);
  if (normalizedLabel === "hit") {
    return "命中";
  }
  if (normalizedLabel === "miss") {
    return "漏检";
  }
  if (normalizedLabel === "false_alarm") {
    return "虚警";
  }
  if (normalizedLabel === "correct_reject") {
    return "静音正确";
  }
  return label;
}

function normalizeVadLabel(label: string): string {
  const normalized = label.trim().toLowerCase().replace(/[-\s]+/g, "_");
  if (["falsealarm", "false_alarm", "fa", "fp"].includes(normalized)) {
    return "false_alarm";
  }
  if (["miss", "missed", "fn"].includes(normalized)) {
    return "miss";
  }
  if (["hit", "tp", "speech_correct"].includes(normalized)) {
    return "hit";
  }
  if (
    ["correctreject", "correct_reject", "tn", "silence_correct"].includes(
      normalized,
    )
  ) {
    return "correct_reject";
  }
  return normalized;
}
