import type {
  EvaluationFormState,
  EvaluationTask,
  EvaluationRequest,
  JobStatus,
} from "../../types";
export const DEFAULT_FORM_STATE: EvaluationFormState = {
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

export const TASK_DEFAULTS: Record<
  EvaluationTask,
  Partial<EvaluationFormState>
> = {
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

type TaskRememberedFields = Pick<
  EvaluationFormState,
  "target" | "dataset_path"
>;

export const TASK_REMEMBERED_DEFAULTS: Record<
  EvaluationTask,
  TaskRememberedFields
> = {
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

export const STATUS_LABELS: Record<JobStatus | "idle" | "started", string> = {
  idle: "未启动",
  queued: "排队中",
  running: "运行中",
  started: "已开始",
  completed: "已完成",
  failed: "失败",
};

export function buildRequest(rawState: EvaluationFormState): EvaluationRequest {
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

export function requestToFormState(
  request: EvaluationRequest,
): EvaluationFormState {
  return {
    task: request.task,
    target: request.target,
    dataset_path: request.dataset_path,
    split: request.split,
    limit: request.limit === null ? "" : String(request.limit),
    language_code: request.language_code,
    sample_rate: String(request.sample_rate),
    min_reference_words: String(request.min_reference_words),
    hotwords: request.hotwords.join(", "),
    hotword_bias: String(request.hotword_bias),
    connect_timeout_seconds:
      request.connect_timeout_seconds === null
        ? ""
        : String(request.connect_timeout_seconds),
    request_timeout_seconds: String(request.request_timeout_seconds),
    interim_results: request.interim_results,
    inference_concurrency: String(request.inference_concurrency),
    asr_inference_concurrency: String(request.asr_inference_concurrency),
    vad_inference_concurrency: String(request.vad_inference_concurrency),
    lid_inference_concurrency: String(request.lid_inference_concurrency),
    enable_mos: request.enable_mos,
    mos_target: request.mos_target,
    enable_snr: request.enable_snr,
    snr_target: request.snr_target,
    sqa_inference_concurrency: String(request.sqa_inference_concurrency),
    lid_confidence_threshold: String(request.lid_confidence_threshold),
    remove_punctuation: request.remove_punctuation,
    mask_frame_seconds: String(request.mask_frame_seconds),
    chunk_duration_seconds: String(request.chunk_duration_seconds),
    speech_padding_seconds: String(request.speech_padding_seconds),
    hit_threshold: String(request.hit_threshold),
    streaming: request.streaming,
  };
}

export function mergeReevaluationFormState(
  currentState: EvaluationFormState,
  request: EvaluationRequest,
): EvaluationFormState {
  return {
    ...normalizeFormState(currentState),
    task: request.task,
    target: request.target,
    dataset_path: request.dataset_path,
    split: request.split,
    limit: request.limit === null ? "" : String(request.limit),
  };
}

export function normalizeFormState(
  state: EvaluationFormState,
): EvaluationFormState {
  const {
    enable_sqa: _enableSqa,
    sqa_engines: _sqaEngines,
    ...knownState
  } = state as EvaluationFormState & {
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

function toOptionalNumber(value: string) {
  return value.trim() ? Number(value) : null;
}
function toNumber(value: string, fallback: number) {
  return value.trim() ? Number(value) : fallback;
}

export const TASKS: {
  id: EvaluationTask;
  label: string;
  description: string;
}[] = [
  { id: "asr", label: "ASR", description: "语音识别" },
  { id: "vad", label: "VAD", description: "语音活动检测" },
  { id: "lid", label: "LID", description: "语种识别" },
  { id: "keyword", label: "关键词", description: "关键词检出" },
  { id: "denoise", label: "SE", description: "语音增强" },
];
export function taskLabel(task: EvaluationTask) {
  return TASKS.find((t) => t.id === task)?.label ?? task;
}
export function initialDraft(
  task: EvaluationTask,
  defaults = DEFAULT_FORM_STATE,
): EvaluationFormState {
  return { ...defaults, ...TASK_DEFAULTS[task], task };
}
export function readStored<T>(key: string, fallback: T): T {
  try {
    return JSON.parse(localStorage.getItem(key) ?? "null") ?? fallback;
  } catch {
    return fallback;
  }
}
export function migratedDrafts(): Partial<
  Record<EvaluationTask, EvaluationFormState>
> {
  if (!readStored("prama.evaluationForm", null)) return {};
  const old = normalizeFormState(
    readStored("prama.evaluationForm", DEFAULT_FORM_STATE),
  );
  const remembered = readStored(
    "prama.taskRememberedFields",
    TASK_REMEMBERED_DEFAULTS,
  );
  return Object.fromEntries(
    TASKS.map(({ id }) => [
      id,
      id === old.task ? old : { ...initialDraft(id, old), ...remembered[id] },
    ]),
  );
}
