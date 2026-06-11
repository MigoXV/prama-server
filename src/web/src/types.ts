export type JobStatus = "queued" | "running" | "completed" | "failed";

export type ThemeMode = "system" | "light" | "dark";
export type EvaluationTask = "asr" | "vad" | "lid" | "denoise" | "keyword";

export interface HelpDocument {
  title: string;
  markdown: string;
}

export interface SqaScore {
  engine_name: string;
  target: string;
  score: number | null;
  error?: string | null;
}

export interface SqaSummary {
  engine_name: string;
  target: string;
  mean_score: number | null;
  scored_count: number;
  failed_count: number;
}

export interface EvaluationRequest {
  task: EvaluationTask;
  target: string;
  dataset_path: string;
  split: string;
  limit: number | null;
  language_code: string;
  sample_rate: number;
  min_reference_words: number;
  hotwords: string[];
  hotword_bias: number;
  connect_timeout_seconds: number | null;
  request_timeout_seconds: number;
  interim_results: boolean;
  inference_concurrency: number;
  asr_inference_concurrency: number;
  vad_inference_concurrency: number;
  lid_inference_concurrency: number;
  enable_mos: boolean;
  mos_target: string;
  enable_snr: boolean;
  snr_target: string;
  sqa_inference_concurrency: number;
  lid_confidence_threshold: number;
  remove_punctuation: boolean;
  mask_frame_seconds: number;
  chunk_duration_seconds: number;
  speech_padding_seconds: number;
  hit_threshold: number;
  streaming: boolean;
}

export interface EvaluationCreated {
  job_id: string;
  status: JobStatus;
}

export interface EvaluationProgress {
  status?: JobStatus | "started";
  tag?: string;
  total?: number;
  processed?: number;
  evaluated?: number;
  id?: string;
  current_id?: string;
  reference?: string;
  hypothesis?: string;
  is_final?: boolean;
  result?: EvaluationResult;
  audio_url?: string;
  duration_seconds?: number;
  sqa_scores?: SqaScore[];
}

export interface EvaluationSnapshot {
  job_id: string;
  status: JobStatus;
  request: EvaluationRequest;
  progress: EvaluationProgress | null;
  result: EvaluationResult | null;
  error: string | null;
}

export interface EvaluationResult {
  wer?: number;
  cer?: number;
  accuracy?: number;
  macro_precision?: number;
  recall?: number;
  known_accuracy?: number;
  macro_recall?: number;
  frame?: VadFrameMetrics;
  segment?: VadSegmentMetrics;
  frame_accuracy?: number;
  frame_recall?: number;
  frame_precision?: number;
  frame_f1?: number;
  segment_recall?: number;
  segment_precision?: number;
  reference_segment_count?: number;
  prediction_segment_count?: number;
  sample_count?: number;
  included_sample_count?: number;
  total_sample_count?: number;
  correct_count?: number;
  known_correct_count?: number;
  known_sample_count?: number;
  overall_correct_count?: number;
  unknown_false_accept_count?: number;
  known_reject_count?: number;
  precision?: number;
  f1?: number;
  hit_count?: number;
  miss_count?: number;
  false_alarm_count?: number;
  correct_reject_count?: number;
  positive_sample_count?: number;
  negative_sample_count?: number;
  lid_language_recalls?: LidLanguageRecall[];
  lid_confusion_matrix?: LidConfusionMatrix;
  audio_duration_seconds?: number;
  processing_elapsed_seconds?: number;
  realtime_factor?: number;
  wer_report?: WerReport;
  cer_report?: WerReport;
  vad_report?: VadReport;
  lid_report?: LidReport;
  denoise_report?: DenoiseReport;
  keyword_report?: KeywordReport;
  sqa_summary?: SqaSummary[];
  [key: string]: unknown;
}

export interface KeywordReport {
  samples: KeywordReportSample[];
}

export interface KeywordReportSample {
  id: string;
  index?: number;
  audio_url?: string;
  duration_seconds?: number;
  keyword: string;
  expected_hit: boolean;
  predicted_hit: boolean;
  correct: boolean;
  transcript: string;
  match_text: string;
  sqa_scores?: SqaScore[];
}

export interface DenoiseReport {
  samples: DenoiseReportSample[];
}

export interface DenoiseReportSample {
  id: string;
  index?: number;
  audio_url?: string;
  denoised_audio_url?: string | null;
  duration_seconds?: number;
  original_sqa_scores?: SqaScore[];
  denoised_sqa_scores?: SqaScore[];
  original_snr?: number | null;
  denoised_snr?: number | null;
  snr_delta?: number | null;
  original_mos?: number | null;
  denoised_mos?: number | null;
  mos_delta?: number | null;
  error?: string | null;
}

export interface LidReport {
  samples: LidReportSample[];
}

export interface LidLanguageRecall {
  language: string;
  correct_count: number;
  sample_count: number;
  predicted_count?: number;
  precision?: number;
  recall: number;
}

export interface LidConfusionMatrix {
  reference_languages: string[];
  predicted_languages: string[];
  rows: LidConfusionMatrixRow[];
}

export interface LidConfusionMatrixRow {
  reference_language: string;
  total: number;
  counts: Record<string, number>;
}

export interface LidReportSample {
  id: string;
  index?: number;
  audio_url?: string;
  duration_seconds?: number;
  reference_language: string;
  predicted_language: string;
  raw_language: string;
  confidence: number;
  correct: boolean;
  sqa_scores?: SqaScore[];
}

export interface VadFrameMetrics {
  frame_total?: number;
  frame_speech?: number;
  frame_non_speech?: number;
  frame_true_positive?: number;
  frame_true_negative?: number;
  frame_false_positive?: number;
  frame_false_negative?: number;
  frame_accuracy?: number;
  frame_recall?: number;
  frame_precision?: number;
  frame_f1?: number;
  frame_specificity?: number;
  frame_false_alarm_rate?: number;
  frame_miss_rate?: number;
  frame_balanced_accuracy?: number;
  [key: string]: unknown;
}

export interface VadSegmentMetrics {
  reference_segment_count?: number;
  prediction_segment_count?: number;
  segment_hit_count?: number;
  segment_miss_count?: number;
  segment_false_alarm_count?: number;
  segment_recall?: number;
  segment_precision?: number;
  segment_f1?: number;
  segment_miss_rate?: number;
  segment_false_alarm_rate?: number;
  [key: string]: unknown;
}

export interface VadReport {
  samples: VadReportSample[];
}

export interface VadReportSample {
  id: string;
  index?: number;
  duration_seconds: number;
  frame_seconds: number;
  audio_url?: string;
  metrics?: EvaluationResult;
  sqa_scores?: SqaScore[];
  reference_segments: VadReportSegment[];
  prediction_segments: VadReportSegment[];
  regions: VadReportRegion[];
}

export type VadSegmentStatus = "hit" | "miss" | "false_alarm" | string;
export type VadRegionLabel = "hit" | "miss" | "false_alarm" | "correct_reject" | string;

export interface VadReportSegment {
  start: number;
  end: number;
  duration: number;
  start_frame: number;
  end_frame: number;
  status: VadSegmentStatus;
}

export interface VadReportRegion {
  start: number;
  end: number;
  duration: number;
  start_frame: number;
  end_frame: number;
  label: VadRegionLabel;
}

export interface WerReport {
  summary: WerSummary;
  utterances: WerUtterance[];
}

export interface WerSummary {
  ref_words: number;
  hyp_words: number;
  correct: number;
  substitutions: number;
  deletions: number;
  insertions: number;
  sentence_count: number;
  sentence_errors: number;
  wer: number;
  accuracy: number;
}

export interface WerUtterance {
  id: string;
  index?: number;
  audio_url?: string;
  duration_seconds?: number;
  sqa_scores?: SqaScore[];
  summary?: WerSummary;
  tokens: WerToken[];
}

export type WerTokenLabel = "correct" | "substitution" | "deletion" | "insertion" | string;

export interface WerToken {
  label: WerTokenLabel;
  ref: string;
  hyp: string;
}

export interface InferenceRow {
  index: number;
  sampleId: string;
  reference: string;
  hypothesis: string;
  audioUrl?: string;
  durationSeconds?: number;
}

export interface EvaluationFormState {
  task: EvaluationTask;
  target: string;
  dataset_path: string;
  split: string;
  limit: string;
  language_code: string;
  sample_rate: string;
  min_reference_words: string;
  hotwords: string;
  hotword_bias: string;
  connect_timeout_seconds: string;
  request_timeout_seconds: string;
  interim_results: boolean;
  inference_concurrency: string;
  asr_inference_concurrency: string;
  vad_inference_concurrency: string;
  lid_inference_concurrency: string;
  enable_mos: boolean;
  mos_target: string;
  enable_snr: boolean;
  snr_target: string;
  sqa_inference_concurrency: string;
  lid_confidence_threshold: string;
  remove_punctuation: boolean;
  mask_frame_seconds: string;
  chunk_duration_seconds: string;
  speech_padding_seconds: string;
  hit_threshold: string;
  streaming: boolean;
}

export interface ServerDirectoryEntry {
  name: string;
  path: string;
  kind: "directory" | "file";
}

export interface ServerDirectoryListing {
  currentPath: string;
  parentPath: string | null;
  entries: ServerDirectoryEntry[];
}

export interface DatasetUploadResult {
  dataset_path: string;
  imported_count: number;
  skipped_count?: number;
  message?: string;
}
