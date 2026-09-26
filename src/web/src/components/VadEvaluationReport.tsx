import { ChevronLeft, Pause, Play } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import type {
  EvaluationProgress,
  EvaluationRequest,
  EvaluationResult,
  JobStatus,
  VadReportRegion,
  VadReportSample,
  VadReportSegment,
} from "../types";

export type VadDiagnosisFilter =
  "all" | "miss" | "false_alarm" | "low_frame_f1" | "oversegmentation";

type VadSampleDiagnostic = {
  sample: VadReportSample;
  frameF1: number | null;
  missDuration: number;
  falseAlarmDuration: number;
  missCount: number;
  falseAlarmCount: number;
  oversegmentationCount: number;
};

type VadEvaluationReportProps = {
  result: EvaluationResult;
  request: EvaluationRequest | null;
  view: "overview" | "diagnosis";
  filter: VadDiagnosisFilter;
  search?: string;
  selectedSampleId: string | null;
  onSampleChange: (id: string | null) => void;
  onViewChange: (view: "overview" | "diagnosis") => void;
  onFilterChange: (filter: VadDiagnosisFilter) => void;
};

type VadRunWorkspaceProps = {
  status: JobStatus | "started";
  jobId: string;
  progress: EvaluationProgress | null;
  request: EvaluationRequest | null;
  events: string[];
  errorMessage?: string;
  connectionWarning?: string;
  onEditConfiguration?: () => void;
};

type VadRunInformationDrawerProps = {
  open: boolean;
  jobId: string;
  request: EvaluationRequest | null;
  result: EvaluationResult | null;
  onClose: () => void;
  onReevaluate: () => void;
};

const FILTER_LABELS: Record<VadDiagnosisFilter, string> = {
  all: "全部",
  miss: "漏检",
  false_alarm: "虚警",
  low_frame_f1: "低 Frame F1",
  oversegmentation: "疑似过度切分",
};

export function VadRunInformationDrawer({
  open,
  jobId,
  request,
  result,
  onClose,
  onReevaluate,
}: VadRunInformationDrawerProps) {
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const drawerRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const previousFocus =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    closeRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
      if (event.key === "Tab" && drawerRef.current) {
        const focusable = Array.from(
          drawerRef.current.querySelectorAll<HTMLElement>(
            "button:not(:disabled), [href], input:not(:disabled), [tabindex]:not([tabindex='-1'])",
          ),
        );
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault();
          last?.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault();
          first?.focus();
        }
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      previousFocus?.focus();
    };
  }, [open, onClose]);

  if (!open) return null;
  const included =
    result?.included_sample_count ??
    result?.sample_count ??
    result?.vad_report?.samples.length;
  return (
    <div
      className="vad-drawer-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <aside
        ref={drawerRef}
        className="vad-run-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="vad-run-drawer-title"
      >
        <div className="vad-run-drawer-heading">
          <h2 id="vad-run-drawer-title">运行信息</h2>
          <button
            ref={closeRef}
            type="button"
            aria-label="关闭运行信息"
            onClick={onClose}
          >
            ×
          </button>
        </div>
        <p className="vad-run-drawer-status">
          <i aria-hidden="true" />
          任务已完成
        </p>
        <dl>
          <DrawerItem label="任务 ID" value={jobId || "-"} mono />
          <DrawerItem label="引擎地址" value={request?.target || "-"} />
          <DrawerItem
            label="数据集"
            value={request ? `${request.dataset_path} / ${request.split}` : "-"}
          />
          <DrawerItem
            label="样本范围"
            value={`${included ?? "-"} 条 · ${request?.limit ? `限制 ${request.limit}` : "不限制"}`}
          />
          <DrawerItem
            label="命中阈值"
            value={formatDecimal(request?.hit_threshold)}
          />
          <DrawerItem
            label="帧长"
            value={formatMilliseconds(request?.mask_frame_seconds)}
          />
          <DrawerItem
            label="处理耗时"
            value={formatSeconds(result?.processing_elapsed_seconds)}
          />
          <DrawerItem
            label="音频总时长"
            value={formatSeconds(result?.audio_duration_seconds)}
          />
        </dl>
        <div className="vad-run-drawer-actions">
          <button
            type="button"
            className="vad-secondary-button"
            onClick={onReevaluate}
          >
            基于此配置重新评估
          </button>
          <p>重新评估会创建新的任务，不覆盖本次结果。</p>
        </div>
      </aside>
    </div>
  );
}

function DrawerItem({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div>
      <dt>{label}</dt>
      <dd className={mono ? "mono" : ""}>{value}</dd>
    </div>
  );
}

export function VadRunWorkspace({
  status,
  jobId,
  progress,
  request,
  events,
  errorMessage,
  connectionWarning,
  onEditConfiguration,
}: VadRunWorkspaceProps) {
  const total = progress?.total ?? 0;
  const processed = progress?.processed ?? 0;
  const percent = total > 0 ? Math.min(100, (processed / total) * 100) : 0;
  const failed = status === "failed";
  const queued = status === "queued" || status === "started";

  return (
    <section className={`vad-run-workspace ${failed ? "is-failed" : ""}`}>
      <div className="vad-run-identity">
        <div>
          <span className="vad-eyebrow">
            {failed ? "运行中断" : queued ? "正在准备" : "正在评估"}
          </span>
          <h2>
            {failed
              ? "评估未完成"
              : queued
                ? "任务已进入队列"
                : `${percent.toFixed(0)}%`}
          </h2>
          <p>
            {failed
              ? errorMessage || "任务执行失败，请检查运行配置后重试。"
              : queued
                ? "服务正在准备数据和推理引擎。"
                : `${processed} / ${total || "-"} 个样本已处理`}
          </p>
        </div>
        <span className={`vad-lifecycle-chip ${failed ? "failed" : "running"}`}>
          <i aria-hidden="true" />
          {failed ? "失败" : queued ? "排队中" : "运行中"}
        </span>
      </div>

      {!failed && !queued ? (
        <div
          className="vad-run-progress"
          role="progressbar"
          aria-label="VAD 评估进度"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(percent)}
        >
          <span style={{ width: `${percent}%` }} />
        </div>
      ) : null}

      <div className="vad-run-grid">
        <section
          className="vad-run-current"
          aria-labelledby="vad-current-sample-title"
        >
          <div className="vad-section-heading">
            <div>
              <h3 id="vad-current-sample-title">
                {failed ? "失败信息" : "当前样本"}
              </h3>
              <p>{failed ? "可返回配置修正后创建新任务" : "事件流实时更新"}</p>
            </div>
          </div>
          <dl className="vad-definition-list">
            <div>
              <dt>样本</dt>
              <dd>{progress?.current_id || progress?.id || "-"}</dd>
            </div>
            <div>
              <dt>已评估</dt>
              <dd>{progress?.evaluated ?? 0}</dd>
            </div>
            <div>
              <dt>任务 ID</dt>
              <dd className="mono">{jobId || "-"}</dd>
            </div>
          </dl>
          {connectionWarning ? (
            <p className="vad-inline-warning">{connectionWarning}</p>
          ) : null}
          {failed && onEditConfiguration ? (
            <button
              type="button"
              className="vad-secondary-button"
              onClick={onEditConfiguration}
            >
              检查运行配置
            </button>
          ) : null}
        </section>

        <section
          className="vad-run-events"
          aria-labelledby="vad-run-events-title"
        >
          <div className="vad-section-heading">
            <div>
              <h3 id="vad-run-events-title">最近事件</h3>
              <p>仅保留当前浏览器会话中的最近记录</p>
            </div>
          </div>
          {events.length ? (
            <ol aria-live="polite">
              {events
                .slice(-8)
                .reverse()
                .map((event, index) => (
                  <li key={`${event}:${index}`}>{event}</li>
                ))}
            </ol>
          ) : (
            <p className="vad-empty-copy">等待首个评估事件</p>
          )}
        </section>
      </div>

      {request ? <VadReadOnlyConfiguration request={request} /> : null}
    </section>
  );
}

export function VadEvaluationReport({
  result,
  request,
  view,
  filter,
  onViewChange,
  onFilterChange,
  search = "",
  selectedSampleId,
  onSampleChange,
}: VadEvaluationReportProps) {
  const diagnostics = useMemo(() => deriveVadDiagnostics(result), [result]);
  const lowF1Ids = useMemo(() => getLowFrameF1Ids(diagnostics), [diagnostics]);
  const filtered = useMemo(
    () =>
      filterDiagnostics(diagnostics, filter, lowF1Ids).filter((item) =>
        item.sample.id.toLowerCase().includes(search.toLowerCase()),
      ),
    [diagnostics, filter, lowF1Ids, search],
  );
  const openDiagnosis = (nextFilter: VadDiagnosisFilter, sampleId?: string) => {
    onFilterChange(nextFilter);
    onViewChange("diagnosis");
    onSampleChange(sampleId ?? null);
  };

  if (view === "overview") {
    return (
      <VadReportOverview
        result={result}
        request={request}
        diagnostics={diagnostics}
        onOpenDiagnosis={openDiagnosis}
      />
    );
  }

  const selected = selectedSampleId
    ? (diagnostics.find((item) => item.sample.id === selectedSampleId) ?? null)
    : null;

  return (
    <VadDiagnosisView
      diagnostics={diagnostics}
      filtered={filtered}
      filter={filter}
      lowF1Ids={lowF1Ids}
      selected={selected}
      request={request}
      onFilterChange={onFilterChange}
      onSelect={(sampleId) => onSampleChange(sampleId)}
      onBack={() => onSampleChange(null)}
    />
  );
}

function VadReportOverview({
  result,
  request,
  diagnostics,
  onOpenDiagnosis,
}: {
  result: EvaluationResult;
  request: EvaluationRequest | null;
  diagnostics: VadSampleDiagnostic[];
  onOpenDiagnosis: (filter: VadDiagnosisFilter, sampleId?: string) => void;
}) {
  const frame = result.frame ?? result;
  const segment = result.segment ?? result;
  const referenceCount = numberValue(
    segment.reference_segment_count ?? result.reference_segment_count,
  );
  const predictionCount = numberValue(
    segment.prediction_segment_count ?? result.prediction_segment_count,
  );
  const missCount = numberValue(segment.segment_miss_count);
  const falseAlarmCount = numberValue(segment.segment_false_alarm_count);
  const recallHits = Math.max(0, referenceCount - missCount);
  const precisionHits = Math.max(0, predictionCount - falseAlarmCount);
  const oversegmentationSamples = diagnostics.filter(
    (item) => item.oversegmentationCount > 0,
  ).length;
  const attention = [...diagnostics]
    .filter(
      (item) =>
        item.missCount || item.falseAlarmCount || item.oversegmentationCount,
    )
    .sort(compareAllDiagnostics)
    .slice(0, 5);

  return (
    <div className="vad-report-overview">
      <div className="vad-section-heading vad-core-heading">
        <div>
          <h2>核心结果</h2>
        </div>
        <p>质量指标 4 · 性能指标 1</p>
      </div>
      <div className="vad-core-metrics">
        <VadCoreMetric
          label="Frame F1"
          value={formatRate(frame.frame_f1 ?? result.frame_f1)}
          detail="帧级综合质量"
          onClick={() => onOpenDiagnosis("low_frame_f1")}
        />
        <VadCoreMetric
          label="Segment F1"
          value={formatRate(segment.segment_f1)}
          detail="综合段级质量"
          onClick={() => onOpenDiagnosis("all")}
        />
        <VadCoreMetric
          label="Segment Recall"
          value={formatRate(segment.segment_recall ?? result.segment_recall)}
          detail={`${recallHits} / ${referenceCount} 命中`}
          warning
          onClick={() => onOpenDiagnosis("miss")}
        />
        <VadCoreMetric
          label="Segment Precision"
          value={formatRate(
            segment.segment_precision ?? result.segment_precision,
          )}
          detail={`${precisionHits} / ${predictionCount} 重叠`}
          onClick={() => onOpenDiagnosis("false_alarm")}
        />
        <VadCoreMetric
          label="Realtime"
          value={formatRealtime(result.realtime_factor)}
          detail={`处理耗时 ${formatSeconds(result.processing_elapsed_seconds)}`}
        />
      </div>

      <div className="vad-analysis-grid">
        <section
          className="vad-analysis-panel"
          aria-labelledby="vad-matching-definition"
        >
          <h2 id="vad-matching-definition">段级匹配口径</h2>
          <p>
            <strong>
              Recall&nbsp; {recallHits} / {referenceCount}
            </strong>{" "}
            · 参考段被预测覆盖 ≥ {formatThreshold(request?.hit_threshold)}{" "}
            才计为命中
          </p>
          <p>
            <strong>
              Precision&nbsp; {precisionHits} / {predictionCount}
            </strong>{" "}
            · 预测段只要与任一参考段有正长度重叠即命中
          </p>
          <small>
            Micro 聚合 · frame {formatMilliseconds(request?.mask_frame_seconds)}{" "}
            · hit_threshold {formatDecimal(request?.hit_threshold)}
          </small>
        </section>
        <section
          className="vad-analysis-panel vad-attention-summary"
          aria-labelledby="vad-attention-summary"
        >
          <h2 id="vad-attention-summary">需要追查</h2>
          <button type="button" onClick={() => onOpenDiagnosis("miss")}>
            <span>漏检参考段</span>
            <strong>{missCount}</strong>
          </button>
          <p>
            <button
              type="button"
              onClick={() => onOpenDiagnosis("false_alarm")}
            >
              虚警预测段 {falseAlarmCount}
            </button>
            <span> · </span>
            <button
              type="button"
              onClick={() => onOpenDiagnosis("oversegmentation")}
            >
              疑似过度切分 {oversegmentationSamples} 个样本
            </button>
          </p>
        </section>
      </div>

      <section
        className="vad-attention-samples"
        aria-labelledby="vad-attention-title"
      >
        <div className="vad-section-heading">
          <div>
            <h2 id="vad-attention-title">需要关注的样本</h2>
          </div>
          <button
            type="button"
            className="vad-link-button"
            onClick={() => onOpenDiagnosis("all")}
          >
            查看全部 {diagnostics.length} 个样本 →
          </button>
        </div>
        {attention.length ? (
          <VadSampleTable
            diagnostics={attention}
            onSelect={(sampleId) => onOpenDiagnosis("all", sampleId)}
          />
        ) : (
          <p className="vad-empty-copy">当前结果中没有需要追查的段级错误。</p>
        )}
      </section>
    </div>
  );
}

function VadDiagnosisView({
  diagnostics,
  filtered,
  filter,
  lowF1Ids,
  selected,
  request,
  onFilterChange,
  onSelect,
  onBack,
}: {
  diagnostics: VadSampleDiagnostic[];
  filtered: VadSampleDiagnostic[];
  filter: VadDiagnosisFilter;
  lowF1Ids: Set<string>;
  selected: VadSampleDiagnostic | null;
  request: EvaluationRequest | null;
  onFilterChange: (filter: VadDiagnosisFilter) => void;
  onSelect: (sampleId: string) => void;
  onBack: () => void;
}) {
  const counts = {
    all: diagnostics.length,
    miss: diagnostics.filter((item) => item.missCount > 0).length,
    false_alarm: diagnostics.filter((item) => item.falseAlarmCount > 0).length,
    low_frame_f1: diagnostics.filter((item) => lowF1Ids.has(item.sample.id))
      .length,
    oversegmentation: diagnostics.filter(
      (item) => item.oversegmentationCount > 0,
    ).length,
  };

  if (selected) {
    return (
      <div className="vad-diagnosis-detail-layout">
        <aside
          className="vad-sample-context"
          aria-label={`${FILTER_LABELS[filter]}样本`}
        >
          <div className="vad-sample-context-heading">
            <h2>{FILTER_LABELS[filter]}样本</h2>
            <p>
              {filtered.length} 个 · {filterSortDescription(filter)}
            </p>
          </div>
          {filtered.map((item) => (
            <button
              type="button"
              className={item.sample.id === selected.sample.id ? "active" : ""}
              key={item.sample.id}
              onClick={() => onSelect(item.sample.id)}
            >
              <strong>{item.sample.id}</strong>
              <span>
                Frame F1 {formatRate(item.frameF1)} · 漏检{" "}
                {formatSeconds(item.missDuration)}
              </span>
            </button>
          ))}
        </aside>
        <VadSampleDetail diagnostic={selected} onBack={onBack} />
      </div>
    );
  }

  return (
    <section className="vad-diagnosis" aria-labelledby="vad-diagnosis-heading">
      <div className="vad-section-heading vad-diagnosis-heading">
        <div>
          <h2 id="vad-diagnosis-heading">样本诊断</h2>
          <p>
            {filter === "all"
              ? "查看全部样本"
              : `由 ${FILTER_LABELS[filter]} 指标下钻`}{" "}
            · {filterSortDescription(filter)}
          </p>
        </div>
        {filter !== "all" ? (
          <button
            type="button"
            className="vad-link-button"
            onClick={() => onFilterChange("all")}
          >
            清除筛选
          </button>
        ) : null}
      </div>
      <div className="vad-filter-list" role="group" aria-label="样本诊断筛选">
        {(Object.keys(FILTER_LABELS) as VadDiagnosisFilter[]).map((item) => (
          <button
            type="button"
            aria-pressed={filter === item}
            className={filter === item ? "active" : ""}
            key={item}
            onClick={() => onFilterChange(item)}
          >
            {FILTER_LABELS[item]} {counts[item]}
          </button>
        ))}
      </div>
      <div className="vad-active-rule">
        <strong>筛选规则</strong>
        <span>{filterRuleDescription(filter, request, filtered)}</span>
      </div>
      {filtered.length ? (
        <VadSampleTable diagnostics={filtered} onSelect={onSelect} />
      ) : (
        <p className="vad-empty-copy">没有符合当前规则的样本。</p>
      )}
    </section>
  );
}

function VadSampleTable({
  diagnostics,
  onSelect,
}: {
  diagnostics: VadSampleDiagnostic[];
  onSelect: (sampleId: string) => void;
}) {
  return (
    <div className="vad-sample-table" aria-label="VAD 样本诊断">
      <div className="vad-sample-table-row header" aria-hidden="true">
        <span>样本</span>
        <span>Frame F1</span>
        <span>漏检时长</span>
        <span>虚警时长</span>
        <span>段级错误</span>
        <span>操作</span>
      </div>
      {diagnostics.map((item) => (
        <button
          type="button"
          className="vad-sample-table-row"
          key={item.sample.id}
          onClick={() => onSelect(item.sample.id)}
        >
          <strong>{item.sample.id}</strong>
          <span>{formatRate(item.frameF1)}</span>
          <span>{formatSeconds(item.missDuration)}</span>
          <span>{formatSeconds(item.falseAlarmDuration)}</span>
          <span
            className={
              item.missCount ||
              item.falseAlarmCount ||
              item.oversegmentationCount
                ? "vad-error-label"
                : "vad-normal-label"
            }
          >
            {diagnosticErrorLabel(item)}
          </span>
          <span className="vad-row-action">查看 →</span>
        </button>
      ))}
    </div>
  );
}

function VadSampleDetail({
  diagnostic,
  onBack,
}: {
  diagnostic: VadSampleDiagnostic;
  onBack: () => void;
}) {
  const { sample } = diagnostic;
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const activeEndRef = useRef<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const errorRegions = sample.regions.filter((region) => {
    const label = normalizeVadLabel(region.label);
    return label === "miss" || label === "false_alarm";
  });

  useEffect(() => {
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
    }
    activeEndRef.current = null;
    setPlaying(false);
  }, [sample.id]);

  const playRegion = (region?: VadReportRegion) => {
    const audio = audioRef.current;
    if (!audio || !sample.audio_url) return;
    if (!audio.paused && !region) {
      audio.pause();
      return;
    }
    const target = region ?? errorRegions[0];
    if (!target) return;
    audio.currentTime = target.start;
    activeEndRef.current = target.end;
    void audio.play().catch(() => setPlaying(false));
  };

  return (
    <section
      className="vad-sample-detail"
      aria-labelledby="vad-sample-detail-title"
    >
      <div className="vad-sample-detail-heading">
        <button type="button" className="vad-detail-back" onClick={onBack}>
          <ChevronLeft size={16} />
          返回样本列表
        </button>
        <div>
          <h2 id="vad-sample-detail-title">{sample.id}</h2>
          <p>
            {formatSeconds(sample.duration_seconds)} · Frame F1{" "}
            {formatRate(diagnostic.frameF1)}
          </p>
        </div>
        <button
          type="button"
          className="vad-play-error"
          disabled={!sample.audio_url || !errorRegions.length}
          onClick={() => playRegion()}
        >
          {playing ? <Pause size={15} /> : <Play size={15} />} 播放错误区间
        </button>
      </div>
      <audio
        ref={audioRef}
        src={sample.audio_url}
        preload="metadata"
        controls
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onTimeUpdate={(event) => {
          const end = activeEndRef.current;
          if (end !== null && event.currentTarget.currentTime >= end) {
            event.currentTarget.pause();
            activeEndRef.current = null;
          }
        }}
      />
      <div className="vad-sample-summary">
        <VadMiniMetric
          label="漏检时长"
          value={formatSeconds(diagnostic.missDuration)}
          tone="warning"
        />
        <VadMiniMetric
          label="虚警时长"
          value={formatSeconds(diagnostic.falseAlarmDuration)}
          tone="danger"
        />
        <VadMiniMetric
          label="漏检段"
          value={String(diagnostic.missCount)}
          tone="warning"
        />
        <VadMiniMetric
          label="预测段"
          value={String(sample.prediction_segments.length)}
        />
      </div>
      <div className="vad-timeline-legend" aria-label="时间轴图例">
        <span>
          <i className="reference" />
          Reference
        </span>
        <span>
          <i className="prediction" />
          Prediction
        </span>
        <span>
          <i className="miss" />
          漏检
        </span>
        <span>
          <i className="false-alarm" />
          虚警
        </span>
      </div>
      <div className="vad-detail-timeline">
        <Waveform src={sample.audio_url} />
        <TimelineTrack
          label="Reference"
          duration={sample.duration_seconds}
          segments={sample.reference_segments}
        />
        <TimelineTrack
          label="Prediction"
          duration={sample.duration_seconds}
          segments={sample.prediction_segments}
        />
        <div className="vad-detail-track error-track">
          <span>Errors</span>
          <div>
            {errorRegions.map((region) => (
              <button
                type="button"
                className={`error-region ${normalizeVadLabel(region.label)}`}
                key={`${region.start_frame}:${region.end_frame}:${region.label}`}
                style={positionStyle(
                  region.start,
                  region.duration,
                  sample.duration_seconds,
                )}
                title={`${regionLabel(region.label)} ${formatSeconds(region.start)} – ${formatSeconds(region.end)}`}
                aria-label={`播放${regionLabel(region.label)}区间 ${formatSeconds(region.start)} 到 ${formatSeconds(region.end)}`}
                onClick={() => playRegion(region)}
              />
            ))}
          </div>
        </div>
        <div className="vad-detail-ruler">
          <span>Time</span>
          <div>
            {buildTicks(sample.duration_seconds).map((tick) => (
              <i
                key={tick}
                style={{ left: `${percent(tick, sample.duration_seconds)}%` }}
              >
                {formatCompactSeconds(tick)}
              </i>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function TimelineTrack({
  label,
  duration,
  segments,
}: {
  label: string;
  duration: number;
  segments: VadReportSegment[];
}) {
  return (
    <div className="vad-detail-track">
      <span>{label}</span>
      <div>
        {segments.map((segment) => (
          <i
            key={`${label}:${segment.start_frame}:${segment.end_frame}`}
            className={label === "Reference" ? "reference" : "prediction"}
            style={positionStyle(segment.start, segment.duration, duration)}
            title={`${formatSeconds(segment.start)} – ${formatSeconds(segment.end)}`}
          />
        ))}
      </div>
    </div>
  );
}

function Waveform({ src }: { src?: string }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !src) return;
    const controller = new AbortController();
    let audioContext: AudioContext | null = null;
    fetch(src, { signal: controller.signal })
      .then((response) => response.arrayBuffer())
      .then(async (buffer) => {
        audioContext = new AudioContext();
        const decoded = await audioContext.decodeAudioData(buffer);
        if (controller.signal.aborted || !canvasRef.current) return;
        drawWaveform(canvasRef.current, decoded.getChannelData(0));
      })
      .catch(() => undefined);
    return () => {
      controller.abort();
      void audioContext?.close();
    };
  }, [src]);

  return (
    <div className="vad-waveform">
      <span>Waveform</span>
      <canvas ref={canvasRef} aria-hidden="true" />
    </div>
  );
}

function drawWaveform(canvas: HTMLCanvasElement, samples: Float32Array) {
  const width = Math.max(640, canvas.clientWidth * window.devicePixelRatio);
  const height = Math.max(70, canvas.clientHeight * window.devicePixelRatio);
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) return;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#a1a1aa";
  const bucket = Math.max(1, Math.floor(samples.length / width));
  for (let x = 0; x < width; x += 3) {
    let peak = 0;
    const start = x * bucket;
    const end = Math.min(samples.length, start + bucket * 3);
    for (let index = start; index < end; index += 1)
      peak = Math.max(peak, Math.abs(samples[index]));
    const lineHeight = Math.max(1, peak * height * 0.86);
    context.fillRect(x, (height - lineHeight) / 2, 2, lineHeight);
  }
}

function VadCoreMetric({
  label,
  value,
  detail,
  warning = false,
  onClick,
}: {
  label: string;
  value: string;
  detail: string;
  warning?: boolean;
  onClick?: () => void;
}) {
  const Tag = onClick ? "button" : "div";
  return (
    <Tag
      type={onClick ? "button" : undefined}
      className={`vad-core-metric ${warning ? "warning" : ""} ${onClick ? "actionable" : ""}`}
      onClick={onClick}
    >
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </Tag>
  );
}

function VadMiniMetric({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: string;
  tone?: "default" | "warning" | "danger";
}) {
  return (
    <div className={`vad-mini-metric ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function VadReadOnlyConfiguration({ request }: { request: EvaluationRequest }) {
  return (
    <section
      className="vad-readonly-config"
      aria-labelledby="vad-run-config-title"
    >
      <div className="vad-section-heading">
        <div>
          <h3 id="vad-run-config-title">运行配置</h3>
          <p>任务运行期间配置保持只读</p>
        </div>
      </div>
      <dl>
        <div>
          <dt>引擎</dt>
          <dd>{request.target}</dd>
        </div>
        <div>
          <dt>数据集</dt>
          <dd>
            {request.dataset_path} / {request.split}
          </dd>
        </div>
        <div>
          <dt>命中阈值</dt>
          <dd>{formatDecimal(request.hit_threshold)}</dd>
        </div>
        <div>
          <dt>帧长</dt>
          <dd>{formatMilliseconds(request.mask_frame_seconds)}</dd>
        </div>
      </dl>
    </section>
  );
}

export function deriveVadEvent(progress: EvaluationProgress): string {
  const id = progress.current_id || progress.id;
  if (progress.status === "completed") return "评估计算完成，正在生成报告";
  if (id)
    return `${id} · 已处理 ${progress.processed ?? 0} / ${progress.total ?? "-"}`;
  if (progress.status === "queued" || progress.status === "started")
    return "任务已进入队列";
  return `进度更新 · 已处理 ${progress.processed ?? 0} / ${progress.total ?? "-"}`;
}

function deriveVadDiagnostics(result: EvaluationResult): VadSampleDiagnostic[] {
  return (result.vad_report?.samples ?? []).map((sample) => {
    const metrics = sample.metrics;
    const missRegions = sample.regions.filter(
      (region) => normalizeVadLabel(region.label) === "miss",
    );
    const falseAlarmRegions = sample.regions.filter(
      (region) => normalizeVadLabel(region.label) === "false_alarm",
    );
    return {
      sample,
      frameF1: optionalNumber(metrics?.frame?.frame_f1 ?? metrics?.frame_f1),
      missDuration: sumDuration(missRegions),
      falseAlarmDuration: sumDuration(falseAlarmRegions),
      missCount:
        numberValue(metrics?.segment?.segment_miss_count) ||
        sample.reference_segments.filter(
          (segment) => normalizeVadLabel(segment.status) === "miss",
        ).length,
      falseAlarmCount:
        numberValue(metrics?.segment?.segment_false_alarm_count) ||
        sample.prediction_segments.filter(
          (segment) => normalizeVadLabel(segment.status) === "false_alarm",
        ).length,
      oversegmentationCount: countOversegmentation(
        sample.reference_segments,
        sample.prediction_segments,
      ),
    };
  });
}

function getLowFrameF1Ids(diagnostics: VadSampleDiagnostic[]) {
  const candidates = diagnostics
    .filter((item) => item.frameF1 !== null)
    .sort((a, b) => (a.frameF1 ?? 1) - (b.frameF1 ?? 1));
  return new Set(
    candidates
      .slice(0, Math.max(1, Math.ceil(candidates.length * 0.25)))
      .map((item) => item.sample.id),
  );
}

function filterDiagnostics(
  diagnostics: VadSampleDiagnostic[],
  filter: VadDiagnosisFilter,
  lowF1Ids: Set<string>,
) {
  const filtered = diagnostics.filter((item) => {
    if (filter === "miss") return item.missCount > 0;
    if (filter === "false_alarm") return item.falseAlarmCount > 0;
    if (filter === "low_frame_f1") return lowF1Ids.has(item.sample.id);
    if (filter === "oversegmentation") return item.oversegmentationCount > 0;
    return true;
  });
  return filtered.sort((a, b) => {
    if (filter === "miss")
      return b.missDuration - a.missDuration || compareIndex(a, b);
    if (filter === "false_alarm")
      return b.falseAlarmDuration - a.falseAlarmDuration || compareIndex(a, b);
    if (filter === "low_frame_f1")
      return (a.frameF1 ?? 1) - (b.frameF1 ?? 1) || compareIndex(a, b);
    if (filter === "oversegmentation")
      return (
        b.oversegmentationCount - a.oversegmentationCount ||
        compareAllDiagnostics(a, b)
      );
    return compareAllDiagnostics(a, b);
  });
}

function compareAllDiagnostics(a: VadSampleDiagnostic, b: VadSampleDiagnostic) {
  return (
    b.missDuration +
      b.falseAlarmDuration -
      (a.missDuration + a.falseAlarmDuration) ||
    (a.frameF1 ?? 1) - (b.frameF1 ?? 1) ||
    compareIndex(a, b)
  );
}

function compareIndex(a: VadSampleDiagnostic, b: VadSampleDiagnostic) {
  return (
    (a.sample.index ?? Number.MAX_SAFE_INTEGER) -
      (b.sample.index ?? Number.MAX_SAFE_INTEGER) ||
    a.sample.id.localeCompare(b.sample.id)
  );
}

function countOversegmentation(
  reference: VadReportSegment[],
  predictions: VadReportSegment[],
) {
  return reference.reduce((total, item) => {
    const overlaps = predictions.filter(
      (prediction) =>
        Math.min(item.end, prediction.end) >
        Math.max(item.start, prediction.start),
    ).length;
    return total + Math.max(0, overlaps - 1);
  }, 0);
}

function diagnosticErrorLabel(item: VadSampleDiagnostic) {
  const labels: string[] = [];
  if (item.missCount) labels.push(`漏检 ${item.missCount} 段`);
  if (item.falseAlarmCount) labels.push(`虚警 ${item.falseAlarmCount} 段`);
  if (item.oversegmentationCount) labels.push("疑似过度切分");
  return labels.join(" · ") || "无段级错误";
}

function filterSortDescription(filter: VadDiagnosisFilter) {
  if (filter === "miss") return "按漏检时长降序";
  if (filter === "false_alarm") return "按虚警时长降序";
  if (filter === "low_frame_f1") return "按 Frame F1 升序";
  if (filter === "oversegmentation") return "按切分数量降序";
  return "按错误时长降序";
}

function filterRuleDescription(
  filter: VadDiagnosisFilter,
  request: EvaluationRequest | null,
  filtered: VadSampleDiagnostic[],
) {
  if (filter === "miss")
    return `参考段覆盖 < ${formatThreshold(request?.hit_threshold)} · ${filtered.length} 个样本 · ${filtered.reduce((sum, item) => sum + item.missCount, 0)} 个漏检段`;
  if (filter === "false_alarm")
    return `预测段与参考段无正长度重叠 · ${filtered.length} 个样本 · ${filtered.reduce((sum, item) => sum + item.falseAlarmCount, 0)} 个虚警段`;
  if (filter === "low_frame_f1")
    return `Frame F1 最低四分位 · ${filtered.length} 个样本`;
  if (filter === "oversegmentation")
    return `同一参考段与至少两个预测段重叠 · ${filtered.length} 个样本`;
  return `全部 ${filtered.length} 个样本 · 优先展示错误时长较长的样本`;
}

function sumDuration(regions: VadReportRegion[]) {
  return regions.reduce((sum, region) => sum + numberValue(region.duration), 0);
}
function numberValue(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}
function optionalNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
function normalizeVadLabel(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[-\s]+/g, "_");
}
function regionLabel(value: string) {
  return normalizeVadLabel(value) === "miss" ? "漏检" : "虚警";
}
function percent(value: number, duration: number) {
  return duration > 0
    ? Math.max(0, Math.min(100, (value / duration) * 100))
    : 0;
}
function positionStyle(start: number, duration: number, total: number) {
  return {
    left: `${percent(start, total)}%`,
    width: `${percent(duration, total)}%`,
  };
}
function formatRate(value: unknown) {
  const number = optionalNumber(value);
  return number === null ? "-" : `${(number * 100).toFixed(2)}%`;
}
function formatSeconds(value: unknown) {
  const number = optionalNumber(value);
  return number === null ? "-" : `${number.toFixed(2)} s`;
}
function formatCompactSeconds(value: number) {
  return value >= 10
    ? `${value.toFixed(1)}s`
    : `${value.toFixed(2).replace(/0+$/, "").replace(/\.$/, "")}s`;
}
function formatRealtime(value: unknown) {
  const number = optionalNumber(value);
  return number === null ? "-" : `${number.toFixed(2)}×`;
}
function formatDecimal(value: unknown) {
  const number = optionalNumber(value);
  return number === null ? "-" : number.toFixed(2);
}
function formatThreshold(value: unknown) {
  const number = optionalNumber(value);
  return number === null ? "-" : `${(number * 100).toFixed(0)}%`;
}
function formatMilliseconds(value: unknown) {
  const number = optionalNumber(value);
  return number === null ? "-" : `${Math.round(number * 1000)} ms`;
}
function buildTicks(duration: number) {
  if (duration <= 0) return [0];
  return Array.from({ length: 5 }, (_, index) => (duration * index) / 4);
}
