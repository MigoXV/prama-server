import { Pause, Play } from "lucide-react";
import katex from "katex";
import "katex/dist/katex.min.css";
import type { ChangeEvent, MouseEvent, ReactNode } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Field, MetricTile, StatusChip } from "./ui";
import type {
  EvaluationResult,
  JobStatus,
  KeywordAudioReportSample,
  KeywordReportSample,
  DenoiseReportSample,
  LidReportSample,
  SqaScore,
  SqaSummary,
  WerReport,
  WerSummary,
  WerToken,
  WerUtterance,
} from "../types";
import { STATUS_LABELS } from "../features/evaluation/model";
type AlignmentMetric = "wer" | "cer";
type ReportSortMode =
  "index-asc" | "index-desc" | "wer-desc" | "wer-asc" | "cer-desc" | "cer-asc";
type MarkdownBlock =
  | { type: "heading"; level: 1 | 2 | 3; text: string; id: string }
  | { type: "paragraph"; text: string }
  | { type: "list"; items: string[] }
  | { type: "code"; language: string; text: string }
  | { type: "formula"; text: string }
  | { type: "table"; headers: string[]; rows: string[][] };

const KEYWORD_REPORT_INITIAL_VISIBLE = 100;
const KEYWORD_REPORT_LOAD_STEP = 100;
export function StatusPill({
  status,
}: {
  status: JobStatus | "idle" | "started";
}) {
  return (
    <StatusChip
      className={`status-pill status-${status}`}
      role="status"
      aria-live="polite"
    >
      <span className="status-dot" aria-hidden="true" />
      <span>{STATUS_LABELS[status]}</span>
    </StatusChip>
  );
}

export function TextField({
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

function TextBlock({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="text-block">
      <span>{label}</span>
      <p>{value}</p>
    </div>
  );
}

type KeywordHighlightTone = "hit" | "false_alarm";

interface KeywordHighlight {
  keyword: string;
  tone: KeywordHighlightTone;
}

function KeywordMatchTextBlock({
  matchText,
  highlights,
}: {
  matchText: string;
  highlights: KeywordHighlight[];
}) {
  return (
    <TextBlock
      label="正则化后的推理结果"
      value={
        matchText ? (
          <HighlightedKeywordText text={matchText} highlights={highlights} />
        ) : (
          "-"
        )
      }
    />
  );
}

function keywordHighlightsFromSample(
  sample: KeywordReportSample,
): KeywordHighlight[] {
  if (!sample.predicted_hit) {
    return [];
  }
  return [
    {
      keyword: sample.keyword,
      tone: sample.expected_hit ? "hit" : "false_alarm",
    },
  ];
}

function keywordHighlightsFromAudioSample(
  sample: KeywordAudioReportSample,
): KeywordHighlight[] {
  return sample.keywords
    .filter((keyword) => keyword.predicted_hit)
    .map((keyword) => ({
      keyword: keyword.keyword,
      tone: keyword.expected_hit ? "hit" : "false_alarm",
    }));
}

function HighlightedKeywordText({
  text,
  highlights,
}: {
  text: string;
  highlights: KeywordHighlight[];
}) {
  const ranges = keywordHighlightRanges(text, highlights);
  if (!ranges.length) {
    return <>{text}</>;
  }

  const parts: ReactNode[] = [];
  let cursor = 0;
  ranges.forEach((range, index) => {
    if (range.start > cursor) {
      parts.push(text.slice(cursor, range.start));
    }
    parts.push(
      <mark
        className={`keyword-match-highlight ${range.tone}`}
        key={`${range.start}-${range.end}-${index}`}
      >
        {text.slice(range.start, range.end)}
      </mark>,
    );
    cursor = range.end;
  });
  if (cursor < text.length) {
    parts.push(text.slice(cursor));
  }
  return <>{parts}</>;
}

function keywordHighlightRanges(text: string, highlights: KeywordHighlight[]) {
  const ranges = highlights
    .flatMap((highlight) =>
      keywordRangesForHighlight(text, highlight).map((range) => ({
        ...range,
        tone: highlight.tone,
      })),
    )
    .sort((left, right) => left.start - right.start || right.end - left.end);

  const merged: Array<{
    start: number;
    end: number;
    tone: KeywordHighlightTone;
  }> = [];
  ranges.forEach((range) => {
    const previous = merged[merged.length - 1];
    if (!previous || range.start >= previous.end) {
      merged.push(range);
      return;
    }
    if (range.end > previous.end && range.tone === previous.tone) {
      previous.end = range.end;
    }
  });
  return merged;
}

function keywordRangesForHighlight(text: string, highlight: KeywordHighlight) {
  const normalizedKeyword = normalizeKeywordText(highlight.keyword);
  if (!normalizedKeyword) {
    return [];
  }
  if (/^[a-z0-9]+(?: [a-z0-9]+)*$/.test(normalizedKeyword)) {
    const tokens = Array.from(text.matchAll(/\S+/g));
    const keywordTokens = normalizedKeyword.split(" ");
    const ranges: Array<{ start: number; end: number }> = [];
    for (
      let index = 0;
      index <= tokens.length - keywordTokens.length;
      index += 1
    ) {
      const tokenSlice = tokens.slice(index, index + keywordTokens.length);
      if (
        tokenSlice.every(
          (token, tokenIndex) => token[0] === keywordTokens[tokenIndex],
        )
      ) {
        const firstToken = tokenSlice[0];
        const lastToken = tokenSlice[tokenSlice.length - 1];
        ranges.push({
          start: firstToken.index ?? 0,
          end: (lastToken.index ?? 0) + lastToken[0].length,
        });
      }
    }
    return ranges;
  }

  const ranges: Array<{ start: number; end: number }> = [];
  let start = text.indexOf(normalizedKeyword);
  while (start >= 0) {
    ranges.push({ start, end: start + normalizedKeyword.length });
    start = text.indexOf(normalizedKeyword, start + normalizedKeyword.length);
  }
  return ranges;
}

function normalizeKeywordText(text: string) {
  return text.toLowerCase().replace(/\p{P}/gu, " ").trim().replace(/\s+/g, " ");
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
  const progress =
    duration > 0 ? Math.min((currentTime / duration) * 100, 100) : 0;

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
      void audio.play().catch(() => setPlaying(false));
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
        onTimeUpdate={(event) =>
          setCurrentTime(event.currentTarget.currentTime)
        }
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

export function PerformanceMetrics({
  result,
}: {
  result: EvaluationResult | null;
}) {
  if (!result) {
    return null;
  }

  return (
    <div className="metric-strip performance-metrics">
      <Metric
        label="音频时长"
        value={formatSeconds(result.audio_duration_seconds)}
      />
      <Metric
        label="处理耗时"
        value={formatSeconds(result.processing_elapsed_seconds)}
      />
      <Metric
        label="倍时"
        value={formatRealtimeFactor(result.realtime_factor)}
      />
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
    {
      label: "处理耗时",
      value: formatSeconds(result?.processing_elapsed_seconds),
    },
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

export function SqaSummaryMetrics({ summary }: { summary?: SqaSummary[] }) {
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
        const failed =
          item.score === null || item.score === undefined || item.error;
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

export function MarkdownDocument({ markdown }: { markdown: string }) {
  const blocks = useMemo(() => parseMarkdown(markdown), [markdown]);
  const headings = blocks.filter(
    (block): block is Extract<MarkdownBlock, { type: "heading" }> =>
      block.type === "heading" && block.level === 2 && block.text !== "目录",
  );
  const contentBlocks = blocks.filter((block, index) => {
    if (block.type === "heading" && block.level === 1) {
      return false;
    }
    if (block.type === "heading" && block.text === "目录") {
      return false;
    }
    const previous = blocks[index - 1];
    return !(
      block.type === "list" &&
      previous?.type === "heading" &&
      previous.text === "目录"
    );
  });
  return (
    <div className="help-document-layout">
      {headings.length ? (
        <details className="markdown-toc" open>
          <summary>本文目录</summary>
          <nav aria-label="帮助文档目录">
            {headings.map((heading) => (
              <a href={`#${heading.id}`} key={heading.id}>
                {heading.text}
              </a>
            ))}
          </nav>
        </details>
      ) : null}
      <article className="markdown-document">
        {contentBlocks.map((block, index) => {
          if (block.type === "heading") {
            const HeadingTag = `h${block.level}` as "h1" | "h2" | "h3";
            return (
              <HeadingTag id={block.id} key={index}>
                {renderInlineMarkdown(block.text)}
              </HeadingTag>
            );
          }
          if (block.type === "formula") {
            return <LatexFormula key={index} text={block.text} displayMode />;
          }
          if (block.type === "code") {
            return (
              <pre key={index} className="markdown-code-block">
                {block.language ? <span>{block.language}</span> : null}
                <code>{block.text}</code>
              </pre>
            );
          }
          if (block.type === "list") {
            return (
              <ul key={index}>
                {block.items.map((item, itemIndex) => (
                  <li key={itemIndex}>{renderInlineMarkdown(item)}</li>
                ))}
              </ul>
            );
          }
          if (block.type === "table") {
            return (
              <div className="table-wrap markdown-table-wrap" key={index}>
                <table>
                  <thead>
                    <tr>
                      {block.headers.map((header, headerIndex) => (
                        <th key={headerIndex}>
                          {renderInlineMarkdown(header)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {block.rows.map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        {block.headers.map((_, cellIndex) => (
                          <td key={cellIndex}>
                            {renderInlineMarkdown(row[cellIndex] ?? "")}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          }
          return <p key={index}>{renderInlineMarkdown(block.text)}</p>;
        })}
      </article>
    </div>
  );
}

function LatexFormula({
  text,
  displayMode = false,
}: {
  text: string;
  displayMode?: boolean;
}) {
  const html = useMemo(
    () =>
      katex.renderToString(text, {
        displayMode,
        throwOnError: false,
        strict: false,
        trust: false,
      }),
    [displayMode, text],
  );

  if (displayMode) {
    return (
      <div
        className="markdown-formula"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  }
  return (
    <span
      className="markdown-inline-formula"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

export function AsrOverviewMetrics({
  result,
}: {
  result: EvaluationResult | null;
}) {
  if (!result) {
    return null;
  }
  const wordAccuracy =
    result.word_accuracy ??
    result.accuracy ??
    result.wer_report?.summary?.accuracy;
  const characterAccuracy =
    result.character_accuracy ?? result.cer_report?.summary?.accuracy;

  return (
    <div className="panel asr-overview-panel">
      <div className="metric-strip asr-overview-metrics">
        <Metric label="词正确率" value={formatPercentScale(wordAccuracy)} />
        <Metric
          label="字正确率"
          value={formatPercentScale(characterAccuracy)}
        />
        <Metric label="WER" value={formatPercentScale(result.wer)} />
        <Metric label="CER" value={formatPercentScale(result.cer)} />
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

export function LidOverviewMetrics({
  result,
}: {
  result: EvaluationResult | null;
}) {
  if (!result) {
    return null;
  }

  return (
    <div className="metric-strip lid-overview-metrics">
      <Metric
        label="已知语种准确率"
        value={formatRate(result.known_accuracy ?? result.accuracy)}
      />
      <Metric
        label="宏平均精确率"
        value={formatRate(result.macro_precision ?? result.precision)}
      />
      <Metric
        label="宏平均召回率"
        value={formatRate(result.macro_recall ?? result.recall)}
      />
      <Metric
        label="未知误接收"
        value={formatNumber(result.unknown_false_accept_count)}
      />
      <Metric
        label="已知被拒识"
        value={formatNumber(result.known_reject_count)}
      />
    </div>
  );
}

export function KeywordOverviewMetrics({
  result,
}: {
  result: EvaluationResult | null;
}) {
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
        <Metric
          label="False Alarm"
          value={formatNumber(result.false_alarm_count)}
        />
      </div>
      <SampleCountStrip
        result={result}
        fallbackCount={result.keyword_report?.samples.length ?? 0}
      />
    </div>
  );
}

export function DenoiseOverviewMetrics({
  result,
}: {
  result: EvaluationResult | null;
}) {
  if (!result) {
    return null;
  }

  return (
    <div className="panel denoise-overview-panel">
      <div className="metric-strip denoise-overview-metrics">
        <Metric
          label="SNR Δ"
          value={formatSignedScore(result.mean_snr_delta)}
        />
        <Metric
          label="MOS Δ"
          value={formatSignedScore(result.mean_mos_delta)}
        />
        <Metric
          label="SNR Samples"
          value={formatNumber(result.scored_snr_sample_count)}
        />
        <Metric
          label="MOS Samples"
          value={formatNumber(result.scored_mos_sample_count)}
        />
        <Metric
          label="Failed"
          value={formatNumber(result.failed_sample_count)}
        />
      </div>
      <SampleCountStrip
        result={result}
        fallbackCount={result.denoise_report?.samples.length ?? 0}
      />
    </div>
  );
}

export function AsrAlignmentReportPanel({
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
  const sampleCount =
    werReport?.utterances.length ?? cerReport?.utterances.length ?? 0;
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
              accuracy={result?.word_accuracy}
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
          <div
            className="alignment-metric-tabs"
            role="tablist"
            aria-label="对齐指标"
          >
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
                  <WerAlignmentRows
                    tokens={utterance.tokens}
                    wrap={wrapAlignment}
                  />
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

export function LidReportPanel({
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
              label="已知语种准确率"
              value={formatRate(result.known_accuracy ?? result.accuracy)}
            />
            <Metric
              label="宏平均精确率"
              value={formatRate(result.macro_precision ?? result.precision)}
            />
            <Metric
              label="宏平均召回率"
              value={formatRate(result.macro_recall ?? result.recall)}
            />
            <Metric
              label="未知误接收"
              value={formatNumber(result.unknown_false_accept_count)}
            />
            <Metric
              label="已知被拒识"
              value={formatNumber(result.known_reject_count)}
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

export function KeywordReportPanel({
  result,
  viewMode,
  onViewModeChange,
}: {
  result: EvaluationResult | null;
  viewMode: "keyword" | "audio";
  onViewModeChange: (mode: "keyword" | "audio") => void;
}) {
  const samples = result?.keyword_report?.samples ?? [];
  const audioSamples = result?.keyword_audio_report?.samples ?? [];

  return (
    <div className="panel report-panel keyword-report-panel compact-report-panel">
      <div className="panel-heading compact-heading">
        <div>
          <h2>关键词报告</h2>
          <span>
            {formatNumber(result?.sample_count)} 个关键词 /{" "}
            {formatNumber(result?.audio_sample_count ?? audioSamples.length)}{" "}
            条语音
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
          {audioSamples.length ? (
            <div
              className="keyword-view-tabs"
              role="tablist"
              aria-label="关键词报告视图"
            >
              <button
                type="button"
                className={viewMode === "keyword" ? "active" : ""}
                onClick={() => onViewModeChange("keyword")}
              >
                按关键词
              </button>
              <button
                type="button"
                className={viewMode === "audio" ? "active" : ""}
                onClick={() => onViewModeChange("audio")}
              >
                按语音
              </button>
            </div>
          ) : null}
          {viewMode === "audio" && audioSamples.length ? (
            <KeywordAudioSampleList samples={audioSamples} />
          ) : (
            <KeywordSampleList samples={samples} />
          )}
        </>
      ) : (
        <div className="empty-state">评估完成后生成关键词报告</div>
      )}
    </div>
  );
}

function KeywordAudioSampleList({
  samples,
}: {
  samples: KeywordAudioReportSample[];
}) {
  const [visibleCount, setVisibleCount] = useState(
    KEYWORD_REPORT_INITIAL_VISIBLE,
  );
  useEffect(() => {
    setVisibleCount(KEYWORD_REPORT_INITIAL_VISIBLE);
  }, [samples]);

  if (!samples.length) {
    return <div className="empty-state">评估完成后生成语音聚合结果</div>;
  }
  const visibleSamples = samples.slice(0, visibleCount);

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
            className="keyword-sample keyword-audio-sample"
            key={sample.id}
          >
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
                  key={keyword.id}
                >
                  <span title={keyword.id}>{keyword.keyword}</span>
                  <small>
                    {keyword.expected_hit ? "Expected Hit" : "Expected No Hit"}{" "}
                    /{" "}
                    {keyword.predicted_hit
                      ? "Predicted Hit"
                      : "Predicted No Hit"}
                  </small>
                </div>
              ))}
            </div>
            <KeywordMatchTextBlock
              matchText={sample.match_text}
              highlights={keywordHighlightsFromAudioSample(sample)}
            />
          </section>
        ))}
        {visibleCount < samples.length ? (
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
  const [visibleCount, setVisibleCount] = useState(
    KEYWORD_REPORT_INITIAL_VISIBLE,
  );
  useEffect(() => {
    setVisibleCount(KEYWORD_REPORT_INITIAL_VISIBLE);
  }, [samples]);

  if (!samples.length) {
    return <div className="empty-state">评估完成后生成关键词结果</div>;
  }
  const visibleSamples = samples.slice(0, visibleCount);

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
              <span
                className={`keyword-status ${sample.correct ? "correct" : "incorrect"}`}
              >
                {sample.correct ? "正确" : "错误"}
              </span>
            </div>
            <AudioPlayer
              src={sample.audio_url}
              durationSeconds={sample.duration_seconds}
            />
            <div className="keyword-sample-metrics">
              <Metric label="Keyword" value={sample.keyword || "-"} />
              <Metric
                label="Expected"
                value={sample.expected_hit ? "Hit" : "No Hit"}
              />
              <Metric
                label="Prediction"
                value={sample.predicted_hit ? "Hit" : "No Hit"}
              />
            </div>
            <SqaScoreChips scores={sample.sqa_scores} />
            <div className="keyword-transcript-grid">
              <TextBlock label="Transcript" value={sample.transcript || "-"} />
              <KeywordMatchTextBlock
                matchText={sample.match_text}
                highlights={keywordHighlightsFromSample(sample)}
              />
            </div>
          </section>
        ))}
        {visibleCount < samples.length ? (
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

export function DenoiseReportPanel({
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
            <Metric
              label="SNR Δ"
              value={formatSignedScore(result.mean_snr_delta)}
            />
            <Metric
              label="MOS Δ"
              value={formatSignedScore(result.mean_mos_delta)}
            />
            <Metric
              label="SNR Before"
              value={formatSqaScore(result.mean_original_snr)}
            />
            <Metric
              label="SNR After"
              value={formatSqaScore(result.mean_denoised_snr)}
            />
            <Metric
              label="MOS Before"
              value={formatSqaScore(result.mean_original_mos)}
            />
            <Metric
              label="MOS After"
              value={formatSqaScore(result.mean_denoised_mos)}
            />
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

function DenoiseSampleList({ samples }: { samples: DenoiseReportSample[] }) {
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
    (matrix?.rows ?? []).length > 0 &&
    (matrix?.predicted_languages ?? []).length > 0;
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
          {formatNumber(errorCount)} / 未知误接收{" "}
          {formatNumber(result.unknown_false_accept_count)}
          {" / "}
          已知拒识 {formatNumber(result.known_reject_count)}
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
                  <th>精确率</th>
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
                    <th title={row.reference_language}>
                      {row.reference_language || "-"}
                    </th>
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
                    <td>
                      {formatRate(
                        getLidMatrixRowRecall(result, row.reference_language),
                      )}
                    </td>
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
  const correctCount = toDisplayNumber(
    result.overall_correct_count ?? result.correct_count,
  );
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

function LidSampleList({ samples }: { samples: LidReportSample[] }) {
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
  accuracyLabel: string;
}) {
  return (
    <section className="asr-summary-card">
      <div className="asr-summary-title">
        <span>{label}</span>
        <strong>{formatPercentScale(summary?.wer ?? fallbackRate)}</strong>
      </div>
      <div className="report-summary">
        <Metric
          label={accuracyLabel}
          value={formatPercentScale(accuracy ?? summary?.accuracy)}
        />
        <Metric label="Correct" value={formatNumber(summary?.correct)} />
        <Metric label="Sub" value={formatNumber(summary?.substitutions)} />
        <Metric label="Del" value={formatNumber(summary?.deletions)} />
        <Metric label="Ins" value={formatNumber(summary?.insertions)} />
      </div>
    </section>
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
    { correct: 0, substitution: 0, deletion: 0, insertion: 0 } as Record<
      string,
      number
    >,
  );

  return (
    <span className="token-counts">
      {summary ? `${metricLabel} ${formatPercentScale(summary.wer)} · ` : ""}C{" "}
      {summary?.correct ?? counts.correct ?? 0} · S{" "}
      {summary?.substitutions ?? counts.substitution ?? 0} · D{" "}
      {summary?.deletions ?? counts.deletion ?? 0} · I{" "}
      {summary?.insertions ?? counts.insertion ?? 0}
    </span>
  );
}

function WerAlignmentRows({
  tokens,
  wrap,
}: {
  tokens: WerToken[];
  wrap: boolean;
}) {
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
      const metric: AlignmentMetric = sortMode.startsWith("cer")
        ? "cer"
        : "wer";
      const leftRate = getUtteranceRate(left, reports[metric]);
      const rightRate = getUtteranceRate(right, reports[metric]);
      if (leftRate !== rightRate) {
        return sortMode.endsWith("desc")
          ? rightRate - leftRate
          : leftRate - rightRate;
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

function formatRate(value: unknown): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "-";
  }
  const percentage = Math.abs(value) <= 1 ? value * 100 : value;
  return `${percentage.toFixed(2)}%`;
}

function formatPercentScale(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value.toFixed(2)}%`
    : "-";
}

function formatNumber(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? String(value)
    : "—";
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

function parseMarkdown(markdown: string): MarkdownBlock[] {
  const blocks: MarkdownBlock[] = [];
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  let paragraph: string[] = [];
  let listItems: string[] = [];
  let codeLines: string[] = [];
  let formulaLines: string[] = [];
  let codeLanguage = "";
  let inCode = false;
  let inFormula = false;

  function flushParagraph() {
    if (paragraph.length) {
      blocks.push({ type: "paragraph", text: paragraph.join(" ") });
      paragraph = [];
    }
  }

  function flushList() {
    if (listItems.length) {
      blocks.push({ type: "list", items: listItems });
      listItems = [];
    }
  }

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();

    if (trimmed === "$$") {
      if (inFormula) {
        blocks.push({ type: "formula", text: formulaLines.join("\n") });
        formulaLines = [];
        inFormula = false;
      } else {
        flushParagraph();
        flushList();
        inFormula = true;
      }
      continue;
    }

    if (inFormula) {
      formulaLines.push(line);
      continue;
    }

    if (line.startsWith("```")) {
      if (inCode) {
        const text = codeLines.join("\n");
        blocks.push(
          codeLanguage === "math" || codeLanguage === "formula"
            ? { type: "formula", text }
            : { type: "code", language: codeLanguage, text },
        );
        codeLines = [];
        codeLanguage = "";
        inCode = false;
      } else {
        flushParagraph();
        flushList();
        codeLanguage = line.slice(3).trim();
        inCode = true;
      }
      continue;
    }

    if (inCode) {
      codeLines.push(line);
      continue;
    }

    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    const heading = /^(#{1,3})\s+(.+)$/.exec(trimmed);
    if (heading) {
      flushParagraph();
      flushList();
      const text = heading[2];
      blocks.push({
        type: "heading",
        level: heading[1].length as 1 | 2 | 3,
        text,
        id: slugifyHeading(text),
      });
      continue;
    }

    if (isMarkdownTableStart(lines, index)) {
      flushParagraph();
      flushList();
      const parsedTable = parseMarkdownTable(lines, index);
      blocks.push(parsedTable.block);
      index = parsedTable.nextIndex - 1;
      continue;
    }

    if (trimmed.startsWith("- ")) {
      flushParagraph();
      listItems.push(trimmed.slice(2));
      continue;
    }

    flushList();
    paragraph.push(trimmed);
  }

  if (inCode) {
    const text = codeLines.join("\n");
    blocks.push(
      codeLanguage === "math" || codeLanguage === "formula"
        ? { type: "formula", text }
        : { type: "code", language: codeLanguage, text },
    );
  }
  if (inFormula) {
    blocks.push({ type: "formula", text: formulaLines.join("\n") });
  }
  flushParagraph();
  flushList();
  return blocks;
}

function isMarkdownTableStart(lines: string[], index: number): boolean {
  const current = lines[index]?.trim() ?? "";
  const next = lines[index + 1]?.trim() ?? "";
  return (
    current.startsWith("|") &&
    current.endsWith("|") &&
    /^\|[\s:\-|]+\|$/.test(next)
  );
}

function parseMarkdownTable(
  lines: string[],
  startIndex: number,
): { block: MarkdownBlock; nextIndex: number } {
  const headers = splitMarkdownTableRow(lines[startIndex]);
  const rows: string[][] = [];
  let index = startIndex + 2;
  while (index < lines.length) {
    const line = lines[index].trim();
    if (!line.startsWith("|") || !line.endsWith("|")) {
      break;
    }
    rows.push(splitMarkdownTableRow(line));
    index += 1;
  }
  return { block: { type: "table", headers, rows }, nextIndex: index };
}

function splitMarkdownTableRow(line: string): string[] {
  return line
    .trim()
    .slice(1, -1)
    .split("|")
    .map((cell) => cell.trim());
}

function renderInlineMarkdown(text: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const pattern = /(`([^`]+)`|\[([^\]]+)\]\((#[^)]+)\)|\$([^$]+)\$)/g;
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }
    if (match[2]) {
      nodes.push(<code key={`code-${match.index}`}>{match[2]}</code>);
    } else if (match[3] && match[4]) {
      nodes.push(
        <a href={match[4]} key={`link-${match.index}`}>
          {match[3]}
        </a>,
      );
    } else if (match[5]) {
      nodes.push(<LatexFormula key={`math-${match.index}`} text={match[5]} />);
    }
    lastIndex = pattern.lastIndex;
  }
  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }
  return nodes;
}

function slugifyHeading(text: string): string {
  return text
    .trim()
    .toLowerCase()
    .replace(/`/g, "")
    .replace(/\s+/g, "-")
    .replace(/[^\p{L}\p{N}\-_]/gu, "");
}
