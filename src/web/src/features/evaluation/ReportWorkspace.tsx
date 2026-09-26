import type { EvaluationSnapshot, EvaluationResult } from "../../types";
import {
  AsrAlignmentReportPanel,
  AsrOverviewMetrics,
  DenoiseOverviewMetrics,
  DenoiseReportPanel,
  KeywordOverviewMetrics,
  KeywordReportPanel,
  LidOverviewMetrics,
  LidReportPanel,
  PerformanceMetrics,
  SqaSummaryMetrics,
} from "../../components/Reports";
import {
  VadEvaluationReport,
  type VadDiagnosisFilter,
} from "../../components/VadEvaluationReport";

export function ReportWorkspace({
  snapshot,
  diagnosis,
  params,
  query,
}: {
  snapshot: EvaluationSnapshot;
  diagnosis: boolean;
  params: URLSearchParams;
  query: (values: Record<string, string | null>, replace?: boolean) => void;
}) {
  const result = snapshot.result,
    task = snapshot.request.task;
  const search = params.get("q") ?? "",
    filter = params.get("filter") ?? "all";
  if (!result)
    return (
      <div className="paper-empty">
        <h2>
          {snapshot.status === "failed" ? "本次任务未生成报告" : "报告尚未生成"}
        </h2>
        <p>
          {snapshot.status === "failed"
            ? "返回运行页查看失败原因。"
            : "收到评估结果后会在这里展示。运行状态可在“运行”中查看。"}
        </p>
      </div>
    );
  if (task === "vad") {
    const valid = [
      "all",
      "miss",
      "false_alarm",
      "low_frame_f1",
      "oversegmentation",
    ];
    const vadFilter = (
      valid.includes(filter) ? filter : "all"
    ) as VadDiagnosisFilter;
    return (
      <>
        {diagnosis && (
          <div className="diagnosis-toolbar">
            <label>
              查找样本
              <input
                type="search"
                placeholder="输入样本 ID"
                value={search}
                onChange={(e) =>
                  query({ q: e.target.value, sample: null }, true)
                }
              />
            </label>
          </div>
        )}
        <VadEvaluationReport
          result={result}
          request={snapshot.request}
          view={diagnosis ? "diagnosis" : "overview"}
          filter={vadFilter}
          search={search}
          selectedSampleId={params.get("sample")}
          onSampleChange={(id) => query({ sample: id })}
          onViewChange={(view) =>
            query({ view: view === "overview" ? "report" : "diagnosis" })
          }
          onFilterChange={(f) =>
            query({
              filter: f === "all" ? null : f,
              sample: null,
              view: "diagnosis",
            })
          }
        />
      </>
    );
  }
  if (!diagnosis)
    return (
      <div className="report-overview">
        <div className="section-heading">
          <div>
            <h2>评估结果</h2>
            <p>指标基于本次运行的配置与数据集</p>
          </div>
          <button
            className="text-action"
            onClick={() => query({ view: "diagnosis" })}
          >
            查看样本诊断 →
          </button>
        </div>
        {task === "asr" ? (
          <AsrOverviewMetrics result={result} />
        ) : task === "lid" ? (
          <LidOverviewMetrics result={result} />
        ) : task === "keyword" ? (
          <KeywordOverviewMetrics result={result} />
        ) : (
          <DenoiseOverviewMetrics result={result} />
        )}
        <PerformanceMetrics result={result} />
        <SqaSummaryMetrics summary={result.sqa_summary} />
        <section className="report-explainer">
          <h3>从结果进入诊断</h3>
          <p>
            {task === "asr"
              ? "查看词级与字级对齐，按错误率定位识别问题。"
              : task === "lid"
                ? "结合类别指标、混淆矩阵与逐样本预测，定位语种识别错误。"
                : task === "keyword"
                  ? "按关键词或语音查看命中、漏检与虚警。"
                  : "对照原始与处理后的音频，检查质量变化和处理失败。"}
          </p>
          <button
            className="ui-button ui-button-secondary"
            onClick={() => query({ view: "diagnosis" })}
          >
            打开样本诊断
          </button>
        </section>
      </div>
    );
  const matches = (id: string) =>
    id.toLowerCase().includes(search.toLowerCase());
  const filtered: EvaluationResult = { ...result };
  if (task === "asr")
    for (const key of ["wer_report", "cer_report"] as const) {
      const report = result[key];
      if (report)
        filtered[key] = {
          ...report,
          utterances: report.utterances.filter(
            (s) =>
              matches(s.id) &&
              (filter !== "error" || (s.summary?.wer ?? 0) > 0),
          ),
        };
    }
  if (task === "lid" && result.lid_report)
    filtered.lid_report = {
      samples: result.lid_report.samples.filter(
        (s) =>
          matches(s.id) &&
          (filter === "all" ||
            (filter === "correct" ? s.correct : !s.correct)) &&
          (!params.get("language") ||
            s.reference_language === params.get("language")),
      ),
    };
  const keywordMatch = (s: {
    correct: boolean;
    expected_hit: boolean;
    predicted_hit: boolean;
  }) =>
    filter === "all" ||
    (filter === "hit"
      ? s.expected_hit && s.predicted_hit
      : filter === "miss"
        ? s.expected_hit && !s.predicted_hit
        : filter === "false_alarm"
          ? !s.expected_hit && s.predicted_hit
          : !s.correct);
  if (task === "keyword") {
    if (result.keyword_report)
      filtered.keyword_report = {
        samples: result.keyword_report.samples.filter(
          (s) => (matches(s.id) || matches(s.keyword)) && keywordMatch(s),
        ),
      };
    if (result.keyword_audio_report)
      filtered.keyword_audio_report = {
        samples: result.keyword_audio_report.samples.filter(
          (s) => matches(s.id) && s.keywords.some(keywordMatch),
        ),
      };
  }
  if (task === "denoise" && result.denoise_report)
    filtered.denoise_report = {
      samples: result.denoise_report.samples.filter(
        (s) => matches(s.id) && (filter !== "error" || Boolean(s.error)),
      ),
    };
  const filters =
    task === "keyword"
      ? [
          ["all", "全部"],
          ["hit", "命中"],
          ["miss", "漏检"],
          ["false_alarm", "虚警"],
        ]
      : task === "lid"
        ? [
            ["all", "全部"],
            ["error", "识别错误"],
            ["correct", "正确"],
          ]
        : [
            ["all", "全部"],
            ["error", task === "denoise" ? "处理失败" : "错误样本"],
          ];
  const count =
    task === "asr"
      ? (filtered.wer_report?.utterances.length ?? 0)
      : task === "lid"
        ? (filtered.lid_report?.samples.length ?? 0)
        : task === "keyword"
          ? (filtered.keyword_report?.samples.length ?? 0)
          : (filtered.denoise_report?.samples.length ?? 0);
  return (
    <>
      <div className="diagnosis-toolbar">
        <div className="filter-buttons" role="group" aria-label="样本筛选">
          {filters.map(([id, label]) => (
            <button
              key={id}
              aria-pressed={filter === id}
              onClick={() =>
                query({ filter: id === "all" ? null : id, sample: null })
              }
            >
              {label}
            </button>
          ))}
        </div>
        <label>
          查找样本
          <input
            type="search"
            value={search}
            placeholder={
              task === "keyword" ? "样本 ID 或关键词" : "输入样本 ID"
            }
            onChange={(e) => query({ q: e.target.value }, true)}
          />
        </label>
        {task === "lid" && (
          <label>
            参考语种
            <select
              value={params.get("language") ?? ""}
              onChange={(e) => query({ language: e.target.value })}
            >
              <option value="">全部语种</option>
              {Array.from(
                new Set(
                  result.lid_report?.samples.map((s) => s.reference_language),
                ),
              ).map((x) => (
                <option key={x}>{x}</option>
              ))}
            </select>
          </label>
        )}
      </div>
      <p className="result-count">
        匹配 {count} 个样本 · 汇总指标保持本次完整评估口径
      </p>
      {count === 0 ? (
        <div className="paper-empty">
          <h2>没有符合条件的样本</h2>
          <button
            className="text-action"
            onClick={() => query({ q: null, filter: null, language: null })}
          >
            清除筛选
          </button>
        </div>
      ) : task === "asr" ? (
        <AsrAlignmentReportPanel
          result={result}
          werReport={filtered.wer_report}
          cerReport={filtered.cer_report}
          activeMetric={params.get("metric") === "cer" ? "cer" : "wer"}
          onActiveMetricChange={(m) => query({ metric: m })}
          sortMode={
            ([
              "index-asc",
              "index-desc",
              "wer-desc",
              "wer-asc",
              "cer-desc",
              "cer-asc",
            ].includes(params.get("sort") ?? "")
              ? params.get("sort")
              : "index-asc") as "index-asc"
          }
          onSortModeChange={(s) => query({ sort: s })}
          wrapAlignment={params.get("wrap") !== "off"}
          onWrapAlignmentChange={(v) => query({ wrap: v ? null : "off" })}
        />
      ) : task === "lid" ? (
        <LidReportPanel result={filtered} />
      ) : task === "keyword" ? (
        <KeywordReportPanel
          result={filtered}
          viewMode={params.get("mode") === "audio" ? "audio" : "keyword"}
          onViewModeChange={(mode) => query({ mode })}
        />
      ) : (
        <DenoiseReportPanel result={filtered} />
      )}
    </>
  );
}
