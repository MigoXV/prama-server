import { useState } from "react";
import type { EvaluationFormState, EvaluationRequest } from "../../types";
import { TextField } from "../../components/Reports";
import { testEngineConnectivity } from "../../services/evaluations";
import { taskLabel } from "./model";

type Key = keyof EvaluationFormState;
interface Spec {
  key: Key;
  label: string;
  hint?: string;
  min?: string;
  max?: string;
  step?: string;
  type?: string;
  optional?: boolean;
}
const common: Spec[] = [
  { key: "sample_rate", label: "采样率（Hz）", min: "1" },
  {
    key: "connect_timeout_seconds",
    label: "连接超时（秒）",
    min: "0.1",
    step: "0.1",
    optional: true,
  },
  {
    key: "request_timeout_seconds",
    label: "请求超时（秒）",
    min: "0.1",
    step: "0.1",
  },
];
const asr: Spec[] = [
  {
    key: "language_code",
    label: "识别语言",
    type: "text",
    hint: "例如 en-US、zh-CN",
  },
  { key: "min_reference_words", label: "最少参考词数", min: "0" },
  {
    key: "asr_inference_concurrency",
    label: "识别并发数",
    min: "0",
    hint: "0 表示使用引擎默认值",
  },
  {
    key: "hotwords",
    label: "热词",
    type: "text",
    optional: true,
    hint: "多个热词以英文逗号分隔",
  },
  { key: "hotword_bias", label: "热词权重", step: "0.1" },
];
const vad: Spec[] = [
  {
    key: "mask_frame_seconds",
    label: "帧长（秒）",
    min: "0.001",
    step: "0.001",
  },
  {
    key: "chunk_duration_seconds",
    label: "分块时长（秒）",
    min: "0.01",
    step: "0.01",
  },
  {
    key: "speech_padding_seconds",
    label: "语音扩展（秒）",
    min: "0",
    step: "0.01",
  },
  {
    key: "hit_threshold",
    label: "段命中阈值",
    min: "0",
    max: "1",
    step: "0.01",
  },
  {
    key: "vad_inference_concurrency",
    label: "VAD 并发数",
    min: "0",
    hint: "0 表示使用引擎默认值",
  },
];
const lid: Spec[] = [
  { key: "lid_inference_concurrency", label: "LID 并发数", min: "0" },
  {
    key: "lid_confidence_threshold",
    label: "置信度阈值",
    min: "0",
    max: "1",
    step: "0.01",
  },
];
export const FIELD_LABELS: Partial<Record<Key, string>> = Object.fromEntries(
  [...common, ...asr, ...vad, ...lid].map((x) => [x.key, x.label]),
);
export function ConnectionTest({
  target,
  timeout,
}: {
  target: string;
  timeout: string;
}) {
  const [status, setStatus] = useState<{
    target: string;
    text: string;
    busy: boolean;
    ok?: boolean;
  }>({ target: "", text: "", busy: false });
  async function test() {
    setStatus({ target, text: "正在连接…", busy: true });
    try {
      const r = await testEngineConnectivity(target, Number(timeout) || 3);
      setStatus({
        target,
        text: r.ok ? "引擎连接正常" : r.message,
        busy: false,
        ok: r.ok,
      });
    } catch (e) {
      setStatus({
        target,
        text: e instanceof Error ? e.message : "连接失败",
        busy: false,
        ok: false,
      });
    }
  }
  return (
    <div className="connection-test">
      <button
        type="button"
        className="ui-button ui-button-secondary"
        disabled={!target.trim() || status.busy}
        onClick={test}
      >
        {status.busy ? "连接中…" : "测试连接"}
      </button>
      {status.text && status.target === target ? (
        <span role="status" className={status.ok ? "success-text" : "muted"}>
          {status.text}
        </span>
      ) : null}
    </div>
  );
}
export function AdvancedSettings({
  value,
  onChange,
  all = false,
}: {
  value: EvaluationFormState;
  onChange: (patch: Partial<EvaluationFormState>) => void;
  all?: boolean;
}) {
  function fields(specs: Spec[]) {
    return (
      <div className="parameter-grid">
        {specs.map((s) => (
          <div key={s.key} className={s.key === "hotwords" ? "wide-field" : ""}>
            <TextField
              label={s.label}
              value={String(value[s.key] ?? "")}
              type={s.type ?? "number"}
              required={!s.optional}
              min={s.min}
              max={s.max}
              step={s.step}
              onChange={(v) => onChange({ [s.key]: v })}
            />
            {s.hint ? <p className="field-hint">{s.hint}</p> : null}
          </div>
        ))}
      </div>
    );
  }
  function check(key: Key, label: string) {
    return (
      <label className="check-field">
        <input
          type="checkbox"
          checked={Boolean(value[key])}
          onChange={(e) => onChange({ [key]: e.target.checked })}
        />
        {label}
      </label>
    );
  }
  return (
    <div className="parameter-sections">
      <section>
        <h3>通用参数</h3>
        {fields(common)}
      </section>
      {(all || value.task === "asr" || value.task === "keyword") && (
        <section>
          <h3>识别参数</h3>
          {fields(asr)}
          <div className="check-row">
            {check("interim_results", "启用临时识别结果")}
            {check("remove_punctuation", "评估时去掉标点")}
          </div>
        </section>
      )}
      {(all || value.task === "vad") && (
        <section>
          <h3>语音活动检测</h3>
          {fields(vad)}
          {check("streaming", "使用 VAD 流式接口")}
        </section>
      )}
      {(all || value.task === "lid") && (
        <section>
          <h3>语种识别</h3>
          {fields(lid)}
        </section>
      )}
      <section>
        <h3>语音质量评估</h3>
        <p className="field-hint">启用后调用对应的 MOS / SNR 服务。</p>
        <div className="quality-options">
          {(
            [
              ["enable_mos", "mos_target", "MOS"],
              ["enable_snr", "snr_target", "SNR"],
            ] as const
          ).map(([enabled, target, label]) => (
            <div key={enabled}>
              {check(enabled, label)}
              {value[enabled] && (
                <div className="quality-address">
                  <TextField
                    label={`${label} 引擎地址`}
                    value={value[target]}
                    required
                    onChange={(v) => onChange({ [target]: v })}
                  />
                  <ConnectionTest
                    target={value[target]}
                    timeout={value.connect_timeout_seconds}
                  />
                </div>
              )}
            </div>
          ))}
        </div>
        {fields([
          {
            key: "sqa_inference_concurrency",
            label: "MOS / SNR 并发数",
            min: "0",
            hint: "0 表示使用引擎默认值",
          },
        ])}
      </section>
    </div>
  );
}
export function RequestSummary({ request }: { request: EvaluationRequest }) {
  const labels: Record<string, string> = {
    task: "评估类型",
    target: "引擎地址",
    dataset_path: "数据集路径",
    split: "数据划分",
    limit: "样本上限",
    ...FIELD_LABELS,
    enable_mos: "启用 MOS",
    enable_snr: "启用 SNR",
    mos_target: "MOS 引擎",
    snr_target: "SNR 引擎",
    sqa_inference_concurrency: "质量评估并发数",
    streaming: "VAD 流式接口",
    interim_results: "临时识别结果",
    remove_punctuation: "去掉标点",
  };
  const relevant = new Set<string>([
    "task",
    "target",
    "dataset_path",
    "split",
    "limit",
    ...common.map((x) => x.key),
    "enable_mos",
    "enable_snr",
  ]);
  for (const s of request.task === "vad"
    ? vad
    : request.task === "lid"
      ? lid
      : request.task === "denoise"
        ? []
        : asr)
    relevant.add(s.key);
  if (request.task === "vad") relevant.add("streaming");
  if (request.task === "asr" || request.task === "keyword") {
    relevant.add("interim_results");
    relevant.add("remove_punctuation");
  }
  if (request.enable_mos) relevant.add("mos_target");
  if (request.enable_snr) relevant.add("snr_target");
  if (request.enable_mos || request.enable_snr)
    relevant.add("sqa_inference_concurrency");
  return (
    <dl className="request-summary">
      {Object.entries(request)
        .filter(([k]) => relevant.has(k))
        .map(([key, value]) => (
          <div key={key}>
            <dt>{labels[key] ?? key}</dt>
            <dd>
              {key === "task"
                ? taskLabel(request.task)
                : value === null
                  ? "不限制"
                  : typeof value === "boolean"
                    ? value
                      ? "是"
                      : "否"
                    : Array.isArray(value)
                      ? value.join("、") || "未设置"
                      : String(value) || "未设置"}
            </dd>
          </div>
        ))}
    </dl>
  );
}
