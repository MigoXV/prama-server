import {
  Activity,
  ArrowRight,
  Check,
  Download,
  FolderOpen,
  Plus,
  RefreshCw,
  Upload,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { AppShell } from "./components/AppShell";
import { ConfirmDialog } from "./components/ConfirmDialog";
import { DirectoryBrowserDialog } from "./components/DirectoryBrowserDialog";
import { MarkdownDocument, StatusPill, TextField } from "./components/Reports";
import { Button } from "./components/ui";
import { usePersistentState } from "./hooks/usePersistentState";
import { useRoute, type EvaluationView } from "./hooks/useRoute";
import {
  AdvancedSettings,
  ConnectionTest,
  FIELD_LABELS,
  RequestSummary,
} from "./features/evaluation/Configuration";
import { ReportWorkspace } from "./features/evaluation/ReportWorkspace";
import {
  buildRequest,
  DEFAULT_FORM_STATE,
  initialDraft,
  mergeReevaluationFormState,
  migratedDrafts,
  normalizeFormState,
  readStored,
  TASKS,
  taskLabel,
} from "./features/evaluation/model";
import { useEvaluationJobs } from "./features/evaluation/useEvaluationJobs";
import {
  createEvaluation,
  getHelpDocument,
  listServerDirectory,
  uploadDatasetFiles,
} from "./services/evaluations";
import type {
  EvaluationFormState,
  EvaluationRequest,
  EvaluationTask,
  HelpDocument,
} from "./types";

export default function App() {
  const { route, navigate, query } = useRoute();
  const [defaults, setDefaults] = usePersistentState(
    "prama.defaults",
    normalizeFormState(readStored("prama.evaluationForm", DEFAULT_FORM_STATE)),
  );
  const [seed] = useState(migratedDrafts);
  const [drafts, setDrafts] = usePersistentState<
    Partial<Record<EvaluationTask, EvaluationFormState>>
  >("prama.drafts", seed);
  const task = (
    TASKS.some((t) => t.id === route.params.get("task"))
      ? route.params.get("task")
      : "asr"
  ) as EvaluationTask;
  const draft = normalizeFormState(
    drafts[task] ?? initialDraft(task, defaults),
  );
  const { jobs, latest, load, register } = useEvaluationJobs();
  const record = route.jobId ? jobs[route.jobId] : undefined;
  const snapshot = record?.snapshot;
  const view = (
    ["run", "report", "diagnosis", "configuration"].includes(
      route.params.get("view") ?? "",
    )
      ? route.params.get("view")
      : "run"
  ) as EvaluationView;
  const [directoryOpen, setDirectoryOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false),
    [uploading, setUploading] = useState(false),
    [formError, setFormError] = useState("");
  const [notice, setNotice] = useState("");
  const [settings, setSettings] = useState(defaults),
    [settingsDirty, setSettingsDirty] = useState(false),
    [resetOpen, setResetOpen] = useState(false);
  const [help, setHelp] = useState<HelpDocument | null>(null),
    [helpError, setHelpError] = useState(""),
    [helpAttempt, setHelpAttempt] = useState(0);
  const upload = useRef<HTMLInputElement>(null);
  const previousPage = useRef(route.page);
  const sourceId = route.params.get("from");
  const sourceRequest = sourceId
    ? jobs[sourceId]?.snapshot?.request
    : undefined;
  const activeSameTask = latest[task]
    ? jobs[latest[task]!]?.snapshot
    : undefined;
  const blocked =
    activeSameTask?.status === "queued" || activeSameTask?.status === "running";
  const differences = useMemo(
    () =>
      sourceRequest
        ? Object.entries(buildRequest(draft)).filter(
            ([key, value]) =>
              JSON.stringify(value) !==
              JSON.stringify(sourceRequest[key as keyof EvaluationRequest]),
          )
        : [],
    [draft, sourceRequest],
  );
  useEffect(() => {
    if (route.page === "job" && route.jobId) void load(route.jobId);
  }, [route.page, route.jobId, load]);
  useEffect(() => {
    if (sourceId && !jobs[sourceId]) void load(sourceId);
  }, [sourceId, load, jobs]);
  useEffect(() => {
    if (route.page !== previousPage.current) {
      setFormError("");
      previousPage.current = route.page;
    }
  }, [route.page]);
  useEffect(() => {
    if (route.page !== "help" || help) return;
    let canceled = false;
    setHelpError("");
    getHelpDocument()
      .then((x) => {
        if (!canceled) setHelp(x);
      })
      .catch((e) => {
        if (!canceled) setHelpError(e.message);
      });
    return () => {
      canceled = true;
    };
  }, [route.page, helpAttempt, help]);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 4000);
    return () => clearTimeout(timer);
  }, [notice]);
  function change(patch: Partial<EvaluationFormState>) {
    setDrafts((d) => ({ ...d, [task]: { ...draft, ...patch } }));
    setFormError("");
  }
  function chooseTask(next: EvaluationTask) {
    setDrafts((d) => ({ ...d, [task]: draft }));
    query({ task: next, from: null });
    setFormError("");
  }
  async function submit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (submitting || blocked) return;
    setSubmitting(true);
    setFormError("");
    try {
      const request = buildRequest(draft);
      const created = await createEvaluation(request);
      register({
        job_id: created.job_id,
        status: created.status,
        request,
        result: null,
        progress: null,
        error: null,
      });
      navigate(`/evaluations/${encodeURIComponent(created.job_id)}?view=run`);
    } catch (e) {
      setFormError(e instanceof Error ? e.message : "创建任务失败");
    } finally {
      setSubmitting(false);
    }
  }
  function reevaluate() {
    if (!snapshot) return;
    const t = snapshot.request.task;
    const current = normalizeFormState(drafts[t] ?? initialDraft(t, defaults));
    setDrafts((d) => ({
      ...d,
      [t]: mergeReevaluationFormState(current, snapshot.request),
    }));
    navigate(
      `/evaluations/new?task=${t}&from=${encodeURIComponent(snapshot.job_id)}`,
    );
  }
  function exportResult() {
    if (!snapshot?.result) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(snapshot.result, null, 2)], {
        type: "application/json",
      }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `${snapshot.job_id}-result.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  async function importFiles(files: File[]) {
    if (!files.length) return;
    const uploadTask = task;
    setUploading(true);
    setFormError("");
    try {
      const result = await uploadDatasetFiles(files);
      setDrafts((d) => ({
        ...d,
        [uploadTask]: {
          ...normalizeFormState(d[uploadTask] ?? draft),
          dataset_path: result.dataset_path,
        },
      }));
      setNotice(result.message || `已导入 ${result.imported_count} 个文件`);
    } catch (e) {
      setFormError(e instanceof Error ? e.message : "上传失败");
    } finally {
      setUploading(false);
      if (upload.current) upload.current.value = "";
    }
  }
  const latestEntries = TASKS.filter((t) => latest[t.id]);
  return (
    <AppShell page={route.page} navigate={navigate}>
      {route.page === "new" && (
        <>
          <header className="page-heading">
            <div>
              <h1>{sourceId ? "重新评估" : "新建评估"}</h1>
              <p>选择评估任务，连接引擎并配置本次运行。</p>
            </div>
            {sourceId && (
              <button
                className="text-action"
                onClick={() =>
                  navigate(
                    `/evaluations/${encodeURIComponent(sourceId)}?view=report`,
                  )
                }
              >
                返回原任务 →
              </button>
            )}
          </header>
          <form
            className="creation-form"
            onSubmit={submit}
            onInvalid={(e) => {
              const details = (e.target as HTMLElement).closest("details");
              if (details) details.open = true;
            }}
          >
            <fieldset
              className="task-selection"
              disabled={submitting || uploading}
            >
              <legend>评估类型</legend>
              <div className="task-options">
                {TASKS.map((t) => (
                  <label key={t.id} className={task === t.id ? "selected" : ""}>
                    <input
                      type="radio"
                      name="task"
                      checked={task === t.id}
                      onChange={() => chooseTask(t.id)}
                    />
                    <strong>{t.label}</strong>
                    <span>{t.description}</span>
                    {task === t.id && <Check size={15} aria-hidden />}
                  </label>
                ))}
              </div>
            </fieldset>
            <div className="creation-layout">
              <div className="creation-fields">
                <fieldset disabled={submitting || uploading}>
                  <section className="form-section">
                    <div className="section-label">
                      <span>01</span>
                      <div>
                        <h2>引擎与数据集</h2>
                        <p>指定评估服务与服务器上的数据目录</p>
                      </div>
                    </div>
                    <div className="engine-line">
                      <TextField
                        label={`${taskLabel(task)} 引擎地址`}
                        value={draft.target}
                        required
                        onChange={(target) => change({ target })}
                      />
                      <ConnectionTest
                        target={draft.target}
                        timeout={draft.connect_timeout_seconds}
                      />
                    </div>
                    <div className="path-field">
                      <label htmlFor="evaluation-dataset-path">
                        数据集路径
                      </label>
                      <input
                        id="evaluation-dataset-path"
                        value={draft.dataset_path}
                        required
                        onChange={(e) =>
                          change({ dataset_path: e.target.value })
                        }
                      />
                      <div className="path-actions">
                        <button
                          type="button"
                          className="text-action"
                          onClick={() => setDirectoryOpen(true)}
                        >
                          <FolderOpen size={16} />
                          浏览服务器目录
                        </button>
                        <button
                          type="button"
                          className="text-action"
                          onClick={() => upload.current?.click()}
                        >
                          <Upload size={16} />
                          {uploading ? "上传中…" : "上传本地目录"}
                        </button>
                        <input
                          ref={upload}
                          type="file"
                          hidden
                          multiple
                          {...{ webkitdirectory: "" }}
                          onChange={(e) =>
                            void importFiles(Array.from(e.target.files ?? []))
                          }
                        />
                      </div>
                      <p className="field-hint">
                        路径属于运行 Prama 的服务器。上传后将自动填入新目录。
                      </p>
                    </div>
                  </section>
                  <section className="form-section">
                    <div className="section-label">
                      <span>02</span>
                      <div>
                        <h2>运行范围</h2>
                        <p>控制本次使用的数据划分和样本数量</p>
                      </div>
                    </div>
                    <div className="range-fields">
                      <TextField
                        label="数据划分（Split）"
                        value={draft.split}
                        required
                        onChange={(split) => change({ split })}
                      />
                      <TextField
                        label="样本上限（Limit）"
                        value={draft.limit}
                        type="number"
                        min="1"
                        step="1"
                        placeholder="不限制"
                        onChange={(limit) => change({ limit })}
                      />
                    </div>
                  </section>
                  <div className="draft-reset">
                    <button
                      type="button"
                      className="text-action"
                      onClick={() => {
                        change(initialDraft(task, defaults));
                        setNotice("草稿已使用当前默认值重置");
                      }}
                    >
                      使用默认值重置草稿
                    </button>
                  </div>
                  <details className="advanced-settings">
                    <summary>
                      <span>
                        03 <strong>高级参数</strong>
                      </span>
                      <span>采样率、并发与 {taskLabel(task)} 评估设置</span>
                    </summary>
                    <AdvancedSettings value={draft} onChange={change} />
                  </details>
                </fieldset>
              </div>
              <aside className="submission-summary">
                <p className="eyebrow">本次运行</p>
                <h2>{taskLabel(task)} 评估</h2>
                <dl>
                  <div>
                    <dt>数据划分</dt>
                    <dd>{draft.split || "未填写"}</dd>
                  </div>
                  <div>
                    <dt>样本上限</dt>
                    <dd>{draft.limit || "不限制"}</dd>
                  </div>
                  <div>
                    <dt>采样率</dt>
                    <dd>{draft.sample_rate} Hz</dd>
                  </div>
                  <div>
                    <dt>请求超时</dt>
                    <dd>{draft.request_timeout_seconds} 秒</dd>
                  </div>
                </dl>
                <details className="summary-details">
                  <summary>查看全部生效参数</summary>
                  <RequestSummary request={buildRequest(draft)} />
                </details>
                {sourceRequest && (
                  <div className="reevaluation-diff">
                    <strong>与原任务比较</strong>
                    <p>数据源沿用原任务，高级参数使用当前草稿。</p>
                    {differences.length ? (
                      <ul>
                        {differences.map(([key, value]) => (
                          <li key={key}>
                            {FIELD_LABELS[key as keyof EvaluationFormState] ??
                              key}
                            ：
                            {String(
                              sourceRequest[key as keyof EvaluationRequest],
                            )}{" "}
                            → {String(value)}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p>参数与原任务一致。</p>
                    )}
                  </div>
                )}
                <div className="submission-action">
                  {formError && (
                    <div role="alert" className="error-box submission-error">
                      {formError}
                    </div>
                  )}
                  <Button
                    type="submit"
                    disabled={submitting || uploading || blocked}
                    stretch
                  >
                    <Plus size={16} />
                    {submitting ? "正在创建…" : "启动评估"}
                  </Button>
                  <p>创建独立任务，使用上方配置执行评估</p>
                </div>
              </aside>
            </div>
            {blocked && (
              <div className="inline-message">
                此类型已有任务正在运行。
                <button
                  type="button"
                  className="text-action"
                  onClick={() =>
                    navigate(`/evaluations/${latest[task]}?view=run`)
                  }
                >
                  查看运行进度 →
                </button>
              </div>
            )}
          </form>
        </>
      )}
      {route.page === "tasks" && (
        <>
          <header className="page-heading">
            <div>
              <h1>当前任务</h1>
              <p>
                此浏览器记录的各类型最近任务，最多五项。任务由当前服务进程提供。
              </p>
            </div>
            <Button onClick={() => navigate("/evaluations/new")}>
              <Plus size={16} />
              新建评估
            </Button>
          </header>
          {latestEntries.length ? (
            <div className="task-table">
              <div className="task-table-head">
                <span>评估任务</span>
                <span>数据集</span>
                <span>状态</span>
                <span>进度</span>
                <span />
              </div>
              {latestEntries.map((t) => {
                const id = latest[t.id]!,
                  r = jobs[id],
                  s = r?.snapshot;
                return (
                  <a
                    className="task-table-row"
                    key={id}
                    href={`/evaluations/${id}?view=run`}
                    onClick={(e) => {
                      if (e.ctrlKey || e.metaKey) return;
                      e.preventDefault();
                      navigate(`/evaluations/${id}?view=run`);
                    }}
                  >
                    <span>
                      <strong>{t.label} 评估</strong>
                      <small>{id}</small>
                    </span>
                    <span title={s?.request.dataset_path}>
                      {s?.request.dataset_path ?? "—"}
                    </span>
                    <span>
                      {s ? (
                        <StatusPill status={s.status} />
                      ) : r?.error ? (
                        "不可用"
                      ) : (
                        "读取中"
                      )}
                    </span>
                    <span>
                      {s?.progress?.total
                        ? `${s.progress.processed ?? 0} / ${s.progress.total}`
                        : "—"}
                    </span>
                    <ArrowRight size={16} />
                  </a>
                );
              })}
            </div>
          ) : (
            <div className="paper-empty">
              <Activity size={28} />
              <h2>还没有评估任务</h2>
              <p>从一次评估开始，运行状态和结果会显示在这里。</p>
              <Button onClick={() => navigate("/evaluations/new")}>
                新建评估
              </Button>
            </div>
          )}
        </>
      )}
      {route.page === "job" && (
        <>
          {!snapshot ? (
            <div className="paper-empty">
              <h1>{record?.error ? "任务不可用" : "正在读取任务"}</h1>
              <p>{record?.error || "正在从服务器获取运行状态…"}</p>
              {record?.error && (
                <>
                  <p>服务重启后，内存中的任务可能已失效。</p>
                  <div className="button-row">
                    <Button
                      variant="secondary"
                      onClick={() => void load(route.jobId!)}
                    >
                      重新读取
                    </Button>
                    <Button onClick={() => navigate("/evaluations/new")}>
                      新建评估
                    </Button>
                  </div>
                </>
              )}
            </div>
          ) : (
            <>
              <header className="page-heading job-heading">
                <div>
                  <button
                    className="back-link"
                    onClick={() => navigate("/evaluations")}
                  >
                    ← 当前任务
                  </button>
                  <h1>{taskLabel(snapshot.request.task)} 评估</h1>
                  <p
                    className="job-subtitle"
                    title={snapshot.request.dataset_path}
                  >
                    {snapshot.request.dataset_path}
                    <span> / {snapshot.request.split}</span>
                  </p>
                  <code className="job-id">{snapshot.job_id}</code>
                </div>
                <div className="job-actions">
                  <StatusPill status={snapshot.status} />
                  <button
                    className="ui-button ui-button-secondary"
                    disabled={
                      snapshot.status === "queued" ||
                      snapshot.status === "running"
                    }
                    onClick={reevaluate}
                  >
                    <RefreshCw size={15} />
                    重新评估
                  </button>
                  <button
                    className="text-action"
                    disabled={!snapshot.result}
                    onClick={exportResult}
                  >
                    <Download size={15} />
                    导出 JSON
                  </button>
                </div>
              </header>
              <nav className="detail-tabs" aria-label="任务详情">
                {(
                  [
                    ["run", "运行"],
                    ["report", "报告"],
                    ["diagnosis", "样本诊断"],
                    ["configuration", "运行配置"],
                  ] as const
                ).map(([id, label]) => (
                  <a
                    key={id}
                    href={`?view=${id}`}
                    aria-current={view === id ? "page" : undefined}
                    onClick={(e) => {
                      e.preventDefault();
                      query({ view: id, sample: null });
                    }}
                  >
                    {label}
                  </a>
                ))}
              </nav>
              {(record?.warning || record?.error) && (
                <div role="status" className="inline-message">
                  {record.warning || record.error}
                  <button
                    className="text-action"
                    onClick={() => void load(snapshot.job_id)}
                  >
                    刷新状态
                  </button>
                </div>
              )}
              <div className="job-content" key={snapshot.job_id}>
                {view === "run" && (
                  <div className="run-workspace">
                    <section className="run-primary">
                      <p className="eyebrow">运行状态</p>
                      <h2>
                        {snapshot.status === "queued"
                          ? "任务已进入队列"
                          : snapshot.status === "running"
                            ? "正在评估数据集"
                            : snapshot.status === "completed"
                              ? "评估已完成"
                              : "评估未完成"}
                      </h2>
                      <p>
                        {snapshot.status === "queued"
                          ? "服务正在准备评估，开始处理后将显示进度。"
                          : snapshot.status === "completed"
                            ? "结果已生成，可以查看报告或进入样本诊断。"
                            : snapshot.status === "failed"
                              ? "请检查失败原因，调整配置后创建新的评估任务。"
                              : "你可以离开此页面，服务器会继续处理。"}
                      </p>
                      {(snapshot.status === "running" ||
                        snapshot.status === "completed") && (
                        <>
                          <div className="run-count">
                            <strong>
                              {snapshot.progress?.processed ?? "—"}
                            </strong>
                            <span>
                              / {snapshot.progress?.total ?? "—"} 个样本
                            </span>
                          </div>
                          <progress
                            aria-label="评估进度"
                            max={snapshot.progress?.total || 1}
                            {...(snapshot.progress?.total
                              ? { value: snapshot.progress.processed ?? 0 }
                              : {})}
                          />
                          <p>
                            已评估 {snapshot.progress?.evaluated ?? "—"} 个样本
                          </p>
                        </>
                      )}
                      {snapshot.status === "failed" && (
                        <div className="error-box" role="alert">
                          {snapshot.error || "服务器未返回详细原因"}
                        </div>
                      )}
                      {snapshot.status === "completed" && (
                        <Button onClick={() => query({ view: "report" })}>
                          查看评估报告 <ArrowRight size={16} />
                        </Button>
                      )}
                    </section>
                    <section className="run-context">
                      <h3>当前样本</h3>
                      <code>
                        {snapshot.progress?.current_id ||
                          snapshot.progress?.id ||
                          "尚无样本"}
                      </code>
                      {snapshot.progress?.reference && (
                        <>
                          <p>参考内容</p>
                          <div>{snapshot.progress.reference}</div>
                        </>
                      )}
                      {snapshot.progress?.hypothesis && (
                        <>
                          <p>预测内容</p>
                          <div>{snapshot.progress.hypothesis}</div>
                        </>
                      )}
                      <div className="context-source">
                        <h3>本次配置</h3>
                        <p>{snapshot.request.target}</p>
                        <p>{snapshot.request.dataset_path}</p>
                        <button
                          className="text-action"
                          onClick={() => query({ view: "configuration" })}
                        >
                          查看运行配置 →
                        </button>
                      </div>
                    </section>
                  </div>
                )}
                {(view === "report" || view === "diagnosis") && (
                  <ReportWorkspace
                    snapshot={snapshot}
                    diagnosis={view === "diagnosis"}
                    params={route.params}
                    query={query}
                  />
                )}
                {view === "configuration" && (
                  <section className="configuration-snapshot">
                    <div className="section-heading">
                      <div>
                        <h2>运行配置</h2>
                        <p>
                          创建任务时的只读快照。修改默认设置不会改变本次评估。
                        </p>
                      </div>
                      <button
                        className="text-action"
                        onClick={() => {
                          void navigator.clipboard
                            .writeText(
                              JSON.stringify(snapshot.request, null, 2),
                            )
                            .then(() => setNotice("运行配置已复制"))
                            .catch(() => setNotice("复制失败，请使用结果导出"));
                        }}
                      >
                        复制配置
                      </button>
                    </div>
                    <RequestSummary request={snapshot.request} />
                  </section>
                )}
              </div>
            </>
          )}
        </>
      )}
      {route.page === "settings" && (
        <>
          <header className="page-heading">
            <div>
              <h1>默认设置</h1>
              <p>
                保存到当前浏览器，仅用于新建或重置草稿；已有草稿和运行结果保持独立。
              </p>
            </div>
            <button className="text-action" onClick={() => setResetOpen(true)}>
              恢复初始默认值
            </button>
          </header>
          <form
            className="settings-form"
            onSubmit={(e) => {
              e.preventDefault();
              setDefaults(settings);
              setSettingsDirty(false);
              setNotice("默认设置已保存");
            }}
          >
            <AdvancedSettings
              value={settings}
              all
              onChange={(patch) => {
                setSettings((s) => ({ ...s, ...patch }));
                setSettingsDirty(true);
              }}
            />
            <div className="settings-actions">
              <Button type="submit">保存默认值</Button>
              <span>
                {settingsDirty
                  ? "有未保存的更改"
                  : "默认设置已保存到当前浏览器"}
              </span>
            </div>
          </form>
        </>
      )}
      {route.page === "help" && (
        <>
          <header className="page-heading">
            <div>
              <h1>帮助</h1>
              <p>数据集格式、评估指标与输入要求</p>
            </div>
          </header>
          {help ? (
            <section className="help-panel">
              <MarkdownDocument markdown={help.markdown} />
            </section>
          ) : helpError ? (
            <div role="alert" className="paper-empty">
              <h2>帮助文档加载失败</h2>
              <p>{helpError}</p>
              <Button
                variant="secondary"
                onClick={() => setHelpAttempt((x) => x + 1)}
              >
                重新加载
              </Button>
            </div>
          ) : (
            <p role="status">正在加载帮助文档…</p>
          )}
        </>
      )}
      <DirectoryBrowserDialog
        isOpen={directoryOpen}
        initialPath={draft.dataset_path}
        listDirectory={listServerDirectory}
        onClose={() => setDirectoryOpen(false)}
        onSelect={(path) => {
          change({ dataset_path: path });
          setDirectoryOpen(false);
        }}
      />
      <ConfirmDialog
        isOpen={resetOpen}
        title="恢复初始默认值？"
        description="恢复各类评估的默认高级参数。已有草稿和评估结果不会被修改。"
        confirmLabel="恢复默认值"
        onClose={() => setResetOpen(false)}
        onConfirm={() => {
          setDefaults(DEFAULT_FORM_STATE);
          setSettings(DEFAULT_FORM_STATE);
          setSettingsDirty(false);
          setResetOpen(false);
          setNotice("已恢复初始默认值");
        }}
      />
      {notice && (
        <div className="notice-toast" role="status">
          {notice}
        </div>
      )}
    </AppShell>
  );
}
