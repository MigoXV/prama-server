import { useCallback, useEffect, useRef, useState } from "react";
import {
  getEvaluation,
  subscribeEvaluationEvents,
} from "../../services/evaluations";
import type { EvaluationSnapshot, EvaluationTask } from "../../types";
import { usePersistentState } from "../../hooks/usePersistentState";
import { readStored } from "./model";

export interface JobRecord {
  snapshot?: EvaluationSnapshot;
  loading: boolean;
  error: string;
  warning: string;
}
export function useEvaluationJobs() {
  const [jobs, setJobs] = useState<Record<string, JobRecord>>({});
  const [latest, setLatest] = usePersistentState<
    Partial<Record<EvaluationTask, string>>
  >("prama.latestJobs", {});
  const closers = useRef(new Map<string, () => void>());
  const loading = useRef(new Map<string, Promise<void>>());
  const alive = useRef(true);
  const update = useCallback((id: string, fn: (r: JobRecord) => JobRecord) => {
    if (alive.current)
      setJobs((j) => ({
        ...j,
        [id]: fn(j[id] ?? { loading: false, error: "", warning: "" }),
      }));
  }, []);
  const remember = useCallback(
    (snapshot: EvaluationSnapshot) =>
      setLatest((x) => ({ ...x, [snapshot.request.task]: snapshot.job_id })),
    [setLatest],
  );
  const watch = useCallback(
    (id: string) => {
      if (closers.current.has(id)) return;
      const close = subscribeEvaluationEvents(id, {
        onProgress: (p) =>
          update(id, (r) => ({
            ...r,
            warning: "",
            snapshot: r.snapshot
              ? {
                  ...r.snapshot,
                  status:
                    p.status === "started"
                      ? "running"
                      : (p.status ?? "running"),
                  progress: p,
                  result: p.result ?? r.snapshot.result,
                }
              : undefined,
          })),
        onPartialProgress: (p) =>
          update(id, (r) => ({
            ...r,
            warning: "",
            snapshot: r.snapshot
              ? { ...r.snapshot, progress: p, status: "running" }
              : undefined,
          })),
        onMetricSnapshot: (result) =>
          update(id, (r) => ({
            ...r,
            snapshot: r.snapshot ? { ...r.snapshot, result } : undefined,
          })),
        onMetric: ({ metric, summary, utterance }) =>
          update(id, (r) => {
            if (!r.snapshot) return r;
            const result = r.snapshot.result ?? {},
              key = metric === "wer" ? "wer_report" : "cer_report",
              previous = result[key];
            if (
              previous &&
              previous.summary.sentence_count >= summary.sentence_count
            )
              return r;
            return {
              ...r,
              snapshot: {
                ...r.snapshot,
                result: {
                  ...result,
                  [metric]: summary.wer,
                  [metric === "wer" ? "word_accuracy" : "character_accuracy"]:
                    summary.accuracy,
                  [key]: {
                    summary,
                    utterances: [...(previous?.utterances ?? []), utterance],
                  },
                },
              },
            };
          }),
        onDone: (snapshot) => {
          update(id, (r) => ({
            ...r,
            snapshot,
            loading: false,
            error: "",
            warning: "",
          }));
          closers.current.get(id)?.();
          closers.current.delete(id);
        },
        onError: (error) => {
          update(id, (r) => ({
            ...r,
            snapshot: r.snapshot
              ? { ...r.snapshot, status: "failed", error }
              : undefined,
          }));
          closers.current.get(id)?.();
          closers.current.delete(id);
        },
        onConnectionError: () =>
          update(id, (r) => ({
            ...r,
            warning: "实时连接中断，正在重连；当前显示最近收到的数据。",
          })),
      });
      closers.current.set(id, close);
    },
    [update],
  );
  const load = useCallback(
    (id: string, rememberJob = false): Promise<void> => {
      const pending = loading.current.get(id);
      if (pending) return pending;
      update(id, (r) => ({ ...r, loading: true, error: "" }));
      const promise = getEvaluation(id)
        .then((snapshot) => {
          if (!alive.current) return;
          update(id, (r) => ({
            ...r,
            snapshot,
            loading: false,
            error: "",
            warning: "",
          }));
          if (rememberJob) remember(snapshot);
          if (snapshot.status === "running" || snapshot.status === "queued")
            watch(id);
          else {
            closers.current.get(id)?.();
            closers.current.delete(id);
          }
        })
        .catch((e) =>
          update(id, (r) => ({
            ...r,
            loading: false,
            error: e instanceof Error ? e.message : "任务读取失败",
          })),
        )
        .finally(() => loading.current.delete(id));
      loading.current.set(id, promise);
      return promise;
    },
    [remember, update, watch],
  );
  const register = useCallback(
    (snapshot: EvaluationSnapshot) => {
      update(snapshot.job_id, (r) => ({
        ...r,
        snapshot,
        error: "",
        warning: "",
        loading: false,
      }));
      remember(snapshot);
      watch(snapshot.job_id);
    },
    [remember, watch, update],
  );
  useEffect(() => {
    alive.current = true;
    // URL 决定当前页面；旧版本的最近任务只迁移到列表，不强制跳转。
    const ids = Object.values(
      readStored<Partial<Record<EvaluationTask, string>>>(
        "prama.latestJobs",
        {},
      ),
    );
    const old = readStored<string>("prama.lastEvaluationJobId", "");
    if (!ids.length && old) void load(old, true);
    for (const id of ids) if (id) void load(id);
    return () => {
      alive.current = false;
      closers.current.forEach((close) => close());
      closers.current.clear();
    };
  }, [load]);
  return { jobs, latest, load, register };
}
