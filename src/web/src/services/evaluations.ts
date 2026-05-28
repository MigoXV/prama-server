import type {
  EvaluationCreated,
  EvaluationProgress,
  EvaluationRequest,
  EvaluationResult,
  EvaluationSnapshot,
} from "../types";

export interface EvaluationEventHandlers {
  onProgress: (progress: EvaluationProgress) => void;
  onPartialProgress: (progress: EvaluationProgress) => void;
  onDone: (snapshot: EvaluationSnapshot) => void;
  onError: (message: string) => void;
  onConnectionError?: () => void;
}

export async function createEvaluation(
  request: EvaluationRequest,
): Promise<EvaluationCreated> {
  const response = await fetch("/api/evaluations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json() as Promise<EvaluationCreated>;
}

export function subscribeEvaluationEvents(
  jobId: string,
  handlers: EvaluationEventHandlers,
): () => void {
  const eventSource = new EventSource(`/api/evaluations/${jobId}/events`);

  eventSource.addEventListener("inference_result", (event) => {
    handlers.onProgress(parseEventData<EvaluationProgress>(event));
  });

  eventSource.addEventListener("progress", (event) => {
    handlers.onProgress(parseEventData<EvaluationProgress>(event));
  });

  eventSource.addEventListener("partial_inference_result", (event) => {
    handlers.onPartialProgress(parseEventData<EvaluationProgress>(event));
  });

  eventSource.addEventListener("error", (event) => {
    if ("data" in event && typeof event.data === "string" && event.data) {
      const payload = parseEventData<{ message?: string }>(event as MessageEvent<string>);
      handlers.onError(payload.message || "评估任务失败");
      return;
    }
    handlers.onConnectionError?.();
  });

  eventSource.addEventListener("done", (event) => {
    handlers.onDone(parseEventData<EvaluationSnapshot>(event));
    eventSource.close();
  });

  return () => eventSource.close();
}

export async function recalculateEvaluationMetrics(
  jobId: string,
  excludedSampleIds: string[],
): Promise<EvaluationResult> {
  const response = await fetch(`/api/evaluations/${jobId}/metrics/recalculate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ excluded_sample_ids: excludedSampleIds }),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json() as Promise<EvaluationResult>;
}

function parseEventData<T>(event: MessageEvent<string>): T {
  return JSON.parse(event.data) as T;
}

async function readErrorMessage(response: Response): Promise<string> {
  const text = await response.text();
  if (!text) {
    return `${response.status} ${response.statusText}`;
  }

  try {
    const payload = JSON.parse(text) as { detail?: unknown; message?: unknown };
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    if (typeof payload.message === "string") {
      return payload.message;
    }
  } catch {
    return text;
  }

  return text;
}
