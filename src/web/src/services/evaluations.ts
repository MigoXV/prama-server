import type {
  EvaluationCreated,
  DatasetUploadResult,
  EvaluationProgress,
  EvaluationRequest,
  EvaluationSnapshot,
  HelpDocument,
  ServerDirectoryListing,
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

export async function getEvaluation(jobId: string): Promise<EvaluationSnapshot> {
  const response = await fetch(`/api/evaluations/${jobId}`);

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json() as Promise<EvaluationSnapshot>;
}

export async function testEngineConnectivity(
  target: string,
  timeoutSeconds: number | null,
): Promise<{ ok: boolean; target: string; message: string }> {
  const response = await fetch("/api/engines/connectivity", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      target,
      timeout_seconds: timeoutSeconds ?? 3,
    }),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json() as Promise<{ ok: boolean; target: string; message: string }>;
}

export async function getHelpDocument(): Promise<HelpDocument> {
  const response = await fetch("/api/help");

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json() as Promise<HelpDocument>;
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

export async function listServerDirectory(
  path?: string,
): Promise<ServerDirectoryListing> {
  const params = new URLSearchParams();
  if (path) {
    params.set("path", path);
  }
  const response = await fetch(
    `/api/files/directories${params.size ? `?${params.toString()}` : ""}`,
  );

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json() as Promise<ServerDirectoryListing>;
}

export async function uploadDatasetFiles(
  files: File[],
): Promise<DatasetUploadResult> {
  const formData = new FormData();
  for (const file of files) {
    const relativePath =
      (file as File & { webkitRelativePath?: string }).webkitRelativePath ||
      file.name;
    formData.append("files", file, relativePath);
  }

  const response = await fetch("/api/datasets/upload", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json() as Promise<DatasetUploadResult>;
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
