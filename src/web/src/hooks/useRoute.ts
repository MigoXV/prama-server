import { useCallback, useEffect, useState } from "react";

export type EvaluationView = "run" | "report" | "diagnosis" | "configuration";
export interface AppRoute {
  page: "new" | "tasks" | "job" | "settings" | "help";
  jobId?: string;
  params: URLSearchParams;
}
function readRoute(): AppRoute {
  const params = new URLSearchParams(window.location.search);
  const path = window.location.pathname;
  if (path === "/settings") return { page: "settings", params };
  if (path === "/help") return { page: "help", params };
  if (path === "/evaluations") return { page: "tasks", params };
  if (path.startsWith("/evaluations/") && path !== "/evaluations/new") {
    try {
      return { page: "job", jobId: decodeURIComponent(path.slice(13)), params };
    } catch {
      return { page: "tasks", params };
    }
  }
  return { page: "new", params };
}
export function useRoute() {
  const [route, setRoute] = useState(readRoute);
  useEffect(() => {
    const update = () => setRoute(readRoute());
    window.addEventListener("popstate", update);
    return () => window.removeEventListener("popstate", update);
  }, []);
  const navigate = useCallback((url: string, replace = false) => {
    if (url === window.location.pathname + window.location.search) return;
    window.history[replace ? "replaceState" : "pushState"]({}, "", url);
    setRoute(readRoute());
  }, []);
  const query = useCallback(
    (values: Record<string, string | null>, replace = false) => {
      const params = new URLSearchParams(window.location.search);
      for (const [key, value] of Object.entries(values)) {
        if (value) params.set(key, value);
        else params.delete(key);
      }
      navigate(
        window.location.pathname + (params.size ? `?${params}` : ""),
        replace,
      );
    },
    [navigate],
  );
  return { route, navigate, query };
}
