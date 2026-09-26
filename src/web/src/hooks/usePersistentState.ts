import { useCallback, useEffect, useState } from "react";

export function usePersistentState<T>(
  key: string,
  initialValue: T,
): [T, (value: T | ((current: T) => T)) => void] {
  const [state, setState] = useState<T>(() => {
    try {
      const stored = localStorage.getItem(key);
      if (!stored) return initialValue;
      return JSON.parse(stored) as T;
    } catch {
      return initialValue;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(state));
    } catch {
      // 禁用或写满浏览器存储时，本次会话仍然可以工作。
    }
  }, [key, state]);

  const updateState = useCallback((value: T | ((current: T) => T)) => {
    setState((current) =>
      typeof value === "function"
        ? (value as (current: T) => T)(current)
        : value,
    );
  }, []);

  return [state, updateState];
}
