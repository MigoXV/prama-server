import { useCallback, useEffect, useState } from "react";

export function usePersistentState<T>(
  key: string,
  initialValue: T,
): [T, (value: T | ((current: T) => T)) => void] {
  const [state, setState] = useState<T>(() => {
    const stored = localStorage.getItem(key);
    if (!stored) {
      return initialValue;
    }

    try {
      return JSON.parse(stored) as T;
    } catch {
      return initialValue;
    }
  });

  useEffect(() => {
    localStorage.setItem(key, JSON.stringify(state));
  }, [key, state]);

  const updateState = useCallback((value: T | ((current: T) => T)) => {
    setState((current) =>
      typeof value === "function" ? (value as (current: T) => T)(current) : value,
    );
  }, []);

  return [state, updateState];
}
