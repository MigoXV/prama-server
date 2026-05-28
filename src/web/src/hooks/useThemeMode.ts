import { useEffect, useMemo, useState } from "react";
import type { ThemeMode } from "../types";
import { usePersistentState } from "./usePersistentState";

export function useThemeMode() {
  const [themeMode, setThemeMode] = usePersistentState<ThemeMode>(
    "prama.themeMode",
    "system",
  );
  const [systemDark, setSystemDark] = useState(false);

  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const updateSystemTheme = () => setSystemDark(media.matches);
    updateSystemTheme();
    media.addEventListener("change", updateSystemTheme);
    return () => media.removeEventListener("change", updateSystemTheme);
  }, []);

  const effectiveTheme = useMemo(
    () => (themeMode === "system" ? (systemDark ? "dark" : "light") : themeMode),
    [systemDark, themeMode],
  );

  useEffect(() => {
    document.documentElement.dataset.theme = effectiveTheme;
  }, [effectiveTheme]);

  return { themeMode, effectiveTheme, setThemeMode };
}
