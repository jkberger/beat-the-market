import { useEffect, useState } from "react";

function rgb(name: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v ? `rgb(${v.split(/\s+/).join(", ")})` : "#888";
}

export interface ChartTheme {
  bg: string;
  text: string;
  muted: string;
  border: string;
  accent: string;
  gain: string;
  loss: string;
}

export function readTheme(): ChartTheme {
  return {
    bg: rgb("--surface"),
    text: rgb("--text"),
    muted: rgb("--muted"),
    border: rgb("--border"),
    accent: rgb("--accent"),
    gain: rgb("--gain"),
    loss: rgb("--loss"),
  };
}

/** Re-reads theme colors when the light/dark class on <html> changes. */
export function useChartTheme(): ChartTheme {
  const [theme, setTheme] = useState(readTheme);
  useEffect(() => {
    const obs = new MutationObserver(() => setTheme(readTheme()));
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => obs.disconnect();
  }, []);
  return theme;
}

export const SERIES_COLORS = ["#6366f1", "#f59e0b", "#06b6d4", "#a855f7", "#ec4899", "#84cc16"];
