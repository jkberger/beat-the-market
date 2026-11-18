import type { IndicatorSpec } from "@/api/types";

export const specs: IndicatorSpec[] = [
  { id: "rsi", name: "Relative Strength Index", category: "momentum", outputs: ["rsi"], overlay: false,
    params: [{ name: "length", kind: "int", default: 14, min: 2, max: 200 }] },
  { id: "sma", name: "Simple Moving Average", category: "trend", outputs: ["sma"], overlay: true,
    params: [{ name: "length", kind: "int", default: 20, min: 2, max: 400 }] },
  { id: "macd", name: "MACD", category: "trend", outputs: ["macd", "signal", "hist"], overlay: false,
    params: [
      { name: "fast", kind: "int", default: 12, min: 2, max: 400 },
      { name: "slow", kind: "int", default: 26, min: 2, max: 400 },
      { name: "signal", kind: "int", default: 9, min: 2, max: 400 },
    ] },
];
