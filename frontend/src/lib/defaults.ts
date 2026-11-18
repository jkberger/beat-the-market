import type { Strategy } from "@/api/types";

export const DEFAULT_STRATEGY: Strategy = {
  name: "RSI swing",
  buy: {
    kind: "group",
    join: "and",
    items: [
      {
        kind: "condition",
        left: { kind: "indicator", id: "rsi", params: { length: 14 }, output: "rsi" },
        op: "crosses_below",
        right: { kind: "constant", value: 30 },
        bars: 1,
      },
    ],
  },
  sell: {
    kind: "group",
    join: "and",
    items: [
      {
        kind: "condition",
        left: { kind: "indicator", id: "rsi", params: { length: 14 }, output: "rsi" },
        op: "crosses_above",
        right: { kind: "constant", value: 70 },
        bars: 1,
      },
    ],
  },
  stop_loss: null,
  take_profit: null,
  max_hold_bars: null,
  allow_short: false,
};

export function hasRanges(node: unknown): boolean {
  if (Array.isArray(node)) return node.some(hasRanges);
  if (node && typeof node === "object") {
    const keys = Object.keys(node);
    if (keys.length === 3 && ["min", "max", "step"].every((k) => keys.includes(k))) return true;
    return Object.values(node).some(hasRanges);
  }
  return false;
}
