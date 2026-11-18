import type { Group, Operand, Strategy } from "@/api/types";

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

type Ind = Extract<Operand, { kind: "indicator" }>;

function indicators(g: Group, out: Ind[] = []): Ind[] {
  for (const item of g.items) {
    if (item.kind === "group") indicators(item, out);
    else for (const o of [item.left, item.right]) if (o && o.kind === "indicator") out.push(o);
  }
  return out;
}

const len = (o: Ind) => (typeof o.params.length === "number" ? o.params.length : null);

/** A short name that says what the rules use, e.g. "SMA 50/200 cross" or "RSI + MACD". */
export function autoName(s: Strategy): string {
  const all = [...indicators(s.buy), ...indicators(s.sell)];
  if (!all.length) return "Price rules";
  const ids = [...new Set(all.map((o) => o.id))];
  if (ids.length === 1) {
    const lengths = [...new Set(all.map(len).filter((x): x is number => x !== null))].sort((a, b) => a - b);
    const tag = ids[0].toUpperCase();
    if (lengths.length === 2) return `${tag} ${lengths[0]}/${lengths[1]} cross`;
    return lengths.length === 1 ? `${tag} ${lengths[0]} strategy` : `${tag} strategy`;
  }
  return ids.map((i) => i.toUpperCase()).join(" + ");
}
