import type { Condition, IndicatorSpec, Operand } from "@/api/types";

export type Side = "buy" | "sell";

/** Widely used oversold / overbought levels: [buy level, sell level]. */
const LEVELS: Record<string, { output?: string; buy: number; sell: number }> = {
  rsi: { buy: 30, sell: 70 },
  stoch: { output: "k", buy: 20, sell: 80 },
  stochrsi: { output: "k", buy: 20, sell: 80 },
  willr: { buy: -80, sell: -20 },
  cci: { buy: -100, sell: 100 },
  mfi: { buy: 20, sell: 80 },
};

/** Oscillators that are read by crossing zero. */
const ZERO_LINE = new Set(["roc", "mom", "cmf"]);

/** Bands: buy when price dips under the lower band, sell when it pops over the upper band. */
const BANDS = new Set(["bbands", "keltner", "donchian"]);

function params(spec: IndicatorSpec) {
  return Object.fromEntries(spec.params.map((p) => [p.name, p.default]));
}

function ind(spec: IndicatorSpec, output = spec.outputs[0]): Extract<Operand, { kind: "indicator" }> {
  return { kind: "indicator", id: spec.id, params: params(spec), output };
}

/** Moving averages and the slow length each is compared against. */
const MOVING_AVERAGES: Record<string, number> = { sma: 200, ema: 50, wma: 50, hma: 50, vwma: 50 };

const close: Operand = { kind: "price", field: "close" };
const num = (value: number): Operand => ({ kind: "constant", value });

/**
 * The standard way a trader reads this indicator, as a ready-made buy or sell condition.
 * Oscillators use crossing into oversold (buy) / overbought (sell), matching the default strategy.
 */
export function suggestCondition(spec: IndicatorSpec, side: Side): Condition {
  const buy = side === "buy";
  const lv = LEVELS[spec.id];
  if (lv) {
    return {
      kind: "condition",
      left: ind(spec, lv.output),
      op: buy ? "crosses_below" : "crosses_above",
      right: num(buy ? lv.buy : lv.sell),
      bars: 1,
    };
  }
  if (spec.id === "macd") {
    return { kind: "condition", left: ind(spec, "macd"), op: buy ? "crosses_above" : "crosses_below", right: ind(spec, "signal"), bars: 1 };
  }
  if (spec.id === "aroon") {
    return { kind: "condition", left: ind(spec, "up"), op: buy ? "crosses_above" : "crosses_below", right: ind(spec, "down"), bars: 1 };
  }
  if (spec.id === "adx") {
    return { kind: "condition", left: ind(spec, "plus_di"), op: buy ? "crosses_above" : "crosses_below", right: ind(spec, "minus_di"), bars: 1 };
  }
  if (ZERO_LINE.has(spec.id)) {
    return { kind: "condition", left: ind(spec), op: buy ? "crosses_above" : "crosses_below", right: num(0), bars: 1 };
  }
  if (BANDS.has(spec.id)) {
    return { kind: "condition", left: close, op: buy ? "crosses_below" : "crosses_above", right: ind(spec, buy ? "lower" : "upper"), bars: 1 };
  }
  if (spec.id === "ichimoku") {
    return { kind: "condition", left: close, op: buy ? "crosses_above" : "crosses_below", right: ind(spec, "base"), bars: 1 };
  }
  if (MOVING_AVERAGES[spec.id]) {
    // The picked average stays on the left and is compared with a slower one of the same
    // kind: fast crossing above slow is a buy (SMA 50 over SMA 200 is the golden cross).
    const fast = ind(spec);
    const slow = { ...fast, params: { ...fast.params, length: MOVING_AVERAGES[spec.id] } };
    return { kind: "condition", left: fast, op: buy ? "crosses_above" : "crosses_below", right: slow, bars: 1 };
  }
  if (spec.overlay) {
    // Other lines drawn on price: the line dropping under price is a buy.
    return { kind: "condition", left: ind(spec), op: buy ? "crosses_below" : "crosses_above", right: close, bars: 1 };
  }
  // ATR, OBV and anything else: direction over the last 3 bars.
  return { kind: "condition", left: ind(spec), op: buy ? "rising" : "falling", right: null, bars: 3 };
}
