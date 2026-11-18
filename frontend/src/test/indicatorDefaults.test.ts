import { DEFAULT_STRATEGY } from "@/lib/defaults";
import { suggestCondition } from "@/lib/indicatorDefaults";
import { specs } from "./fixtures";

const rsi = specs.find((s) => s.id === "rsi")!;
const sma = specs.find((s) => s.id === "sma")!;
const macd = specs.find((s) => s.id === "macd")!;

test("default strategy buys on RSI crossing below 30 and sells crossing above 70", () => {
  const buy = DEFAULT_STRATEGY.buy.items[0];
  const sell = DEFAULT_STRATEGY.sell.items[0];
  expect(buy).toMatchObject({ op: "crosses_below", right: { kind: "constant", value: 30 } });
  expect(sell).toMatchObject({ op: "crosses_above", right: { kind: "constant", value: 70 } });
});

test("oscillators get their standard oversold / overbought levels", () => {
  expect(suggestCondition(rsi, "buy")).toMatchObject({
    left: { kind: "indicator", id: "rsi", params: { length: 14 } },
    op: "crosses_below",
    right: { kind: "constant", value: 30 },
  });
  expect(suggestCondition(rsi, "sell")).toMatchObject({ op: "crosses_above", right: { value: 70 } });
});

test("moving averages compare price to the line", () => {
  expect(suggestCondition(sma, "buy")).toMatchObject({
    left: { kind: "price", field: "close" },
    op: "crosses_above",
    right: { kind: "indicator", id: "sma", params: { length: 20 } },
  });
  expect(suggestCondition(sma, "sell")).toMatchObject({ op: "crosses_below" });
});

test("MACD uses its signal line", () => {
  expect(suggestCondition(macd, "buy")).toMatchObject({
    left: { kind: "indicator", id: "macd", output: "macd" },
    op: "crosses_above",
    right: { kind: "indicator", id: "macd", output: "signal" },
  });
});
