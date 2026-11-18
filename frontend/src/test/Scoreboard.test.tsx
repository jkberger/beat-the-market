import { render, screen, within } from "@testing-library/react";
import type { Metrics, Segment } from "@/api/types";
import { Scoreboard } from "@/components/Scoreboard";
import { autoName, DEFAULT_STRATEGY } from "@/lib/defaults";

const m = (over: Partial<Metrics> = {}): Metrics => ({
  total_return: 0.5, cagr: 0.1, volatility: 0.2, sharpe: 0.8, sortino: 1.1, max_drawdown: -0.2,
  calmar: 0.5, win_rate: 0.55, avg_win: 0.04, avg_loss: -0.02, profit_factor: 1.4, num_trades: 42,
  time_in_market: 0.6, ...over,
});

const seg = (label: Segment["label"], metrics: Metrics): Segment => ({
  label, start: "2010-01-04", end: "2020-01-02", equity: [], trades: [], metrics,
  baseline_metrics: { stock: m({ cagr: 0.12, total_return: 2, num_trades: 1 }), spy: m({ cagr: 0.09, num_trades: 1 }) },
});

test("says in dollars what the strategy and holding did, and who won", () => {
  render(<Scoreboard symbol="AAPL" holdoutCount={null} startingAmount={10000} segments={{ out_of_sample: seg("out_of_sample", m()) }} />);
  expect(screen.getByText(/Did worse than just holding AAPL/)).toBeInTheDocument();
  expect(screen.getByText("$15,000.00")).toBeInTheDocument();
  expect(screen.getByText("$30,000.00")).toBeInTheDocument();
});

test("always compares strategy, holding the stock, and the market", () => {
  for (const symbol of ["AAPL", "SPY"]) {
    const { unmount } = render(
      <Scoreboard symbol={symbol} holdoutCount={null} startingAmount={10000} segments={{ backtest: seg("backtest", m({ cagr: 0.2 })) }} />,
    );
    expect(screen.getByText("This strategy")).toBeInTheDocument();
    expect(screen.getByText(`Just hold ${symbol}`)).toBeInTheDocument();
    expect(screen.getByText("The market (SPY)")).toBeInTheDocument();
    unmount();
  }
});

test("on SPY the market row is the same as holding SPY and says so", () => {
  render(<Scoreboard symbol="SPY" holdoutCount={null} startingAmount={10000} segments={{ backtest: seg("backtest", m()) }} />);
  const row = document.querySelector('[data-row="The market (SPY)"]') as HTMLElement;
  expect(within(row).getByText(/same as holding SPY/)).toBeInTheDocument();
  expect(within(row).getByText("12.0%")).toBeInTheDocument();
});

test("shows the final check with its scored count", () => {
  render(
    <Scoreboard symbol="AAPL" holdoutCount={1} startingAmount={10000}
      segments={{ out_of_sample: seg("out_of_sample", m()), holdout: seg("holdout", m({ sharpe: null })) }} />,
  );
  const box = document.querySelector("[data-holdout]") as HTMLElement;
  expect(within(box).getByText(/Scored 1 time\./)).toBeInTheDocument();
  expect(within(box).getAllByText("—").length).toBeGreaterThan(0);
});

test("names strategies after their rules", () => {
  expect(autoName(DEFAULT_STRATEGY)).toBe("RSI 14 strategy");
  const sma = (length: number) => ({ kind: "indicator" as const, id: "sma", params: { length }, output: "sma" });
  const cross = (op: "crosses_above" | "crosses_below") => ({
    kind: "group" as const, join: "and" as const,
    items: [{ kind: "condition" as const, left: sma(50), op, right: sma(200), bars: 1 }],
  });
  expect(autoName({ ...DEFAULT_STRATEGY, buy: cross("crosses_above"), sell: cross("crosses_below") })).toBe("SMA 50/200 cross");
});

test("drawdown is explained on the equity chart", async () => {
  const { DRAWDOWN_HELP } = await import("@/components/EquityChart");
  expect(DRAWDOWN_HELP).toMatch(/below its own best value/);
});
