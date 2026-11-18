import { render, screen, within } from "@testing-library/react";
import type { Metrics, Segment } from "@/api/types";
import { MetricsCards } from "@/components/MetricsCards";

const m = (over: Partial<Metrics> = {}): Metrics => ({
  total_return: 0.5, cagr: 0.1, volatility: 0.2, sharpe: 0.8, sortino: 1.1, max_drawdown: -0.2,
  calmar: 0.5, win_rate: 0.55, avg_win: 0.04, avg_loss: -0.02, profit_factor: 1.4, num_trades: 42,
  time_in_market: 0.6, ...over,
});

const seg = (label: Segment["label"], metrics: Metrics): Segment => ({
  label, start: "2020-01-01", end: "2021-01-01", equity: [], trades: [], metrics,
  baseline_metrics: { stock: m({ cagr: 0.12, num_trades: 1 }), spy: m({ cagr: 0.09, num_trades: 1 }) },
});

test("labels segments, shows holdout count, and dashes for null", () => {
  render(
    <MetricsCards
      symbol="AAPL"
      holdoutCount={3}
      segments={{ out_of_sample: seg("out_of_sample", m({ sharpe: null })), holdout: seg("holdout", m()) }}
    />,
  );
  const oos = screen.getByText("Out-of-sample (walk-forward)").closest("[data-card]") as HTMLElement;
  expect(oos).toBeInTheDocument();
  expect(within(oos).getByTestId("sharpe")).toHaveTextContent("—");
  expect(screen.getByText("Holdout (never seen)")).toBeInTheDocument();
  expect(screen.getByText("Scored 3 times")).toBeInTheDocument();
  expect(screen.getByText("Buy & hold AAPL")).toBeInTheDocument();
  expect(screen.getByText("Buy & hold SPY")).toBeInTheDocument();
});

test("does not show the scored count for a single scoring", () => {
  render(<MetricsCards symbol="AAPL" holdoutCount={1} segments={{ holdout: seg("holdout", m()) }} />);
  expect(screen.queryByText(/Scored/)).not.toBeInTheDocument();
});

test("shows a verdict against buy and hold", () => {
  render(<MetricsCards symbol="AAPL" holdoutCount={null} segments={{ backtest: seg("backtest", m({ cagr: 0.1 })) }} />);
  expect(screen.getByText(/trailed buy & hold AAPL/i)).toBeInTheDocument();
});
