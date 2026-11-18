import { render, screen } from "@testing-library/react";
import { TestedDates } from "@/components/TestedDates";
import type { WindowInfo } from "@/api/types";

const w = (ts: string, te: string, s: string, e: string): WindowInfo => ({
  window: { train_start: ts, train_end: te, test_start: s, test_end: e },
  chosen: {}, qualified: true,
  test: { total_return: 0, cagr: 0, volatility: 0, sharpe: null, sortino: null, max_drawdown: 0,
    calmar: null, win_rate: null, avg_win: null, avg_loss: null, profit_factor: null,
    num_trades: 0, time_in_market: 0 },
});

test("explains training, blind test windows and the holdout", () => {
  render(
    <TestedDates
      windows={[w("1980-12-12", "1983-12-09", "1983-12-12", "1984-06-08"),
                w("1981-06-12", "1984-06-08", "1984-06-11", "2024-11-15")]}
      holdout={{ start: "2024-11-18", end: "2026-11-17" }}
    />,
  );
  expect(screen.getByText(/2 windows/)).toBeInTheDocument();
  expect(screen.getByText(/first trained on 1980-12-12 to 1983-12-09/i)).toBeInTheDocument();
  expect(screen.getByText(/tested blind from 1983-12-12 to 2024-11-15/i)).toBeInTheDocument();
  expect(screen.getByText(/2024-11-18 to 2026-11-17/)).toBeInTheDocument();
});

test("says when the holdout was not scored", () => {
  render(<TestedDates windows={[w("2000-01-03", "2002-12-31", "2003-01-02", "2003-06-30")]} holdout={null} holdoutStart="2024-11-18" />);
  expect(screen.getByText(/not scored in this run/i)).toBeInTheDocument();
});
