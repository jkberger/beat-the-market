import type { Metrics } from "@/api/types";

export const METRIC_HELP: Partial<Record<keyof Metrics, string>> = {
  cagr: "Compound annual growth rate: the steady yearly return that would produce the same result.",
  total_return: "How much the account grew over the whole period.",
  max_drawdown: "The worst drop from a peak to a later low. Smaller is easier to live with.",
  sharpe: "Return per unit of volatility, annualized. Above 1 is good; below 0.5 is weak.",
  sortino: "Like Sharpe, but only counts downside volatility.",
  num_trades: "Completed round trips (buy then sell).",
  win_rate: "Share of trades that made money.",
  profit_factor: "Total gains divided by total losses across trades. Above 1 means winners outweigh losers.",
  time_in_market: "Share of days holding a position.",
  volatility: "Annualized standard deviation of daily returns.",
};
