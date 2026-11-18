# Beat The Market rebuild: design

Date: 2026-11-18
Status: ready for Jason's approval

## 1. Intent

Beat The Market becomes a trading research app. A trader picks indicators from a large library, combines them into buy and sell rules, and sees how those rules would really have performed on stocks they choose. The app also searches a basket of popular stocks for indicator combinations that held up, and for any single stock it reads the current trend and suggests the 3 to 5 best stock or options plays right now.

The first users are Jason and his brother, testing and playing with it. If it gets good enough, it may open to friends, family and paying customers, so the design keeps that path open without building it now.

The one quality that matters most is **honesty**. Every number the app shows must come from data the strategy or model never saw while it was built, after trading costs, next to a buy-and-hold baseline. A result that says "this does not beat buy-and-hold" is a correct, useful result. The app never promises to beat the market.

## 2. Why a rebuild

The current Flask app (about 1.7k lines, idle since February 2025) has problems that can't be patched:

- The model's labels come from the user's own rules on the same bar as the features, so it re-learns the rule and predicts nothing about future prices.
- The backtest runs on the same window the model trained on, with no costs.
- Live Alpaca trading always loads the "Custom" model, uses stale cached data and never updates starting equity.
- The `Templates/` folder casing breaks on Linux. There are no tests or CI.

The old code is removed in phase 1; git history keeps it.

## 3. Goals, non-goals, anti-goals

**Goals**
- One command to start it locally, and a free hosting path.
- A polished, inviting UI that feels like a modern trading app.
- A large indicator library and a rule builder that combines indicators.
- An honest backtester: next-bar execution, costs, walk-forward validation, a locked final holdout, baselines.
- Combo mining across a basket of popular stocks (phase 2).
- A per-stock outlook and 3 to 5 suggested stock or options strategies (phase 3).
- Paper trading through a broker API (phase 4).

**Non-goals (for now)**
- Real-money order execution.
- User accounts, billing, multi-tenant hosting.
- Intraday or high-frequency strategies (free intraday history is too short).
- Backtesting options strategies (no free historical options data).

**Anti-goals: it fails even if it "works" when**
- any shown result uses data the strategy was tuned or trained on, without saying so;
- any calculation uses information from the future (lookahead);
- costs are hidden or left out;
- the UI implies a guaranteed or expected edge over the market.

## 4. Constraints

- Free data only for now. Paid feeds (for example Polygon) may come later, so data sits behind a swappable provider layer.
- Yahoo data (via `yfinance`) is unofficial and not licensed for commercial use. It is fine for personal use and must be replaced before any public or paid launch.
- Must run on a laptop (macOS, Linux, Windows via Docker). The later hosted copy must fit a free tier (about 512 MB of memory, sleeps when idle).

## 5. Approaches considered

1. **Python API (FastAPI) + React web app.** Keeps Python's quant ecosystem, gives a real product UI, deploys free, grows into multi-user. **Chosen by Jason.**
2. **Streamlit, all Python.** Fastest to build, free hosting, but looks like a notebook and is awkward for multiple users. Rejected: dead end for customers.
3. **Next.js, all TypeScript.** Great UI, but indicators and backtesting would be rewritten by hand without Python's libraries. Rejected.

**Data and broker (chosen):** Yahoo for daily history and today's option chains; Alpaca (free) as a second data source and the paper trading account in phase 4. Tradier and Schwab have good free options data but need a funded brokerage account, so they are later options behind the same provider layer.

## 6. Architecture

```
frontend/  React + TypeScript (Vite)  ──HTTP/JSON──▶  backend/  FastAPI (Python 3.12)
                                                         ├─ data/        providers + local cache
                                                         ├─ indicators/  registry of indicators
                                                         ├─ strategy/    rule model + evaluator
                                                         ├─ backtest/    engine, costs, metrics, validation
                                                         ├─ mining/      (phase 2)
                                                         ├─ outlook/     (phase 3) forecast + options strategies
                                                         └─ broker/      (phase 4) paper trading adapter
```

**Backend:** Python 3.12, FastAPI, Pydantic v2, pandas and NumPy, managed with `uv`. Tests with pytest. Lint and format with ruff.

**Frontend:** React, TypeScript, Vite, Tailwind CSS with shadcn/ui components, TanStack Query for API calls, TradingView `lightweight-charts` for price charts with indicator overlays and trade markers. Tests with Vitest.

**Storage:** downloaded price data cached as Parquet files per symbol on disk, refreshed when older than one trading day. Saved strategies, backtest results, settings (including the holdout date) and holdout score counts are stored through SQLAlchemy, on SQLite locally. Because the database sits behind SQLAlchemy, the later hosted copy switches to a free managed Postgres (for example Neon) by changing one connection string.

**Running it (Jason's decision: local-first):**
- Phase 1 targets Jason's own computer. `docker compose up` starts the app at `http://localhost:8000` (the backend serves the built frontend).
- Without Docker: one script (`./dev.sh`, plus `dev.ps1` for Windows) installs dependencies and starts backend and frontend with hot reload.
- **Later, one shared hosted copy** that Jason and his brother reach from anywhere, kept free as long as possible: a single Docker image on Render's free tier (or Fly.io), free managed Postgres for persistent data, and a password gate in front because there are no accounts yet. Heavy jobs (mining, scheduled paper trading) stay local until the hosted copy has a worker that can run them. Hosting is its own small phase after phase 1; phase 1 only has to keep it possible (env-based config, no local-only paths in the database, a Dockerfile).

**CI:** GitHub Actions runs backend tests, frontend tests, lint and a type check on every push and pull request.

## 7. Data layer

- `PriceProvider` interface: `get_daily_bars(symbol, start, end) -> DataFrame` with columns open, high, low, close, adjusted close, volume, indexed by date. Prices adjusted for splits and dividends.
- `OptionsProvider` interface: `get_expirations(symbol)` and `get_chain(symbol, expiration)` returning strikes, bid, ask, last, volume, open interest, implied volatility, and greeks if the source gives them.
- Implementations: `YahooProvider` (both interfaces) in phase 1; `AlpacaProvider` in phase 3 or 4. Which provider is used is a config setting.
- Daily bars only in phase 1. Yahoo gives decades of daily history for large stocks (AMZN back to 1997, SPY to 1993). Intraday is capped (5-minute bars about 60 days, hourly about 2 years) and is out of scope.
- Default universe: a curated list of about 50 large, liquid US stocks and ETFs (for example SPY, QQQ, AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM), stored as a config file and editable. Quick-pick defaults in the UI: SPY, AAPL, MSFT, NVDA, AMZN.
- Survivorship bias: a list of today's popular stocks excludes companies that failed. The UI notes this on mining results.

## 8. Indicator library

- A registry where each indicator declares: id, display name, category (trend, momentum, volatility, volume), parameters with defaults and allowed ranges, output series names, and whether it overlays the price chart or gets its own pane.
- Each indicator is a pure function of past and current bars only. A shared test checks every registered indicator for lookahead: values up to day *t* must not change when data after *t* is appended.
- Phase 1 ships at least these 25: SMA, EMA, WMA, Hull MA, VWMA, MACD, RSI, Stochastic, Stochastic RSI, Williams %R, CCI, ROC, Momentum, ADX/DMI, Aroon, Parabolic SAR, Ichimoku, Supertrend, Bollinger Bands, Keltner Channels, Donchian Channels, ATR, OBV, Chaikin Money Flow, MFI. More can be added by registering a function.
- Implementation source is the builder's call: the `ta` library where it is correct, otherwise hand-written vectorized pandas. Each indicator is checked against known reference values in tests.

## 9. Strategy rules

A strategy is a saved JSON document:

- **Entry rule** and **exit rule**, each a tree of conditions joined by AND or OR.
- A **condition** compares two operands with one of: `>`, `<`, `crosses above`, `crosses below`, `is rising for N bars`, `is falling for N bars`. An operand is an indicator output (with parameters), price (open, high, low, close), or a constant.
- Optional **stop loss** and **take profit** as a percentage or a multiple of ATR, and an optional **max holding period** in bars.
- **Direction:** long-only in phase 1. Short selling is a later option.
- **Position sizing:** all-in (100% of equity) in phase 1; fixed fraction is a later option.
- **Tunable parameters (Claude's call):** any numeric parameter (an indicator period, a threshold, a stop size) can be fixed or marked as a range with a step, for example RSI period 7 to 21 step 7. At most 3 parameters can be ranges in phase 1, capped at 200 combinations. Tuning is a grid search whose objective is Sharpe ratio after costs, with a minimum of 5 trades per training window so a rule that barely trades can't win.
- **One stock per backtest (Claude's call):** a backtest runs one strategy on one stock. Picking several stocks runs the same strategy on each and shows a comparison table (per-stock metrics plus median across stocks). Portfolio backtests across tickers are a later option.

The UI builds these with dropdowns ("RSI(14) crosses above 30 AND close > SMA(200)") and shows a plain-English sentence of the rule.

## 10. Backtest engine

- **No lookahead:** a signal is computed from a bar's close and filled at the **next bar's open**. Stops and targets are checked against the next bars' high and low, assuming the worse case if both are hit on the same bar.
- **Costs**, all editable, with defaults: commission $0, slippage 5 basis points per side, plus half the typical spread (another 2 basis points per side for large caps). Costs are always on by default.
- **Outputs:** equity curve, trade list (entry and exit date and price, return, holding days), and metrics: total return, CAGR, annualized volatility, Sharpe, Sortino, max drawdown, Calmar, win rate, average win and loss, profit factor, number of trades, time in market.
- **Baselines on every result:** buy-and-hold of the same stock, and buy-and-hold of SPY, over the same dates with the same costs on entry.
- Engine is vectorized over the data where possible and must backtest 20 years of daily bars for one stock in under one second.

## 11. Validation: walk-forward plus a locked holdout

This is the core of being honest, and it applies to anything that is tuned (strategy parameters in phase 1, mined combos in phase 2, forecast models in phase 3).

1. **Split off the final holdout first.** The holdout is a **fixed start date** shared by the whole app (Claude's call), not "the latest 20%", so new daily data never leaks old holdout days into training. On first run it is set to 2 years before that day and stored in the settings table; everything from that date onward is holdout for every strategy, every stock, mining and the outlook model. It is never used while choosing parameters, combos or models. Moving it forward is a deliberate settings action that the UI warns about.
2. **Walk-forward on the rest.** Rolling windows: tune on a training window (default 3 years), then test on the next window (default 6 months), roll forward by the test length and repeat. The stitched-together test windows form the **out-of-sample** result. Parameters are chosen using only each window's training data.
3. **Score the holdout once** with the final chosen settings. The app shows it as a separate, clearly labeled "Holdout (never seen)" result, and records how many times the holdout has been scored for a given strategy. A strategy's identity for this count is its rule structure (indicators, conditions, joins), ignoring parameter values, so tweaking a number and re-scoring still adds to the same count. Re-scoring after changes is allowed but the count is shown, because peeking repeatedly turns the holdout into training data.
4. **Untuned strategies:** if the user just runs fixed rules without tuning, a plain backtest over any range is honest, and the app shows it without the walk-forward steps but still with costs and baselines.
5. **Labels:** every chart and metric is tagged "in-sample", "out-of-sample (walk-forward)" or "holdout". In-sample numbers are only shown for diagnostics and are visually de-emphasized.

## 12. UI

Dark theme by default, with a light theme toggle. Clean, spacious layout; green and red used only for gains and losses; one accent color. Plain-English labels with short tooltips explaining every metric.

Phase 1 screens:

- **Home / dashboard:** quick-pick tickers, a search box, recent backtests, saved strategies.
- **Stock page:** price chart (candles, volume) with chosen indicators overlaid; key stats; buttons to backtest or (phase 3) get suggestions.
- **Strategy builder:** indicator picker by category, condition rows for entry and exit, stops, costs, date range, validation mode (simple or walk-forward plus holdout), and a live plain-English summary.
- **Backtest results:** equity curve against both baselines, drawdown chart, price chart with buy and sell markers, metrics table with labeled sections, trade list. Results can be saved and compared side by side.

```
┌ Beat The Market ─────────────── [search ticker] ── ☾ ┐
│ AAPL  $231.40  +1.2%                                 │
│ ┌ price chart + SMA(50) + markers ────────────────┐  │
│ │                                                 │  │
│ └─────────────────────────────────────────────────┘  │
│ Strategy  RSI(14) crosses above 30 AND close > SMA200 │
│ ┌ Out-of-sample ┐ ┌ Holdout (never seen) ┐ ┌ Buy&hold ┐│
│ │ CAGR 9.1%     │ │ CAGR 6.4%            │ │ 12.3%    ││
│ │ MaxDD -18%    │ │ MaxDD -11%           │ │ -31%     ││
│ └───────────────┘ └──────────────────────┘ └──────────┘│
└──────────────────────────────────────────────────────┘
```

A short, non-alarming disclaimer sits in the footer: research tool, not financial advice, past results don't guarantee future ones.

## 13. Phase 2: combo mining

- The user picks a universe (default: the curated list), a date range, and a search space: which indicators, parameter ranges, and how many conditions per rule (default up to 3).
- The miner generates candidate rules (grid or random search, capped by a trial budget, default 2,000), runs walk-forward on every stock, and ranks candidates by out-of-sample performance **across stocks** (median Sharpe and the share of stocks where it beat buy-and-hold), not by the best single stock.
- **Multiple-testing guard:** testing thousands of rules guarantees some look great by luck. Rankings report the number of trials and use a deflated Sharpe ratio (or an equivalent adjustment) so lucky rules are discounted. Only the top few are then scored on the holdout.
- Runs as a background job with progress shown in the UI; results saved and browsable; any result opens in the strategy builder.

## 14. Phase 3: outlook and strategy suggestions

- **Outlook model:** forecasts the distribution of the stock's return over a chosen horizon (default 20 trading days), giving probability of going up and expected range. Target is the actual future return (never the user's rules). Features are indicator values at time *t* only. Model is a gradient-boosted model (for example LightGBM) or quantile regression, trained with walk-forward and scored on the holdout. The app shows its track record: hit rate, calibration (does "70% likely" happen about 70% of the time) and comparison to a naive baseline. If the model is no better than the baseline, the app says so and leans on the market-implied view instead.
- **Market-implied view:** implied volatility from today's option chain gives the market's expected range. The outlook is shown next to it.
- **Candidate strategies:** long stock, long call, long put, covered call, cash-secured put, bull call spread, bear put spread, iron condor, long straddle or strangle. Each is built from real strikes and expirations in today's chain.
- **Scoring:** for each candidate, compute payoff at expiration, max profit, max loss, breakevens, probability of profit and expected value under the outlook distribution, capital required, and greeks (Black-Scholes from implied volatility where the source doesn't give them). Rank by expected value per dollar at risk, filtered by the user's risk preference, and show the top 3 to 5 with payoff diagrams and a plain-English "why".
- Options suggestions cannot be backtested with free data; the UI says so on every suggestion.

## 15. Phase 4: paper trading

- `Broker` interface (account, positions, place order, cancel, order status) with an Alpaca paper trading implementation; keys in a local `.env`, never committed.
- A saved strategy can be attached to a paper account; a scheduled job evaluates it after each daily close and places orders for the next open, logging every decision.
- Real-money trading stays out of scope until explicitly decided.

## 16. Testing

- Unit tests for every indicator (reference values plus the lookahead check), the rule evaluator, costs, metrics and validation splits.
- A **lookahead canary** test: a strategy given access to tomorrow's return would score impossibly well; the engine must make that impossible by construction (the test feeds a rule built on a future-shifted series through the public API and asserts it is rejected or yields no edge).
- Golden backtests: small hand-computed datasets with known trades and returns.
- API tests with FastAPI's test client; data provider tests use recorded fixtures, never live network.
- Frontend component tests for the rule builder and results views.

## 17. Decisions left to the builder

- Exact folder layout inside `backend/` and `frontend/`.
- Which indicators come from the `ta` library versus hand-written code.
- Visual details: accent color, fonts, spacing (within the direction in section 12).
- Whether background jobs (phase 2) use a simple in-process worker or a task queue; start with in-process.
