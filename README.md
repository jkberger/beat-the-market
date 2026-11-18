# Beat The Market

A trading research app. Build buy and sell rules from 25 technical indicators, then test them
honestly: on data the strategy never saw, with repeated trading in and out, next to simply
holding the stock and SPY.

> Research tool, not financial advice. Past results don't guarantee future returns.

## Start it

**Option 1: Docker (one command)**

```bash
docker compose up --build
```

Open http://localhost:8000.

**Option 2: Without Docker** (needs Node.js 20+; the script installs `uv` for Python if missing)

```bash
./dev.sh          # macOS / Linux
.\dev.ps1         # Windows PowerShell
```

Open http://localhost:5173. The API runs on :8000 and both reload when you edit code.

Price data comes from Yahoo Finance (free, unofficial, personal use only) and is cached in
`data/` for a day.

## What it does (phase 1)

- **Stock page:** candlestick chart with any of 25 indicators overlaid or in their own panes.
- **Strategy builder:** buy rules and sell rules built from indicator conditions (`is above`,
  `crosses above`, `is rising for N bars`, …) joined by AND / OR, plus optional stop loss,
  take profit, max holding period, and shorting on sell signals.
- **Risk tolerance:** Conservative, Balanced, Aggressive or Custom. Sets position size, stop,
  a drawdown limit, and what tuning optimizes for.
- **Honest testing:**
  - Signals fire at a day's close and fill at the **next day's open**.
  - **Walk-forward:** tune on all history before each test (or just the last 3 years), trade the next 6 months blind, slide forward, repeat. Many blind tests instead of one lucky one.
  - **Locked holdout:** the last 2 years (a fixed date in settings) are never used for tuning
    and are scored separately, with a count of how often you've peeked.
  - Every result is compared with buy-and-hold of the stock and of SPY over the same dates.
  - Commission is $0; optional slippage in basis points.

Coming next: mining indicator combos across popular stocks, per-stock outlook with suggested
stock and options strategies, and Alpaca paper trading. See
[the design spec](docs/superpowers/specs/2026-11-18-beat-the-market-rebuild-design.md).

## Develop

```bash
cd backend && uv run pytest        # backend tests
cd frontend && npm test            # frontend tests
```

| Folder | What's in it |
|---|---|
| `backend/app/data` | price providers and the Parquet cache |
| `backend/app/indicators` | the indicator registry |
| `backend/app/strategy` | rule model, evaluator, plain-English description, tuning grid, risk presets |
| `backend/app/backtest` | engine, metrics, walk-forward and holdout |
| `backend/app/api` | JSON routes under `/api` |
| `frontend/src` | React app (Vite, Tailwind, TanStack Query, lightweight-charts) |

Settings come from environment variables (see `.env.example`).
