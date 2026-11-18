# Phase 1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old Flask app with a FastAPI + React app that has a 25-indicator library, a buy/sell rule builder, an honest repeated-trading backtester with risk tolerance presets, walk-forward tuning and a locked holdout, all started with one command.

**Architecture:** `backend/` is a FastAPI package (`app`) with pure-Python modules for data, indicators, strategy rules, backtesting and validation, plus SQLAlchemy persistence and JSON routes under `/api`. `frontend/` is a Vite React TypeScript app that talks to `/api`; in production the backend serves the built frontend from `frontend/dist`.

**Tech Stack:** Python 3.12+, uv, FastAPI, Pydantic v2, pandas, NumPy, `ta`, yfinance, pyarrow, SQLAlchemy 2, pytest, ruff. React 18, TypeScript, Vite, Tailwind CSS, shadcn/ui, TanStack Query, React Router, lightweight-charts, Vitest, Testing Library.

**Spec:** `docs/superpowers/specs/2026-11-18-beat-the-market-rebuild-design.md`

## Global Constraints

- Daily bars only. Signals are computed from a bar's close and filled at the **next bar's open**. Nothing may read data after the bar it is computing.
- Costs off by default: commission $0, `slippage_bps` default 0, editable.
- Repeated trading: flat → buy signal → long → sell signal → flat (or short if `allow_short`) → next buy, as many times as signals say.
- Risk presets (editable defaults): Conservative = 25% position, stop 2×ATR(14), stand aside 20 bars after a 10% drawdown, objective Sortino; Balanced = 50%, 3×ATR(14), 20%, Sharpe; Aggressive = 100%, no stop, no drawdown limit, CAGR.
- Tuning: at most 3 ranged parameters, at most 200 combinations, at least 5 trades per training window, objective from the risk preset.
- Walk-forward defaults: 3-year training window, 6-month test window, roll by the test length.
- Holdout: one app-wide fixed start date stored in settings, initialized to today minus 2 years on first run; never used for tuning; score count tracked per strategy structure (ignoring parameter values) and symbol.
- Every result shows buy-and-hold of the same stock and of SPY over the same dates, and labels each number "in-sample", "out-of-sample" or "holdout".
- Quick picks: SPY, AAPL, MSFT, NVDA, AMZN. Universe: about 50 large US stocks and ETFs in `backend/app/data/universe.json`.
- Configuration comes from env vars prefixed `BTM_` (`BTM_DATA_DIR`, `BTM_DATABASE_URL`, `BTM_PRICE_PROVIDER`); no absolute local paths are stored in the database.
- Dark theme by default with a light toggle; green and red only for gains and losses; one accent color. Footer disclaimer: "Research tool, not financial advice. Past results don't guarantee future returns."
- Tests never hit the network: provider tests use recorded fixtures.
- JSON responses never contain NaN or Infinity; they become `null`.

## Review Focus

1. **Typo or unknown ticker** (`AAPLL`): API returns 404 `{"detail": "Unknown symbol: AAPLL"}` and the UI shows that message, not a crash. Test in Task 2 and Task 9.
2. **Too little history for walk-forward** (a stock listed 2 years ago with a 3-year training window): API returns 422 with a message naming the years needed; simple mode still works. Test in Task 7.
3. **A rule that never trades or never loses**: metrics return 0 trades, and `profit_factor` and `win_rate` come back `null`, not NaN or Infinity, so the JSON still parses. Test in Task 6.
4. **Buy and sell signal on the same bar**: when flat, no trade; when in a position, the opposite signal wins (long exits on sell). Test in Task 5.
5. **Yahoo down or returning blank rows**: rows with NaN prices are dropped; if the fetch fails and a cache exists, the cache is used. Test in Task 2.

---

### Task 1: Backend scaffold and removal of the old app

**Files:**
- Delete: `trading_app.py`, `utils/`, `Templates/`, `requirements.txt`
- Create: `backend/pyproject.toml`, `backend/app/__init__.py`, `backend/app/main.py`, `backend/app/config.py`, `backend/tests/test_health.py`, `.gitignore`

**Interfaces:**
- Produces: `app.main.create_app() -> FastAPI`; `app.main.app`; `app.config.Settings` (pydantic-settings, env prefix `BTM_`, fields `data_dir: Path = Path("data")`, `database_url: str = "sqlite:///data/btm.db"`, `price_provider: str = "yahoo"`); `app.config.get_settings() -> Settings` (cached).

- [ ] **Step 1:** Remove the old files with `git rm -r`, create `backend/pyproject.toml` (requires-python `>=3.12`, deps above, dev group pytest, httpx, ruff), and `.gitignore` covering `data/`, `.venv/`, `node_modules/`, `frontend/dist/`, `.env`.
- [ ] **Step 2: Write the failing test** `test_health_returns_ok`: `TestClient(create_app()).get("/api/health")` returns 200 and `{"status": "ok"}`.
- [ ] **Step 3:** Run `cd backend && uv run pytest tests/test_health.py -v`. Expected: FAIL (import error).
- [ ] **Step 4:** Implement `create_app()` with the health route. If `../frontend/dist` exists, mount it at `/` with HTML fallback to `index.html`.
- [ ] **Step 5:** Run the test and `uv run ruff check .`. Expected: PASS, no lint errors.
- [ ] **Step 6:** Commit `chore: replace Flask app with FastAPI scaffold`.

### Task 2: Data layer

**Files:**
- Create: `backend/app/data/providers.py`, `backend/app/data/cache.py`, `backend/app/data/universe.py`, `backend/app/data/universe.json`, `backend/tests/data/test_cache.py`, `backend/tests/data/test_yahoo.py`, `backend/tests/fixtures/aapl_daily.csv`, `backend/tests/conftest.py`

**Interfaces:**
- Produces:
  - `class SymbolNotFound(Exception)` with `.symbol`.
  - `class PriceProvider(Protocol): def get_daily_bars(self, symbol: str, start: date | None = None, end: date | None = None) -> pd.DataFrame`. The frame has columns `open, high, low, close, adj_close, volume`, a tz-naive ascending `DatetimeIndex` named `date`, and split/dividend-adjusted OHLC (`adj_close == close`).
  - `class YahooProvider(PriceProvider)` using `yf.Ticker(symbol).history(period="max", auto_adjust=True)`. An empty result raises `SymbolNotFound`. Rows with NaN prices are dropped.
  - `class CachedPriceProvider(PriceProvider)` with `__init__(self, inner: PriceProvider, cache_dir: Path, max_age: timedelta = timedelta(days=1))`. It stores the full history per symbol as `<cache_dir>/<SYMBOL>.parquet`, refetches when the file is older than `max_age`, falls back to a stale cache when the inner call raises anything except `SymbolNotFound`, and slices to `start`/`end`.
  - `get_price_provider() -> PriceProvider` (cached provider from settings).
  - `load_universe() -> dict` returning `{"symbols": list[str], "quick_picks": list[str]}`.
  - Test fixture `bars` in `conftest.py`: about 10 years of AAPL daily bars loaded from the CSV fixture. The fixture is recorded once from Yahoo with a helper script `backend/scripts/record_fixture.py`, then committed.

- [ ] **Step 1: Write failing tests:**
  - `test_cache_uses_inner_once_within_max_age`: a fake inner provider counts calls; two `get_daily_bars("AAPL")` calls → 1 inner call.
  - `test_cache_refetches_when_stale`: set the file mtime 2 days back → 2 inner calls.
  - `test_cache_falls_back_when_inner_fails`: prime the cache, make inner raise `ConnectionError`, age the file → the cached frame is returned.
  - `test_cache_propagates_symbol_not_found`.
  - `test_slices_start_end`: the returned index lies within `[start, end]`.
  - `test_yahoo_drops_nan_rows_and_normalizes` (monkeypatch `yf.Ticker` to return a frame with tz-aware index, capitalized columns and one NaN row) → lowercase columns, tz-naive index, NaN row removed.
  - `test_yahoo_unknown_symbol_raises` (empty frame → `SymbolNotFound`).
  - `test_universe_has_quick_picks`: `quick_picks == ["SPY","AAPL","MSFT","NVDA","AMZN"]` and `len(symbols) >= 45`.
- [ ] **Step 2:** Run `uv run pytest tests/data -v`. Expected: FAIL.
- [ ] **Step 3:** Implement the interfaces above and record the fixture.
- [ ] **Step 4:** Run `uv run pytest tests/data -v`. Expected: PASS.
- [ ] **Step 5:** Commit `feat: price data providers with parquet cache`.

### Task 3: Indicator library

**Files:**
- Create: `backend/app/indicators/registry.py`, `backend/app/indicators/trend.py`, `momentum.py`, `volatility.py`, `volume.py`, `backend/app/indicators/__init__.py` (imports all modules so they register), `backend/tests/indicators/test_registry.py`, `backend/tests/indicators/test_values.py`

**Interfaces:**
- Produces:
  - `@dataclass Param(name: str, kind: Literal["int","float"], default: float, min: float, max: float)`.
  - `@dataclass IndicatorSpec(id: str, name: str, category: Literal["trend","momentum","volatility","volume"], params: list[Param], outputs: list[str], overlay: bool, fn: Callable[..., pd.DataFrame])`.
  - `register(spec: IndicatorSpec) -> None`, `REGISTRY: dict[str, IndicatorSpec]`.
  - `compute(indicator_id: str, bars: pd.DataFrame, params: dict[str, float] | None = None) -> pd.DataFrame`, with columns equal to `spec.outputs`, index equal to `bars.index`, and defaults filled in. An unknown id or an out-of-range param raises `ValueError`.
  - Ids and outputs (exact): `sma[sma]`, `ema[ema]`, `wma[wma]`, `hma[hma]`, `vwma[vwma]`, `macd[macd,signal,hist]`, `rsi[rsi]`, `stoch[k,d]`, `stochrsi[k,d]`, `willr[willr]`, `cci[cci]`, `roc[roc]`, `mom[mom]`, `adx[adx,plus_di,minus_di]`, `aroon[up,down]`, `psar[psar]`, `ichimoku[conversion,base,span_a,span_b]`, `supertrend[supertrend,direction]`, `bbands[upper,middle,lower]`, `keltner[upper,middle,lower]`, `donchian[upper,middle,lower]`, `atr[atr]`, `obv[obv]`, `cmf[cmf]`, `mfi[mfi]`.
  - The default length param is named `length` wherever an indicator has a single window.
  - Ichimoku spans are **not** shifted forward, because that would be lookahead. They use the values known at bar t.

- [ ] **Step 1: Write failing tests:**
  - `test_registry_has_all_25_ids` (the exact id set above).
  - `test_every_indicator_has_no_lookahead`, parametrized over `REGISTRY`: compute on `bars.iloc[:1500]` and on full `bars`. The first 1500 rows must be equal, with NaN equal to NaN (`pd.testing.assert_frame_equal`).
  - `test_outputs_match_spec`, parametrized.
  - `test_out_of_range_param_raises`: `rsi` with `length=0`.
  - `test_reference_values`: on a tiny hand-made series, `sma(length=3)` of `[1,2,3,4,5]` gives `[nan,nan,2,3,4]`. `rsi` and `macd`, `atr` and `bbands` must match the `ta` library on the fixture to within 1e-8.
- [ ] **Step 2:** Run `uv run pytest tests/indicators -v`. Expected: FAIL.
- [ ] **Step 3:** Implement. Use `ta` for indicators it has, with `fillna=False`. Hand-write HMA, VWMA, Supertrend and (unshifted) Ichimoku in pandas.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat: indicator registry with 25 indicators`.

### Task 4: Strategy rules, evaluator, description and tuning grid

**Files:**
- Create: `backend/app/strategy/models.py`, `backend/app/strategy/evaluate.py`, `backend/app/strategy/describe.py`, `backend/app/strategy/grid.py`, `backend/app/strategy/risk.py`, `backend/tests/strategy/test_evaluate.py`, `test_describe.py`, `test_grid.py`, `test_risk.py`

**Interfaces:**
- Consumes: `compute` from Task 3.
- Produces (Pydantic v2 models, discriminated on `kind`):
  - `RangeSpec(min: float, max: float, step: float)`; `ParamValue = float | RangeSpec`.
  - `PriceOperand(kind="price", field: Literal["open","high","low","close"])`; `IndicatorOperand(kind="indicator", id: str, params: dict[str, ParamValue] = {}, output: str)`; `ConstantOperand(kind="constant", value: ParamValue)`; `Operand` = union.
  - `Condition(kind="condition", left: Operand, op: Literal[">","<","crosses_above","crosses_below","rising","falling"], right: Operand | None = None, bars: int = 1)`. `right` is required for the first four ops; `bars` is used by `rising` and `falling`.
  - `Group(kind="group", join: Literal["and","or"], items: list[Condition | Group])`, with at least 1 item.
  - `StopSpec(kind: Literal["percent","atr"], value: ParamValue)`.
  - `Strategy(name: str, buy: Group, sell: Group, stop_loss: StopSpec | None = None, take_profit: StopSpec | None = None, max_hold_bars: int | None = None, allow_short: bool = False)`.
  - `RiskSettings(preset: Literal["conservative","balanced","aggressive","custom"], position_pct: float, stop_atr_mult: float | None, max_drawdown_pct: float | None, standaside_bars: int = 20, objective: Literal["sortino","sharpe","cagr"])`; `RISK_PRESETS: dict[str, RiskSettings]` with the Global Constraints values.
  - `BacktestConfig(initial_equity: float = 10_000, slippage_bps: float = 0.0)`.
  - `evaluate_group(group: Group, bars: pd.DataFrame) -> pd.Series` (bool, NaN-driven comparisons are False); `signals(strategy: Strategy, bars) -> tuple[pd.Series, pd.Series]` (buy, sell). Concrete strategies only: a `RangeSpec` raises `ValueError`.
  - `describe(strategy: Strategy) -> str`, e.g. `"Buy when RSI(14) crosses above 30 and Close > SMA(200). Sell when RSI(14) crosses below 70."`
  - `expand_grid(strategy: Strategy) -> list[Strategy]` (Cartesian product of ranges, inclusive of max). More than 3 ranges or more than 200 combos raises `ValueError`. With no ranges it returns `[strategy]`.
  - `structure_key(strategy: Strategy) -> str`: a sha256 of the JSON with all numeric param, constant and stop values removed.

- [ ] **Step 1: Write failing tests:**
  - `test_crosses_above`: series `[1,2,3]` vs constant 2.5 → `[F,F,T]`.
  - `test_crosses_below`.
  - `test_rising_for_n_bars`: `[1,2,3,4]`, bars=2 → `[F,F,T,T]`.
  - `test_and_or_groups`.
  - `test_nan_compares_false` (SMA(200) warmup bars are False).
  - `test_range_in_signals_raises`.
  - `test_describe_matches_example` (the exact sentence above).
  - `test_expand_grid_counts`: RSI length 7..21 step 7 × threshold 25..35 step 5 = 9 strategies.
  - `test_expand_grid_rejects_4_ranges`.
  - `test_expand_grid_rejects_over_200`.
  - `test_structure_key_ignores_params_but_not_indicators`.
  - `test_presets_values` (the three rows of the table).
- [ ] **Step 2:** Run `uv run pytest tests/strategy -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat: buy/sell rule model, evaluator and tuning grid`.

### Task 5: Backtest engine with repeated trades

**Files:**
- Create: `backend/app/backtest/engine.py`, `backend/tests/backtest/test_engine.py`

**Interfaces:**
- Consumes: `Strategy`, `RiskSettings`, `BacktestConfig`, `signals` (Task 4), `compute("atr")` (Task 3).
- Produces:
  - `@dataclass Trade(direction: Literal["long","short"], entry_date: date, entry_price: float, exit_date: date, exit_price: float, return_pct: float, bars_held: int, exit_reason: Literal["signal","stop","target","max_hold","drawdown","end"])`.
  - `@dataclass EngineResult(equity: pd.Series, position: pd.Series, trades: list[Trade])`. `position` takes values in {-1, 0, 1} per bar.
  - `run_engine(bars: pd.DataFrame, buy: pd.Series, sell: pd.Series, strategy: Strategy, risk: RiskSettings, config: BacktestConfig, start: date | None = None) -> EngineResult`. Signals before `start` are ignored, but the data before it is still available for indicator warmup.
  - `backtest(bars, strategy, risk, config, start=None, end=None) -> EngineResult` (computes signals on bars up to `end`, then calls `run_engine`).

Rules the tests pin down:
- A signal at bar t fills at `open[t+1]`. Slippage moves the fill against the trader by `slippage_bps/1e4`.
- Flat + buy → long. Long + sell → exit; with `allow_short` the same open also enters a short. Short + buy → cover, then go long. Flat + buy and sell on the same bar → nothing. In a position, an opposite signal on the same bar as a same-side signal still exits.
- Position size is `risk.position_pct` of current equity at entry. The rest stays in cash at 0%.
- Stop: `strategy.stop_loss` if set, else `risk.stop_atr_mult × ATR(14)` measured at the signal bar. Target: `strategy.take_profit`. Each bar from the entry bar on: if the stop is hit (low ≤ stop for longs), the fill is `min(open, stop)`. If the target is hit, the fill is `max(open, target)`. If both are hit on the same bar, the stop wins.
- Max hold: exit at the open after `max_hold_bars` bars.
- Drawdown limit: when equity falls `max_drawdown_pct` below its running peak, exit at the next open (`exit_reason="drawdown"`) and ignore entries for `standaside_bars` bars. After that, reset the peak to the current equity.
- Equity is marked at each close. An open position at the last bar closes at that close with `"end"`.

- [ ] **Step 1: Write failing tests**, using hand-built 10 to 20 bar frames with known opens and closes:
  - `test_round_trips_repeat`: buy signals at bars 1 and 6, sell signals at bars 3 and 8 → 2 trades, entries at open[2] and open[7], exits at open[4] and open[9].
  - `test_fill_is_next_open_not_close`.
  - `test_same_bar_buy_and_sell_when_flat_does_nothing`.
  - `test_allow_short_flips_on_sell`.
  - `test_position_pct_scales_pnl`: 50% size → half the equity change of 100%.
  - `test_stop_hit_exits_at_stop_or_gap_open`.
  - `test_stop_wins_when_both_hit`.
  - `test_take_profit`.
  - `test_max_hold`.
  - `test_drawdown_stand_aside`.
  - `test_slippage_worsens_fills`.
  - `test_end_closes_open_position`.
  - `test_lookahead_canary`: build a buy signal from `close.shift(-1) > close` and pass it straight to `run_engine` on the fixture. Assert that the Sharpe ratio of the result is far below that of a strategy that trades at the same close. That is, filling at the next open means a "knows tomorrow's close" signal earns less than its naive in-close version. Separately, assert that `signals()` cannot express a negative shift, because operands carry no shift field.
  - `test_20_years_under_one_second`: 5,000 synthetic bars, SMA cross strategy, under 1.0 s.
- [ ] **Step 2:** Run `uv run pytest tests/backtest/test_engine.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `run_engine` as a single pass over bars with NumPy arrays. The signals are already vectorized.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat: repeated buy/sell backtest engine with risk controls`.

### Task 6: Metrics and baselines

**Files:**
- Create: `backend/app/backtest/metrics.py`, `backend/tests/backtest/test_metrics.py`

**Interfaces:**
- Produces:
  - `compute_metrics(equity: pd.Series, trades: list[Trade], position: pd.Series) -> dict[str, float | int | None]`. The keys are exactly: `total_return, cagr, volatility, sharpe, sortino, max_drawdown, calmar, win_rate, avg_win, avg_loss, profit_factor, num_trades, time_in_market`. Returns are fractions (0.12 = 12%). Annualization uses 252 trading days and a 0% risk-free rate. Any undefined or infinite value becomes `None`.
  - `buy_and_hold(bars: pd.DataFrame, config: BacktestConfig, start: date | None = None, end: date | None = None) -> EngineResult`: buy at the first open in range, hold, and close at the last close.
  - `objective_value(metrics: dict, objective: str) -> float`: `sortino` is the Sortino ratio minus `abs(max_drawdown)`, `sharpe` is Sharpe, and `cagr` is CAGR. `None` becomes `-inf`.

- [ ] **Step 1: Write failing tests:**
  - `test_total_return_and_cagr`: equity doubles over 504 bars → total 1.0, CAGR ≈ 0.4142.
  - `test_max_drawdown`: `[100,120,90,130]` → -0.25.
  - `test_no_trades_gives_none_ratios`: `num_trades == 0`, `win_rate is None`, `profit_factor is None`.
  - `test_no_losses_profit_factor_none`.
  - `test_buy_and_hold_matches_price_return`.
  - `test_objective_value_per_preset`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat: performance metrics and buy-and-hold baseline`.

### Task 7: Walk-forward tuning and holdout

**Files:**
- Create: `backend/app/backtest/validation.py`, `backend/tests/backtest/test_validation.py`

**Interfaces:**
- Consumes: `expand_grid` (Task 4), `backtest` (Task 5), `compute_metrics` and `objective_value` (Task 6).
- Produces:
  - `class NotEnoughHistory(ValueError)`.
  - `@dataclass Window(train_start: date, train_end: date, test_start: date, test_end: date)`.
  - `walk_forward_windows(index: pd.DatetimeIndex, train_years: int = 3, test_months: int = 6) -> list[Window]`. Windows roll by `test_months` and the last test window ends at the end of `index`. Fewer than 1 window raises `NotEnoughHistory("Walk-forward needs at least 3.5 years before the holdout; AAPL has 2.1.")`.
  - `tune(strategy: Strategy, bars, risk, config, start: date, end: date) -> tuple[Strategy, dict]`: the best concrete strategy by `objective_value` over the range, considering only candidates with at least 5 trades. If none qualifies, it returns the first candidate and sets `"qualified": False` in the info dict.
  - `@dataclass SegmentResult(label: Literal["in_sample","out_of_sample","holdout","backtest"], equity: pd.Series, trades: list[Trade], metrics: dict, start: date, end: date)`.
  - `run_walk_forward(strategy, bars, risk, config, holdout_start: date) -> tuple[SegmentResult, list[dict], Strategy]`. It returns the out-of-sample segment, the per-window info (`window`, `chosen` params, `test` metrics) and the strategy tuned on the final training window. Each test window starts from the previous window's ending equity, so the stitched equity is continuous.
  - `score_holdout(strategy: Strategy, bars, risk, config, holdout_start: date) -> SegmentResult` (`label="holdout"`).
  - `split_dev(bars, holdout_start) -> pd.DataFrame` (rows before `holdout_start`).

- [ ] **Step 1: Write failing tests:**
  - `test_windows_roll_by_six_months`: 10 years of index → the first train window covers years 0 to 3, the first test window covers 3.0 to 3.5, and the windows are contiguous with no overlap between test windows.
  - `test_tuning_never_sees_test_or_holdout`: wrap `backtest` with a spy that records the `end` it was called with during `tune`. Every call's `end` must be ≤ that window's `train_end`, and every date must be < `holdout_start`.
  - `test_tune_requires_min_trades`.
  - `test_not_enough_history_message`: 2 years of bars → `NotEnoughHistory`, and the message contains "3.5 years".
  - `test_oos_equity_is_continuous`.
  - `test_holdout_uses_only_holdout_dates`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat: walk-forward tuning and locked holdout`.

### Task 8: Persistence

**Files:**
- Create: `backend/app/db.py`, `backend/app/repo.py`, `backend/tests/test_repo.py`

**Interfaces:**
- Produces:
  - SQLAlchemy 2 models: `Setting(key PK, value)`, `SavedStrategy(id, name, body JSON, structure_key, created_at, updated_at)`, `BacktestRun(id, symbol, mode, strategy_name, body JSON, created_at)`, `HoldoutScore(structure_key, symbol, count)` with a composite primary key.
  - `get_session()` is a FastAPI dependency; `init_db()` creates the tables.
  - In `repo.py`:
    - `get_holdout_start(s) -> date` initializes to today minus 2 years on first call.
    - `set_holdout_start(s, d: date)`.
    - `save_strategy(s, strategy) -> int`, `list_strategies(s)`, `get_strategy(s, id)`, `delete_strategy(s, id)`.
    - `record_holdout_score(s, key, symbol) -> int` (returns the new count) and `holdout_count(s, key, symbol) -> int`.
    - `save_run(s, body: dict) -> int`, `list_runs(s, limit=20)`, `get_run(s, id)`.

- [ ] **Step 1: Write failing tests** against an in-memory SQLite:
  - `test_holdout_start_initializes_once` (a second call returns the same date).
  - `test_record_holdout_score_increments`.
  - `test_strategy_crud_roundtrip`.
  - `test_runs_listed_newest_first`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat: SQLAlchemy persistence for strategies, runs and holdout`.

### Task 9: API routes

**Files:**
- Create: `backend/app/api/market.py`, `backend/app/api/strategies.py`, `backend/app/api/backtests.py`, `backend/app/api/settings.py`, `backend/app/api/serialize.py`, `backend/tests/api/test_api.py`
- Modify: `backend/app/main.py` (include routers, call `init_db`, and add a `SymbolNotFound` → 404 handler and a `NotEnoughHistory`/`ValueError` → 422 handler)

**Interfaces:**
- Consumes: everything above. Tests override `get_price_provider` with a fixture-backed fake that knows `AAPL` and `SPY`.
- Produces (JSON, dates as ISO strings, NaN/inf → null via `serialize.clean`):
  - `GET /api/universe` → `{symbols, quick_picks}`.
  - `GET /api/bars/{symbol}?start&end` → `{symbol, bars: [{date, open, high, low, close, volume}]}`.
  - `GET /api/indicators` → `[{id, name, category, params, outputs, overlay}]`.
  - `POST /api/indicators/compute {symbol, id, params}` → `{dates: [...], series: {output: [...]}}`.
  - `GET /api/risk-presets` → `RISK_PRESETS`.
  - `POST /api/strategies/describe` (body: Strategy) → `{text}`.
  - `GET/POST /api/strategies`, `GET/DELETE /api/strategies/{id}`.
  - `GET/PUT /api/settings` → `{holdout_start}`.
  - `POST /api/backtests` with body `{symbols: list[str] (1-10), strategy, risk: RiskSettings, config: BacktestConfig, mode: "simple"|"walk_forward", start: date|null, end: date|null, score_holdout: bool = false}` → `{id, summary: {median: metrics, per_symbol: [{symbol, metrics}]}, runs: [Run]}`.
    - Each `Run` is `{symbol, mode, strategy_text, risk, segments: {[label]: {equity: [{date, value}], trades, metrics, start, end}}, baselines: {stock: {equity, metrics}, spy: {equity, metrics}}, markers: [{date, side: "buy"|"sell", price}], windows: [...], holdout_count: int|null}`.
    - In walk-forward mode, `segments` has `out_of_sample` and, if requested, `holdout`. Scoring the holdout increments the count. In simple mode, `segments` has `backtest`.
  - `GET /api/backtests` (recent) and `GET /api/backtests/{id}`.

- [ ] **Step 1: Write failing tests:**
  - `test_unknown_symbol_404`: `GET /api/bars/AAPLL` → 404, detail `"Unknown symbol: AAPLL"`.
  - `test_backtest_simple_has_baselines_and_markers`.
  - `test_backtest_walk_forward_holdout_counts`: two calls with `score_holdout=true` → `holdout_count` is 1, then 2.
  - `test_walk_forward_short_history_422`.
  - `test_never_trading_rule_serializes`: the response parses with the standard `json` module, and `profit_factor` is null.
  - `test_multi_symbol_summary_median`.
  - `test_indicators_list_has_25`.
- [ ] **Step 2:** Run `uv run pytest tests/api -v`. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the full backend suite with `uv run pytest`. Expected: PASS.
- [ ] **Step 5:** Commit `feat: JSON API for market data, strategies and backtests`.

### Task 10: Frontend scaffold, theme and API client

**Files:**
- Create: `frontend/` (Vite React TS), `frontend/src/api/types.ts`, `frontend/src/api/client.ts`, `frontend/src/components/Layout.tsx`, `frontend/src/components/ThemeToggle.tsx`, `frontend/src/lib/format.ts`, `frontend/src/test/format.test.ts`, `frontend/vite.config.ts` (proxy `/api` to `http://localhost:8000`)

**Interfaces:**
- Produces:
  - TypeScript types that mirror the Task 9 JSON (`Bar`, `IndicatorSpec`, `Strategy`, `Group`, `Condition`, `Operand`, `RiskSettings`, `BacktestRequest`, `BacktestResponse`, `Run`, `Segment`, `Metrics`).
  - `api` object with one function per route.
  - `formatPct(x: number | null) -> string` (`0.1234 → "12.3%"`, `null → "—"`) and `formatMoney`.
  - `Layout` with a top bar (logo "Beat The Market", ticker search, theme toggle) and the footer disclaimer.
  - Theme tokens as CSS variables: dark default, with the choice stored in localStorage.
  - Accent is indigo-500; gains are emerald-500 and losses rose-500. Font is Inter.

- [ ] **Step 1: Write failing test** `format.test.ts`: `formatPct(0.1234) === "12.3%"`, `formatPct(null) === "—"`, `formatPct(-0.05) === "-5.0%"`.
- [ ] **Step 2:** Run `cd frontend && npm test -- --run`. Expected: FAIL.
- [ ] **Step 3:** Scaffold with Vite. Add Tailwind, shadcn/ui (button, card, input, select, slider, switch, tabs, table, tooltip, toggle-group), TanStack Query, React Router (routes `/`, `/stock/:symbol`, `/build`, `/results/:id`), and lightweight-charts. Implement the files.
- [ ] **Step 4:** Run `npm test -- --run && npm run build && npx tsc --noEmit`. Expected: PASS.
- [ ] **Step 5:** Commit `feat: frontend scaffold with theme and API client`.

### Task 11: Stock page with chart

**Files:**
- Create: `frontend/src/pages/StockPage.tsx`, `frontend/src/components/PriceChart.tsx`, `frontend/src/components/IndicatorPicker.tsx`, `frontend/src/test/IndicatorPicker.test.tsx`

**Interfaces:**
- Consumes: `api.bars`, `api.indicators`, `api.computeIndicator`.
- Produces:
  - `PriceChart({bars, overlays: {name, dates, values}[], panes: {...}[], markers?: {date, side, price}[]})` draws candles plus volume. Overlay indicators go on the price pane and others in their own pane below. Buy markers are green up arrows and sell markers are red down arrows.
  - `IndicatorPicker({onAdd(id, params)})` is grouped by category and searchable.
  - `StockPage` shows the header (symbol, last close, day change), the chart with chosen indicators, and "Build a strategy" (navigates to `/build?symbol=`).
  - An unknown symbol shows the API's message.

- [ ] **Step 1: Write failing test** `IndicatorPicker.test.tsx` with a mocked list of 3 indicators in 2 categories: typing "rsi" filters to RSI; clicking it calls `onAdd("rsi", {length: 14})`.
- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `npm test -- --run && npm run build`. Expected: PASS.
- [ ] **Step 5:** Commit `feat: stock page with indicator chart`.

### Task 12: Strategy builder

**Files:**
- Create: `frontend/src/pages/BuilderPage.tsx`, `frontend/src/components/RuleBuilder.tsx`, `frontend/src/components/ConditionRow.tsx`, `frontend/src/components/RiskControl.tsx`, `frontend/src/test/RuleBuilder.test.tsx`, `frontend/src/test/RiskControl.test.tsx`

**Interfaces:**
- Consumes: `api.indicators`, `api.riskPresets`, `api.describe`, `api.runBacktest`, `api.saveStrategy`.
- Produces:
  - `RuleBuilder({title: "Buy when" | "Sell when", value: Group, onChange})`: rows of `ConditionRow`, an AND/OR toggle, add and remove buttons, and nested groups one level deep.
  - `ConditionRow` has an operand select (price / indicator with params / number), an op select (`is above`, `is below`, `crosses above`, `crosses below`, `is rising for N bars`, `is falling for N bars`), and a right operand. Any number field has a "range" switch that reveals min, max and step.
  - `RiskControl({value, onChange})`: a three-stop segmented control (Conservative / Balanced / Aggressive) plus "Custom", which reveals the editable fields.
  - `BuilderPage` holds the symbols (multi-select, default from the query or the quick picks), the buy and sell builders, optional stop, target and max hold, the short-on-sell switch (off), risk, slippage bps (0), the date range, a mode tab (Simple / Walk-forward), a "Score holdout" checkbox, a live plain-English summary (debounced `describe`), Run (navigates to `/results/:id`) and Save.
  - The default strategy on load: buy when `RSI(14) crosses above 30`, sell when `RSI(14) crosses below 70`.

- [ ] **Step 1: Write failing tests:**
  - `RuleBuilder`: adding a row then choosing "crosses above" and constant 30 emits a `Group` whose first item is `{kind:"condition", op:"crosses_above", right:{kind:"constant", value:30}}`.
  - Turning on the range switch emits `{min, max, step}`.
  - `RiskControl`: clicking Conservative emits `position_pct 25` and `objective "sortino"`; editing a field switches the preset to `"custom"`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `npm test -- --run && npm run build`. Expected: PASS.
- [ ] **Step 5:** Commit `feat: buy/sell strategy builder with risk control`.

### Task 13: Results page and home dashboard

**Files:**
- Create: `frontend/src/pages/ResultsPage.tsx`, `frontend/src/components/MetricsCards.tsx`, `frontend/src/components/EquityChart.tsx`, `frontend/src/components/TradeTable.tsx`, `frontend/src/pages/HomePage.tsx`, `frontend/src/test/MetricsCards.test.tsx`

**Interfaces:**
- Consumes: `api.getBacktest`, `api.listBacktests`, `api.listStrategies`, `api.universe`, `PriceChart` (Task 11).
- Produces:
  - `MetricsCards({segments, baselines})` shows one card per segment, labeled "Out-of-sample (walk-forward)", "Holdout (never seen)", "Backtest" or "In-sample (diagnostic)" (muted), plus "Buy & hold {symbol}" and "Buy & hold SPY". Each card shows CAGR, total return, max drawdown, Sharpe and the number of trades, with a tooltip explaining each metric. A holdout card shows "Scored N times" when N > 1.
  - `EquityChart` draws the strategy against both baselines, with a drawdown area below.
  - `TradeTable` is sortable, with the exit reason as a badge.
  - `ResultsPage` has tabs per symbol when there are several, a comparison table first, the strategy sentence, the risk preset name, the price chart with markers, and per-window chosen params in walk-forward mode.
  - `HomePage` has quick-pick ticker cards (with last close and change), search, recent backtests and saved strategies.

- [ ] **Step 1: Write failing test** `MetricsCards`: given an `out_of_sample` segment and a `holdout` segment with `holdout_count 3`, both labels render, "Scored 3 times" renders, and a null Sharpe shows "—".
- [ ] **Step 2:** Run it. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `npm test -- --run && npm run build`. Expected: PASS.
- [ ] **Step 5:** Commit `feat: results page and home dashboard`.

### Task 14: One-command startup, CI and README

**Files:**
- Create: `Dockerfile` (multi-stage: node builds the frontend, then a python slim image with uv runs `uvicorn app.main:app --host 0.0.0.0 --port 8000`), `docker-compose.yml` (one service, port 8000, `./data` volume), `dev.sh`, `dev.ps1`, `.github/workflows/ci.yml`, `README.md` (rewritten), `.env.example`
- Modify: none

**Interfaces:**
- `./dev.sh` installs uv deps and npm deps if missing, then runs the backend (`uvicorn --reload` on 8000) and the frontend (`vite` on 5173) together, and stops both on Ctrl+C. `dev.ps1` does the same on Windows.
- CI runs on push and pull_request with two jobs. The backend job runs `uv sync`, `ruff check`, `ruff format --check` and `pytest`. The frontend job runs `npm ci`, `tsc --noEmit`, `npm test -- --run` and `npm run build`.

- [ ] **Step 1:** Write the files.
- [ ] **Step 2:** Run `docker compose build` (if Docker is available), or `cd frontend && npm run build && cd ../backend && uv run uvicorn app.main:app --port 8000` in the background and `curl -s localhost:8000/api/health`. Expected: `{"status":"ok"}`, and `curl -s localhost:8000/` returns the HTML with `<div id="root">`.
- [ ] **Step 3:** Smoke-test against live Yahoo once: run `POST /api/backtests` for AAPL in walk-forward mode with the default strategy and the Balanced preset. Expected: a 200 response with an out-of-sample segment, more than 5 trades and both baselines.
- [ ] **Step 4:** Commit `chore: docker, dev scripts, CI and README`.
