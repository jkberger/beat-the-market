"""Event-style daily backtest engine.

Signals are evaluated at a bar's close and filled at the NEXT bar's open. The strategy
trades in and out repeatedly: flat -> long on buy, long -> flat (or short) on sell.
"""

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from app.indicators.volatility import atr_series
from app.strategy.evaluate import signals
from app.strategy.models import BacktestConfig, RiskSettings, StopSpec, Strategy

ExitReason = Literal["signal", "stop", "target", "max_hold", "drawdown", "end"]


@dataclass
class Trade:
    direction: Literal["long", "short"]
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    return_pct: float  # fraction: 0.05 = +5%
    bars_held: int
    exit_reason: ExitReason


@dataclass
class EngineResult:
    equity: pd.Series
    position: pd.Series
    trades: list[Trade]


def _level(spec: StopSpec | None, entry: float, atr: float, direction: int, sign: int):
    """Price level `sign` (+1 target, -1 stop) away from entry in the trade's favour."""
    if spec is None:
        return None
    value = float(spec.value)
    dist = entry * value / 100 if spec.kind == "percent" else value * atr
    if np.isnan(dist):
        return None
    return entry + sign * direction * dist


def run_engine(
    bars: pd.DataFrame,
    buy: pd.Series,
    sell: pd.Series,
    strategy: Strategy,
    risk: RiskSettings,
    config: BacktestConfig,
    start: date | None = None,
    segments: list[tuple[date, Strategy]] | None = None,
) -> EngineResult:
    """Run one continuous pass. `segments` (walk-forward) lists (from_date, strategy)
    pairs in date order: the strategy whose segment holds the signal bar sets that
    trade's stop, target, max hold and whether shorting is allowed."""
    idx = bars.index
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    lo = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    b = buy.reindex(idx, fill_value=False).to_numpy(bool)
    s = sell.reindex(idx, fill_value=False).to_numpy(bool)
    atr = atr_series(bars, 14).to_numpy(float)
    n = len(idx)
    first = 0 if start is None else int(idx.searchsorted(pd.Timestamp(start)))
    slip = config.slippage_bps / 1e4
    frac = risk.position_pct / 100

    def stop_spec_for(strat: Strategy) -> StopSpec | None:
        if strat.stop_loss is None and risk.stop_atr_mult is not None:
            return StopSpec(kind="atr", value=risk.stop_atr_mult)
        return strat.stop_loss

    if segments:
        seg_starts = idx.searchsorted([pd.Timestamp(d) for d, _ in segments])
        seg_strats = [st for _, st in segments]
    else:
        seg_starts, seg_strats = np.array([0]), [strategy]

    def strategy_at(i: int) -> Strategy:
        k = int(np.searchsorted(seg_starts, i, side="right")) - 1
        return seg_strats[max(k, 0)]

    entry_strategy = strategy_at(first)

    cash = config.initial_equity
    shares = 0.0
    pos = 0
    entry_price = entry_idx = signal_idx = 0
    stop = target = None
    peak = cash
    standaside_until = -1
    pending_exit: ExitReason | None = None
    pending_entry = 0  # +1 long, -1 short
    trades: list[Trade] = []
    equity = np.full(n - first, np.nan)
    position = np.zeros(n - first, dtype=int)

    def close_position(i: int, price: float, reason: ExitReason):
        nonlocal cash, shares, pos, stop, target
        cash += shares * price
        ret = (price / entry_price - 1) if pos == 1 else (entry_price - price) / entry_price
        trades.append(
            Trade(
                "long" if pos == 1 else "short",
                idx[entry_idx].date(),
                entry_price,
                idx[i].date(),
                price,
                ret,
                i - entry_idx,
                reason,
            )
        )
        shares, pos, stop, target = 0.0, 0, None, None

    def open_position(i: int, direction: int):
        nonlocal cash, shares, pos, entry_price, entry_idx, stop, target, entry_strategy
        price = o[i] * (1 + slip * direction)
        invest = cash * frac
        shares = direction * invest / price
        cash -= shares * price
        pos, entry_price, entry_idx = direction, price, i
        a = atr[signal_idx]
        entry_strategy = strategy_at(signal_idx)
        stop = _level(stop_spec_for(entry_strategy), price, a, direction, -1)
        target = _level(entry_strategy.take_profit, price, a, direction, +1)

    for i in range(first, n):
        # 1. Fill orders decided at the previous close, at this bar's open.
        if pending_exit is not None and pos != 0:
            close_position(i, o[i] * (1 - slip * pos), pending_exit)
            if pending_exit == "drawdown":
                peak = cash
        if pending_entry != 0 and pos == 0:
            open_position(i, pending_entry)
        pending_exit, pending_entry = None, 0

        # 2. Intrabar stop / target (stop wins if both are touched).
        if pos != 0:
            hit_stop = stop is not None and (lo[i] <= stop if pos == 1 else h[i] >= stop)
            hit_target = target is not None and (h[i] >= target if pos == 1 else lo[i] <= target)
            if hit_stop:
                fill = min(o[i], stop) if pos == 1 else max(o[i], stop)
                close_position(i, fill * (1 - slip * pos), "stop")
            elif hit_target:
                fill = max(o[i], target) if pos == 1 else min(o[i], target)
                close_position(i, fill * (1 - slip * pos), "target")

        # 3. Mark to market at the close.
        eq = cash + shares * c[i]
        equity[i - first] = eq
        position[i - first] = pos
        peak = max(peak, eq)
        if i == n - 1:
            break

        # 4. Decide orders for the next open.
        current = strategy_at(i)
        max_hold = entry_strategy.max_hold_bars
        if pos != 0 and max_hold and i - entry_idx + 1 >= max_hold:
            pending_exit = "max_hold"
        if (
            pos != 0
            and risk.max_drawdown_pct is not None
            and eq <= peak * (1 - risk.max_drawdown_pct / 100)
        ):
            pending_exit = "drawdown"
            standaside_until = i + 1 + risk.standaside_bars
        if pending_exit is not None:
            continue
        can_enter = i >= standaside_until
        signal_idx = i
        if pos == 0:
            if b[i] and not s[i] and can_enter:
                pending_entry = 1
            elif s[i] and not b[i] and current.allow_short and can_enter:
                pending_entry = -1
        elif pos == 1 and s[i]:
            pending_exit = "signal"
            if current.allow_short and not b[i] and can_enter:
                pending_entry = -1
        elif pos == -1 and b[i]:
            pending_exit = "signal"
            if not s[i] and can_enter:
                pending_entry = 1

    if pos != 0:
        close_position(n - 1, c[n - 1], "end")
        equity[-1] = cash

    out_idx = idx[first:]
    return EngineResult(
        pd.Series(equity, index=out_idx), pd.Series(position, index=out_idx), trades
    )


def backtest(
    bars: pd.DataFrame,
    strategy: Strategy,
    risk: RiskSettings,
    config: BacktestConfig,
    start: date | None = None,
    end: date | None = None,
) -> EngineResult:
    if end is not None:
        bars = bars[bars.index <= pd.Timestamp(end)]
    buy, sell = signals(strategy, bars)
    return run_engine(bars, buy, sell, strategy, risk, config, start)
