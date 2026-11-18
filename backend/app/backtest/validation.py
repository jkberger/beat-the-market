"""Walk-forward tuning and the locked final holdout.

Nothing in here may let tuning see data from a test window or from the holdout.
"""

from dataclasses import asdict, dataclass
from datetime import date
from typing import Literal

import pandas as pd

from app.backtest.engine import Trade, backtest, run_engine
from app.backtest.metrics import compute_metrics, objective_value
from app.strategy.evaluate import signals
from app.strategy.grid import expand_grid
from app.strategy.models import BacktestConfig, RiskSettings, Strategy

MIN_TRADES = 5


class NotEnoughHistory(ValueError):
    pass


@dataclass
class Window:
    train_start: date
    train_end: date
    test_start: date
    test_end: date


@dataclass
class SegmentResult:
    label: Literal["in_sample", "out_of_sample", "holdout", "backtest"]
    equity: pd.Series
    trades: list[Trade]
    metrics: dict
    start: date
    end: date
    position: pd.Series | None = None


def split_dev(bars: pd.DataFrame, holdout_start: date) -> pd.DataFrame:
    return bars[bars.index < pd.Timestamp(holdout_start)]


def walk_forward_windows(
    index: pd.DatetimeIndex, train_years: int = 3, test_months: int = 6
) -> list[Window]:
    need = train_years + test_months / 12
    if len(index) == 0:
        raise NotEnoughHistory(
            f"Walk-forward needs at least {need:g} years before the holdout, and the chosen "
            "start date is on or after the holdout start."
        )
    first, last = index[0], index[-1]
    have = (last - first).days / 365.25
    if have < need:
        raise NotEnoughHistory(
            f"Walk-forward needs at least {need:g} years before the holdout; "
            f"this stock has {have:.1f}."
        )
    windows = []
    anchor = first
    while True:
        test_from = anchor + pd.DateOffset(years=train_years)
        test_to = test_from + pd.DateOffset(months=test_months)
        train = index[(index >= anchor) & (index < test_from)]
        test = index[(index >= test_from) & (index < test_to)]
        if len(test) == 0 or len(train) == 0:
            break
        windows.append(Window(train[0].date(), train[-1].date(), test[0].date(), test[-1].date()))
        if test[-1] >= last:
            break
        anchor = anchor + pd.DateOffset(months=test_months)
    return windows


def _segment(label, result, start, end) -> SegmentResult:
    return SegmentResult(
        label,
        result.equity,
        result.trades,
        compute_metrics(result.equity, result.trades, result.position),
        start,
        end,
        result.position,
    )


def tune(
    strategy: Strategy,
    bars: pd.DataFrame,
    risk: RiskSettings,
    config: BacktestConfig,
    start: date,
    end: date,
) -> tuple[Strategy, dict]:
    candidates = expand_grid(strategy)
    best, best_score, best_metrics = candidates[0], float("-inf"), None
    qualified = False
    for cand in candidates:
        r = backtest(bars, cand, risk, config, start=start, end=end)
        m = compute_metrics(r.equity, r.trades, r.position)
        if m["num_trades"] < MIN_TRADES:
            continue
        score = objective_value(m, risk.objective)
        if not qualified or score > best_score:
            best, best_score, best_metrics, qualified = cand, score, m, True
    return best, {
        "qualified": qualified,
        "candidates": len(candidates),
        "objective": risk.objective,
        "score": best_score if qualified else None,
        "train_metrics": best_metrics,
    }


def _chosen_values(strategy: Strategy, tuned: Strategy) -> dict:
    """The tuned values, keyed by a readable path, for display."""
    from app.strategy.grid import _find_ranges, _get

    raw, out = strategy.model_dump(), {}
    for path in _find_ranges(raw):
        out[".".join(str(p) for p in path)] = _get(tuned.model_dump(), path)
    return out


def run_walk_forward(
    strategy: Strategy,
    bars: pd.DataFrame,
    risk: RiskSettings,
    config: BacktestConfig,
    holdout_start: date,
    train_years: int = 3,
    test_months: int = 6,
) -> tuple[SegmentResult, list[dict], Strategy]:
    dev = split_dev(bars, holdout_start)
    windows = walk_forward_windows(dev.index, train_years, test_months)
    # Tune each window on its own training span, then trade the whole out-of-sample
    # span in ONE continuous run: a position opened in one window carries into the next
    # instead of being closed every six months. Each window's signals come from the
    # settings tuned for it and are computed only from bars up to that window's end.
    buy_parts, sell_parts, segments, chosen_list = [], [], [], []
    for w in windows:
        chosen, tinfo = tune(strategy, dev, risk, config, w.train_start, w.train_end)
        upto = dev[dev.index <= pd.Timestamp(w.test_end)]
        b, s = signals(chosen, upto)
        in_test = b.index >= pd.Timestamp(w.test_start)
        buy_parts.append(b[in_test])
        sell_parts.append(s[in_test])
        segments.append((w.test_start, chosen))
        chosen_list.append((chosen, tinfo))
    oos_bars = dev[dev.index <= pd.Timestamp(windows[-1].test_end)]
    r = run_engine(
        oos_bars,
        pd.concat(buy_parts),
        pd.concat(sell_parts),
        strategy,
        risk,
        config,
        start=windows[0].test_start,
        segments=segments,
    )
    equity, position, trades = r.equity, r.position, r.trades
    info = []
    for w, (chosen, tinfo) in zip(windows, chosen_list, strict=True):
        lo, hi = pd.Timestamp(w.test_start), pd.Timestamp(w.test_end)
        mask = (equity.index >= lo) & (equity.index <= hi)
        w_trades = [t for t in trades if w.test_start <= t.exit_date <= w.test_end]
        info.append(
            {
                "window": asdict(w),
                "chosen": _chosen_values(strategy, chosen),
                "qualified": tinfo["qualified"],
                "test": compute_metrics(equity[mask], w_trades, position[mask]),
            }
        )
    seg = SegmentResult(
        "out_of_sample",
        equity,
        trades,
        compute_metrics(equity, trades, position),
        windows[0].test_start,
        windows[-1].test_end,
        position,
    )
    # Final settings for the holdout: tuned on the most recent training span before it.
    last_from = (dev.index[-1] - pd.DateOffset(years=train_years)).date()
    final, _ = tune(strategy, dev, risk, config, last_from, dev.index[-1].date())
    return seg, info, final


def score_holdout(
    strategy: Strategy,
    bars: pd.DataFrame,
    risk: RiskSettings,
    config: BacktestConfig,
    holdout_start: date,
) -> SegmentResult:
    if bars.index[-1] < pd.Timestamp(holdout_start):
        raise NotEnoughHistory("There is no data on or after the holdout start date yet.")
    r = backtest(bars, strategy, risk, config, start=holdout_start)
    return _segment("holdout", r, holdout_start, bars.index[-1].date())


def simple_segment(
    strategy: Strategy,
    bars: pd.DataFrame,
    risk: RiskSettings,
    config: BacktestConfig,
    start: date | None,
    end: date | None,
) -> SegmentResult:
    r = backtest(bars, strategy, risk, config, start=start, end=end)
    if len(r.equity) < 2:
        raise NotEnoughHistory(
            f"No price data between {start or 'the first day'} and {end or 'today'}."
        )
    return _segment("backtest", r, r.equity.index[0].date(), r.equity.index[-1].date())
