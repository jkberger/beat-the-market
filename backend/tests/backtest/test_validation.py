from datetime import date

import pandas as pd
import pytest

from app.backtest import validation
from app.backtest.validation import (
    NotEnoughHistory,
    run_walk_forward,
    score_holdout,
    split_dev,
    tune,
    walk_forward_windows,
)
from app.strategy.models import BacktestConfig, RiskSettings
from tests.strategy.helpers import rsi_strategy

RISK = RiskSettings(
    preset="custom", position_pct=100, stop_atr_mult=None, max_drawdown_pct=None, objective="sharpe"
)
CFG = BacktestConfig(initial_equity=1000)
HOLDOUT = date(2021, 9, 1)


def test_windows_roll_by_six_months(bars):
    windows = walk_forward_windows(bars.index)
    first = windows[0]
    assert first.train_start == bars.index[0].date()
    assert date(2016, 12, 25) <= first.train_end < date(2017, 1, 3)
    assert date(2017, 1, 1) <= first.test_start <= date(2017, 1, 5)
    assert date(2017, 6, 25) <= first.test_end < date(2017, 7, 3)
    for a, b in zip(windows, windows[1:], strict=False):
        assert a.test_end < b.test_start
        assert (pd.Timestamp(b.test_start) - pd.Timestamp(a.test_end)).days <= 4
        assert b.train_start > a.train_start
    assert windows[-1].test_end == bars.index[-1].date()


def test_not_enough_history_message(bars):
    with pytest.raises(NotEnoughHistory, match="3.5 years"):
        walk_forward_windows(bars.index[:504])


def test_tune_requires_min_trades(bars):
    # Levels that almost never trigger (RSI crossing above 1) vs normal levels.
    s = rsi_strategy(buy_level={"min": 1, "max": 31, "step": 30})
    best, info = tune(s, bars, RISK, CFG, date(2014, 1, 1), date(2017, 1, 1))
    assert best.buy.items[0].right.value == 31
    assert info["qualified"] is True
    assert info["candidates"] == 2


def test_tune_reports_unqualified_when_nothing_trades(bars):
    s = rsi_strategy(buy_level={"min": 0.5, "max": 1, "step": 0.5})
    _, info = tune(s, bars, RISK, CFG, date(2014, 1, 1), date(2014, 6, 1))
    assert info["qualified"] is False


def test_tuning_never_sees_test_or_holdout(bars, monkeypatch):
    calls = []
    real = validation.backtest

    def spy(b, strategy, risk, config, start=None, end=None):
        calls.append((start, end, b.index[-1].date()))
        return real(b, strategy, risk, config, start=start, end=end)

    monkeypatch.setattr(validation, "backtest", spy)
    s = rsi_strategy(buy_level={"min": 25, "max": 35, "step": 5})
    _, windows, _ = run_walk_forward(s, bars, RISK, CFG, HOLDOUT)
    assert all(e < HOLDOUT and last < HOLDOUT for _, e, last in calls)
    tuned = [w for w in windows]
    for w in tuned:
        assert w["window"]["train_end"] < w["window"]["test_start"]


def test_tune_calls_stay_inside_train_window(bars, monkeypatch):
    seen = []
    real = validation.backtest

    def spy(b, strategy, risk, config, start=None, end=None):
        seen.append(end)
        return real(b, strategy, risk, config, start=start, end=end)

    monkeypatch.setattr(validation, "backtest", spy)
    s = rsi_strategy(buy_level={"min": 25, "max": 35, "step": 5})
    tune(s, bars, RISK, CFG, date(2015, 1, 1), date(2017, 12, 31))
    assert seen and all(e <= date(2017, 12, 31) for e in seen)


def test_oos_equity_is_continuous(bars):
    seg, windows, final = run_walk_forward(rsi_strategy(), bars, RISK, CFG, HOLDOUT)
    assert seg.label == "out_of_sample"
    assert seg.equity.index.is_monotonic_increasing
    assert not seg.equity.index.duplicated().any()
    jumps = seg.equity.pct_change().abs().max()
    assert jumps < 0.25
    assert seg.equity.index[-1] < pd.Timestamp(HOLDOUT)
    assert len(windows) >= 2
    assert seg.metrics["num_trades"] == len(seg.trades) > 5


def test_final_strategy_is_concrete(bars):
    s = rsi_strategy(buy_level={"min": 25, "max": 35, "step": 5})
    _, windows, final = run_walk_forward(s, bars, RISK, CFG, HOLDOUT)
    assert final.buy.items[0].right.value in (25, 30, 35)
    assert "chosen" in windows[0] and "test" in windows[0]


def test_holdout_uses_only_holdout_dates(bars):
    seg = score_holdout(rsi_strategy(), bars, RISK, CFG, HOLDOUT)
    assert seg.label == "holdout"
    assert seg.equity.index[0] >= pd.Timestamp(HOLDOUT)
    assert all(t.entry_date >= HOLDOUT for t in seg.trades)


def test_split_dev(bars):
    dev = split_dev(bars, HOLDOUT)
    assert dev.index[-1] < pd.Timestamp(HOLDOUT)


def test_anchored_windows_train_on_all_prior_history(bars):
    windows = walk_forward_windows(bars.index, anchored=True)
    assert all(w.train_start == bars.index[0].date() for w in windows)
    assert windows[0].test_start == walk_forward_windows(bars.index)[0].test_start
    assert windows[-1].train_end < windows[-1].test_start


def test_anchored_final_strategy_tuned_on_all_dev(bars, monkeypatch):
    starts = []
    real = validation.backtest

    def spy(b, strategy, risk, config, start=None, end=None):
        starts.append(start)
        return real(b, strategy, risk, config, start=start, end=end)

    monkeypatch.setattr(validation, "backtest", spy)
    s = rsi_strategy(buy_level={"min": 25, "max": 35, "step": 5})
    run_walk_forward(s, bars, RISK, CFG, HOLDOUT, anchored=True)
    tune_starts = {st for st in starts if st == bars.index[0].date()}
    assert tune_starts, "anchored tuning should start at the first bar"
