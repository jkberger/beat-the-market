import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import Trade
from app.backtest.metrics import KEYS, buy_and_hold, compute_metrics, objective_value
from app.strategy.models import BacktestConfig


def _eq(values):
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)), dtype=float)


def _trade(ret):
    d = date(2020, 1, 1)
    return Trade("long", d, 100, d, 100 * (1 + ret), ret, 1, "signal")


def test_keys_are_exact():
    m = compute_metrics(_eq([100, 101, 102]), [], _eq([0, 0, 0]))
    assert list(m) == KEYS


def test_total_return_and_cagr():
    eq = _eq(np.geomspace(100, 200, 505))
    m = compute_metrics(eq, [], _eq([1] * 505))
    assert m["total_return"] == pytest.approx(1.0)
    assert m["cagr"] == pytest.approx(math.sqrt(2) - 1, rel=1e-6)
    assert m["time_in_market"] == pytest.approx(1.0)


def test_max_drawdown():
    m = compute_metrics(_eq([100, 120, 90, 130]), [], _eq([1, 1, 1, 1]))
    assert m["max_drawdown"] == pytest.approx(-0.25)


def test_no_trades_gives_none_ratios():
    m = compute_metrics(_eq([100] * 10), [], _eq([0] * 10))
    assert m["num_trades"] == 0
    assert m["win_rate"] is None
    assert m["profit_factor"] is None
    assert m["sharpe"] is None
    assert m["total_return"] == 0


def test_no_losses_profit_factor_none():
    m = compute_metrics(_eq([100, 110, 121]), [_trade(0.1), _trade(0.1)], _eq([1, 1, 1]))
    assert m["profit_factor"] is None
    assert m["win_rate"] == 1.0
    assert m["avg_loss"] is None


def test_win_rate_and_profit_factor():
    trades = [_trade(0.10), _trade(-0.05), _trade(0.05), _trade(-0.05)]
    m = compute_metrics(_eq([100, 101, 102]), trades, _eq([0, 1, 0]))
    assert m["win_rate"] == 0.5
    assert m["profit_factor"] == pytest.approx(1.5)
    assert m["avg_win"] == pytest.approx(0.075)
    assert m["avg_loss"] == pytest.approx(-0.05)


def test_no_nan_or_inf_anywhere(bars):
    m = compute_metrics(_eq([100, 100, 100]), [], _eq([0, 0, 0]))
    for v in m.values():
        assert v is None or math.isfinite(v)


def test_buy_and_hold_matches_price_return(bars):
    r = buy_and_hold(bars, BacktestConfig(initial_equity=1000))
    expected = bars["close"].iloc[-1] / bars["open"].iloc[0]
    assert r.equity.iloc[-1] == pytest.approx(1000 * expected)
    assert len(r.trades) == 1
    assert r.equity.index[0] == bars.index[0]


def test_buy_and_hold_respects_range(bars):
    r = buy_and_hold(bars, BacktestConfig(), start=date(2016, 1, 1), end=date(2017, 1, 1))
    assert r.equity.index[0] >= pd.Timestamp("2016-01-01")
    assert r.equity.index[-1] <= pd.Timestamp("2017-01-01")


def test_objective_value_per_preset():
    m = {"sortino": 1.5, "max_drawdown": -0.2, "sharpe": 1.1, "cagr": 0.12}
    assert objective_value(m, "sortino") == pytest.approx(1.3)
    assert objective_value(m, "sharpe") == 1.1
    assert objective_value(m, "cagr") == 0.12
    assert objective_value({**m, "sharpe": None}, "sharpe") == float("-inf")
