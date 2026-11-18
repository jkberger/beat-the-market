import time

import numpy as np
import pandas as pd
import pytest

from app.backtest.engine import backtest, run_engine
from app.strategy.models import BacktestConfig, IndicatorOperand, RiskSettings
from tests.strategy.helpers import rsi_strategy

FULL = RiskSettings(
    preset="custom", position_pct=100, stop_atr_mult=None, max_drawdown_pct=None, objective="sharpe"
)
NOCOST = BacktestConfig(initial_equity=1000)


def make_bars(opens, closes=None, highs=None, lows=None):
    n = len(opens)
    idx = pd.bdate_range("2024-01-01", periods=n)
    o = np.asarray(opens, dtype=float)
    c = np.asarray(closes if closes is not None else opens, dtype=float)
    h = np.asarray(highs if highs is not None else np.maximum(o, c), dtype=float)
    lo = np.asarray(lows if lows is not None else np.minimum(o, c), dtype=float)
    return pd.DataFrame(
        {"open": o, "high": h, "low": lo, "close": c, "adj_close": c, "volume": 1.0}, index=idx
    )


def sig(n, *on):
    s = pd.Series(False, index=pd.bdate_range("2024-01-01", periods=n))
    s.iloc[list(on)] = True
    return s


def run(bars, buy, sell, risk=FULL, config=NOCOST, **strategy_extra):
    return run_engine(bars, buy, sell, rsi_strategy(**strategy_extra), risk, config)


def test_round_trips_repeat():
    bars = make_bars(list(range(100, 112)))
    r = run(bars, sig(12, 1, 6), sig(12, 3, 8))
    assert len(r.trades) == 2
    t1, t2 = r.trades
    assert (t1.entry_price, t1.exit_price) == (102, 104)
    assert (t2.entry_price, t2.exit_price) == (107, 109)
    assert t1.exit_reason == t2.exit_reason == "signal"
    assert t1.bars_held == 2
    assert r.position.iloc[2] == 1 and r.position.iloc[5] == 0 and r.position.iloc[7] == 1


def test_fill_is_next_open_not_close():
    bars = make_bars(opens=[10, 20, 30, 40, 50], closes=[11, 21, 31, 41, 51])
    r = run(bars, sig(5, 0), sig(5, 2))
    assert r.trades[0].entry_price == 20
    assert r.trades[0].exit_price == 40


def test_equity_marked_at_close():
    bars = make_bars(opens=[10, 10, 10, 10], closes=[10, 12, 15, 10])
    r = run(bars, sig(4, 0), sig(4))
    # 1000 invested at open 10 on bar 1 -> 100 shares; close 12 -> 1200
    assert r.equity.iloc[1] == pytest.approx(1200)
    assert r.equity.iloc[2] == pytest.approx(1500)


def test_same_bar_buy_and_sell_when_flat_does_nothing():
    bars = make_bars([100] * 6)
    r = run(bars, sig(6, 1), sig(6, 1))
    assert r.trades == []
    assert (r.position == 0).all()


def test_opposite_signal_wins_when_in_position():
    bars = make_bars(list(range(100, 108)))
    r = run(bars, sig(8, 0, 3), sig(8, 3))
    assert r.trades[0].exit_date == bars.index[4].date()


def test_allow_short_flips_on_sell():
    bars = make_bars([100, 100, 110, 100, 90, 95, 95])
    r = run(bars, sig(7, 0, 4), sig(7, 1), allow_short=True)
    long, short = r.trades[0], r.trades[1]
    assert long.direction == "long" and short.direction == "short"
    assert short.entry_price == 110 and short.exit_price == 95
    assert short.return_pct == pytest.approx((110 - 95) / 110)
    assert r.position.iloc[3] == -1


def test_no_short_without_toggle():
    bars = make_bars([100, 100, 110, 100, 90, 95, 95])
    r = run(bars, sig(7), sig(7, 1))
    assert r.trades == []


def test_position_pct_scales_pnl():
    bars = make_bars([100, 100, 120, 120])
    half = RiskSettings(**{**FULL.model_dump(), "position_pct": 50})
    full_r = run(bars, sig(4, 0), sig(4, 2))
    half_r = run(bars, sig(4, 0), sig(4, 2), risk=half)
    assert full_r.equity.iloc[-1] - 1000 == pytest.approx(2 * (half_r.equity.iloc[-1] - 1000))


def test_stop_hit_exits_at_stop_or_gap_open():
    # stop 10% below entry of 100 -> 90. Bar 3 trades down to 85 from open 95 -> fill 90.
    bars = make_bars(
        opens=[100, 100, 98, 95, 95], closes=[100, 99, 96, 88, 95], lows=[100, 98, 95, 85, 94]
    )
    r = run(bars, sig(5, 0), sig(5), stop_loss={"kind": "percent", "value": 10})
    assert r.trades[0].exit_reason == "stop"
    assert r.trades[0].exit_price == pytest.approx(90)
    gap = make_bars(opens=[100, 100, 80, 80], closes=[100, 100, 80, 80])
    r2 = run(gap, sig(4, 0), sig(4), stop_loss={"kind": "percent", "value": 10})
    assert r2.trades[0].exit_price == pytest.approx(80)


def test_risk_atr_stop_used_when_strategy_has_none(bars):
    risk = RiskSettings(
        preset="custom",
        position_pct=100,
        stop_atr_mult=0.5,
        max_drawdown_pct=None,
        objective="sharpe",
    )
    r = backtest(bars, rsi_strategy(), risk, NOCOST)
    assert any(t.exit_reason == "stop" for t in r.trades)


def test_stop_wins_when_both_hit():
    bars = make_bars(
        opens=[100, 100, 100, 100],
        closes=[100, 100, 100, 100],
        highs=[100, 100, 130, 100],
        lows=[100, 100, 70, 100],
    )
    r = run(
        bars,
        sig(4, 0),
        sig(4),
        stop_loss={"kind": "percent", "value": 10},
        take_profit={"kind": "percent", "value": 10},
    )
    assert r.trades[0].exit_reason == "stop"
    assert r.trades[0].exit_price == pytest.approx(90)


def test_take_profit():
    bars = make_bars(
        opens=[100, 100, 105, 100], closes=[100, 104, 108, 100], highs=[100, 104, 115, 100]
    )
    r = run(bars, sig(4, 0), sig(4), take_profit={"kind": "percent", "value": 10})
    assert r.trades[0].exit_reason == "target"
    assert r.trades[0].exit_price == pytest.approx(110)


def test_max_hold():
    bars = make_bars(list(range(100, 110)))
    r = run(bars, sig(10, 0), sig(10), max_hold_bars=3)
    t = r.trades[0]
    assert t.exit_reason == "max_hold"
    assert t.entry_price == 101 and t.exit_price == 104 and t.bars_held == 3


def test_drawdown_stand_aside():
    risk = RiskSettings(
        preset="custom",
        position_pct=100,
        stop_atr_mult=None,
        max_drawdown_pct=10,
        standaside_bars=3,
        objective="sharpe",
    )
    prices = [100, 100, 100, 85, 85, 85, 85, 85, 85, 85, 85]
    bars = make_bars(prices)
    buys = sig(11, 0, 4, 5, 8)
    r = run(bars, buys, sig(11), risk=risk)
    first = r.trades[0]
    assert first.exit_reason == "drawdown"
    assert first.exit_date == bars.index[4].date()
    # Entry signals at bars 4 and 5 fall inside the 3-bar stand-aside; bar 8 re-enters.
    assert r.trades[1].entry_date == bars.index[9].date()


def test_slippage_worsens_fills():
    bars = make_bars([100, 100, 100, 100])
    r = run(bars, sig(4, 0), sig(4, 1), config=BacktestConfig(initial_equity=1000, slippage_bps=10))
    assert r.trades[0].entry_price == pytest.approx(100.1)
    assert r.trades[0].exit_price == pytest.approx(99.9)


def test_end_closes_open_position():
    bars = make_bars(opens=[100, 100, 110], closes=[100, 105, 120])
    r = run(bars, sig(3, 0), sig(3))
    assert r.trades[0].exit_reason == "end"
    assert r.trades[0].exit_price == 120
    assert r.equity.iloc[-1] == pytest.approx(1200)


def test_start_ignores_earlier_signals():
    bars = make_bars(list(range(100, 110)))
    r = run_engine(
        bars, sig(10, 1, 6), sig(10, 3, 8), rsi_strategy(), FULL, NOCOST, start=bars.index[5].date()
    )
    assert len(r.trades) == 1
    assert r.equity.index[0] == bars.index[5]
    assert r.equity.iloc[0] == 1000


def test_lookahead_canary(bars):
    """Scrambling every bar after day k must not change anything the engine did up to day k."""
    k = 1500
    strategy = rsi_strategy(stop_loss={"kind": "atr", "value": 2})
    risk = RiskSettings(
        preset="custom",
        position_pct=100,
        stop_atr_mult=None,
        max_drawdown_pct=15,
        objective="sharpe",
    )
    real = backtest(bars, strategy, risk, NOCOST)
    future = bars.copy()
    rng = np.random.default_rng(99)
    future.iloc[k + 1 :, :4] = future.iloc[k + 1 :, :4].to_numpy() * rng.uniform(
        0.5, 1.5, size=(len(bars) - k - 1, 1)
    )
    scrambled = backtest(future, strategy, risk, NOCOST)
    pd.testing.assert_series_equal(real.equity.iloc[: k + 1], scrambled.equity.iloc[: k + 1])
    done_by_k = [t for t in real.trades if t.exit_date <= bars.index[k].date()]
    assert done_by_k == scrambled.trades[: len(done_by_k)]
    # Rules cannot reference the future: operands have no shift/offset field at all.
    assert not {"shift", "offset", "lag"} & set(IndicatorOperand.model_fields)


def test_20_years_under_one_second():
    n = 5000
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    bars = make_bars(opens=c, closes=c, highs=c * 1.01, lows=c * 0.99)
    bars.index = pd.bdate_range("2000-01-03", periods=n)
    strategy = rsi_strategy()
    t0 = time.perf_counter()
    backtest(bars, strategy, FULL, NOCOST)
    assert time.perf_counter() - t0 < 1.0
