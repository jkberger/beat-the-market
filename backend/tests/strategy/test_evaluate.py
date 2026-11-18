import pandas as pd
import pytest

from app.strategy.evaluate import evaluate_group, signals
from app.strategy.models import Group
from tests.strategy.helpers import rsi_strategy


def _bars(values):
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    c = pd.Series(values, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "adj_close": c, "volume": 1.0})


def _group(join, *conds):
    return Group.model_validate({"kind": "group", "join": join, "items": list(conds)})


def _cond(op, right=None, bars=1):
    c = {"kind": "condition", "left": {"kind": "price", "field": "close"}, "op": op, "bars": bars}
    if right is not None:
        c["right"] = {"kind": "constant", "value": right}
    return c


def test_crosses_above():
    out = evaluate_group(_group("and", _cond("crosses_above", 2.5)), _bars([1, 2, 3]))
    assert out.tolist() == [False, False, True]


def test_crosses_below():
    out = evaluate_group(_group("and", _cond("crosses_below", 2.5)), _bars([3, 2, 1]))
    assert out.tolist() == [False, True, False]


def test_greater_and_less():
    b = _bars([1, 3, 2])
    assert evaluate_group(_group("and", _cond(">", 1.5)), b).tolist() == [False, True, True]
    assert evaluate_group(_group("and", _cond("<", 2.5)), b).tolist() == [True, False, True]


def test_rising_for_n_bars():
    out = evaluate_group(_group("and", _cond("rising", bars=2)), _bars([1, 2, 3, 4]))
    assert out.tolist() == [False, False, True, True]


def test_falling_for_n_bars():
    out = evaluate_group(_group("and", _cond("falling", bars=1)), _bars([3, 2, 2]))
    assert out.tolist() == [False, True, False]


def test_and_or_groups():
    b = _bars([1, 3, 5])
    assert evaluate_group(_group("and", _cond(">", 2), _cond("<", 4)), b).tolist() == [
        False,
        True,
        False,
    ]
    assert evaluate_group(_group("or", _cond("<", 2), _cond(">", 4)), b).tolist() == [
        True,
        False,
        True,
    ]


def test_nested_group():
    inner = {"kind": "group", "join": "or", "items": [_cond("<", 2), _cond(">", 4)]}
    out = evaluate_group(_group("and", inner, _cond(">", 0)), _bars([1, 3, 5]))
    assert out.tolist() == [True, False, True]


def test_nan_compares_false(bars):
    cond = {
        "kind": "condition",
        "left": {"kind": "price", "field": "close"},
        "op": ">",
        "right": {"kind": "indicator", "id": "sma", "params": {"length": 200}, "output": "sma"},
    }
    out = evaluate_group(_group("and", cond), bars)
    assert not out.iloc[:199].any()
    assert out.dtype == bool


def test_signals_returns_buy_and_sell(bars):
    buy, sell = signals(rsi_strategy(), bars)
    assert buy.any() and sell.any()
    assert buy.index.equals(bars.index)


def test_range_in_signals_raises(bars):
    s = rsi_strategy(buy_level={"min": 20, "max": 30, "step": 5})
    with pytest.raises(ValueError):
        signals(s, bars)


def test_condition_needs_right_operand():
    with pytest.raises(ValueError):
        _group("and", _cond(">"))
