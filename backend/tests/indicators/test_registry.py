import pandas as pd
import pytest

from app.indicators import REGISTRY, compute

EXPECTED = {
    "sma": ["sma"],
    "ema": ["ema"],
    "wma": ["wma"],
    "hma": ["hma"],
    "vwma": ["vwma"],
    "macd": ["macd", "signal", "hist"],
    "rsi": ["rsi"],
    "stoch": ["k", "d"],
    "stochrsi": ["k", "d"],
    "willr": ["willr"],
    "cci": ["cci"],
    "roc": ["roc"],
    "mom": ["mom"],
    "adx": ["adx", "plus_di", "minus_di"],
    "aroon": ["up", "down"],
    "psar": ["psar"],
    "ichimoku": ["conversion", "base", "span_a", "span_b"],
    "supertrend": ["supertrend", "direction"],
    "bbands": ["upper", "middle", "lower"],
    "keltner": ["upper", "middle", "lower"],
    "donchian": ["upper", "middle", "lower"],
    "atr": ["atr"],
    "obv": ["obv"],
    "cmf": ["cmf"],
    "mfi": ["mfi"],
}


def test_registry_has_all_25_ids():
    assert set(REGISTRY) == set(EXPECTED)
    assert len(REGISTRY) == 25


@pytest.mark.parametrize("indicator_id", sorted(EXPECTED))
def test_outputs_match_spec(indicator_id, bars):
    spec = REGISTRY[indicator_id]
    assert spec.outputs == EXPECTED[indicator_id]
    out = compute(indicator_id, bars)
    assert list(out.columns) == EXPECTED[indicator_id]
    assert out.index.equals(bars.index)
    assert out.iloc[-100:].notna().all().all()


@pytest.mark.parametrize("indicator_id", sorted(EXPECTED))
def test_every_indicator_has_no_lookahead(indicator_id, bars):
    partial = compute(indicator_id, bars.iloc[:1500])
    full = compute(indicator_id, bars).iloc[:1500]
    pd.testing.assert_frame_equal(partial, full, check_exact=False, rtol=1e-9, atol=1e-9)


def test_out_of_range_param_raises(bars):
    with pytest.raises(ValueError):
        compute("rsi", bars, {"length": 0})


def test_unknown_indicator_raises(bars):
    with pytest.raises(ValueError):
        compute("nope", bars)


def test_unknown_param_raises(bars):
    with pytest.raises(ValueError):
        compute("rsi", bars, {"lenght": 14})
