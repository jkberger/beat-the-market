import numpy as np
import pandas as pd
import ta

from app.indicators import compute


def _tiny():
    idx = pd.bdate_range("2024-01-01", periods=5)
    c = pd.Series([1.0, 2, 3, 4, 5], index=idx)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "adj_close": c, "volume": 1.0})


def test_sma_reference():
    out = compute("sma", _tiny(), {"length": 3})["sma"].tolist()
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2:] == [2.0, 3.0, 4.0]


def _close_to(a: pd.Series, b: pd.Series):
    mask = a.notna() & b.notna()
    assert mask.sum() > 2000
    np.testing.assert_allclose(a[mask], b[mask], rtol=1e-8, atol=1e-8)


def test_rsi_matches_ta(bars):
    _close_to(compute("rsi", bars)["rsi"], ta.momentum.RSIIndicator(bars["close"], 14).rsi())


def test_macd_matches_ta(bars):
    ref = ta.trend.MACD(bars["close"], 26, 12, 9)
    out = compute("macd", bars)
    _close_to(out["macd"], ref.macd())
    _close_to(out["signal"], ref.macd_signal())


def test_atr_matches_ta_after_warmup(bars):
    ref = ta.volatility.AverageTrueRange(bars["high"], bars["low"], bars["close"], 14)
    out = compute("atr", bars)["atr"]
    assert out.iloc[:13].isna().all()
    _close_to(out.iloc[14:], ref.average_true_range().iloc[14:])


def test_bbands_matches_ta(bars):
    ref = ta.volatility.BollingerBands(bars["close"], 20, 2)
    out = compute("bbands", bars)
    _close_to(out["upper"], ref.bollinger_hband())
    _close_to(out["lower"], ref.bollinger_lband())
