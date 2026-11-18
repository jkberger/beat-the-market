import numpy as np
import pandas as pd
import ta

from app.indicators.registry import Param, indicator, length
from app.indicators.volatility import atr_series


def _wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


@indicator("sma", "Simple Moving Average", "trend", ["sma"], True, [length(20)])
def sma(bars, length):
    return pd.DataFrame({"sma": bars["close"].rolling(length).mean()})


@indicator("ema", "Exponential Moving Average", "trend", ["ema"], True, [length(20)])
def ema(bars, length):
    ema = bars["close"].ewm(span=length, adjust=False, min_periods=length).mean()
    return pd.DataFrame({"ema": ema})


@indicator("wma", "Weighted Moving Average", "trend", ["wma"], True, [length(20)])
def wma(bars, length):
    return pd.DataFrame({"wma": _wma(bars["close"], length)})


@indicator("hma", "Hull Moving Average", "trend", ["hma"], True, [length(20, lo=4)])
def hma(bars, length):
    c = bars["close"]
    raw = 2 * _wma(c, length // 2) - _wma(c, length)
    return pd.DataFrame({"hma": _wma(raw, int(np.sqrt(length)))})


@indicator("vwma", "Volume-Weighted Moving Average", "trend", ["vwma"], True, [length(20)])
def vwma(bars, length):
    pv = (bars["close"] * bars["volume"]).rolling(length).sum()
    return pd.DataFrame({"vwma": pv / bars["volume"].rolling(length).sum()})


@indicator(
    "macd",
    "MACD",
    "trend",
    ["macd", "signal", "hist"],
    False,
    [length(12, name="fast"), length(26, name="slow"), length(9, name="signal")],
)
def macd(bars, fast, slow, signal):
    m = ta.trend.MACD(bars["close"], slow, fast, signal)
    return pd.DataFrame({"macd": m.macd(), "signal": m.macd_signal(), "hist": m.macd_diff()})


@indicator("adx", "ADX / DMI", "trend", ["adx", "plus_di", "minus_di"], False, [length(14, hi=100)])
def adx(bars, length):
    h, lo = bars["high"], bars["low"]
    up, down = h.diff(), -lo.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=h.index)
    atr = atr_series(bars, length)
    alpha = 1 / length
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=length).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=length).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    return pd.DataFrame({"adx": adx, "plus_di": plus_di, "minus_di": minus_di})


@indicator("aroon", "Aroon", "trend", ["up", "down"], False, [length(25, hi=200)])
def aroon(bars, length):
    w = length + 1
    up = bars["high"].rolling(w).apply(lambda x: float(np.argmax(x)), raw=True) / length * 100
    down = bars["low"].rolling(w).apply(lambda x: float(np.argmin(x)), raw=True) / length * 100
    return pd.DataFrame({"up": up, "down": down})


@indicator(
    "psar",
    "Parabolic SAR",
    "trend",
    ["psar"],
    True,
    [Param("step", "float", 0.02, 0.001, 0.2), Param("max_step", "float", 0.2, 0.01, 1.0)],
)
def psar(bars, step, max_step):
    h, lo = bars["high"].to_numpy(), bars["low"].to_numpy()
    n = len(h)
    out = np.full(n, np.nan)
    if n < 2:
        return pd.DataFrame({"psar": out})
    up = True
    af = step
    ep = h[0]
    sar = lo[0]
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, lo[i - 1], lo[i - 2] if i > 1 else lo[i - 1])
            if lo[i] < sar:
                up, sar, ep, af = False, ep, lo[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_step)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > sar:
                up, sar, ep, af = True, ep, h[i], step
            elif lo[i] < ep:
                ep, af = lo[i], min(af + step, max_step)
        out[i] = sar
    return pd.DataFrame({"psar": out})


@indicator(
    "ichimoku",
    "Ichimoku Cloud (unshifted)",
    "trend",
    ["conversion", "base", "span_a", "span_b"],
    True,
    [length(9, name="conversion"), length(26, name="base"), length(52, name="span_b")],
)
def ichimoku(bars, conversion, base, span_b):
    h, lo = bars["high"], bars["low"]

    def mid(n):
        return (h.rolling(n).max() + lo.rolling(n).min()) / 2

    conv, base_line = mid(conversion), mid(base)
    # Spans are NOT shifted forward: shifting would leak future-plotted values into bar t.
    return pd.DataFrame(
        {
            "conversion": conv,
            "base": base_line,
            "span_a": (conv + base_line) / 2,
            "span_b": mid(span_b),
        }
    )


@indicator(
    "supertrend",
    "Supertrend",
    "trend",
    ["supertrend", "direction"],
    True,
    [length(10, hi=100), Param("multiplier", "float", 3.0, 0.5, 10.0)],
)
def supertrend(bars, length, multiplier):
    h, lo, c = bars["high"].to_numpy(), bars["low"].to_numpy(), bars["close"].to_numpy()
    atr = atr_series(bars, length).to_numpy()
    hl2 = (h + lo) / 2
    upper_basic, lower_basic = hl2 + multiplier * atr, hl2 - multiplier * atr
    n = len(c)
    st = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    upper = lower = np.nan
    trend = 1
    for i in range(n):
        if np.isnan(atr[i]):
            continue
        if np.isnan(upper):
            upper, lower = upper_basic[i], lower_basic[i]
        else:
            upper = upper_basic[i] if upper_basic[i] < upper or c[i - 1] > upper else upper
            lower = lower_basic[i] if lower_basic[i] > lower or c[i - 1] < lower else lower
            if trend == 1 and c[i] < lower:
                trend = -1
            elif trend == -1 and c[i] > upper:
                trend = 1
        st[i] = lower if trend == 1 else upper
        direction[i] = trend
    return pd.DataFrame({"supertrend": st, "direction": direction})
