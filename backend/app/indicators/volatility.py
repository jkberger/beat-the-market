import numpy as np
import pandas as pd

from app.indicators.registry import Param, indicator, length


def true_range(bars: pd.DataFrame) -> pd.Series:
    prev = bars["close"].shift(1)
    tr = pd.concat(
        [bars["high"] - bars["low"], (bars["high"] - prev).abs(), (bars["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr_series(bars: pd.DataFrame, n: int) -> pd.Series:
    """Wilder ATR: first value is the mean of the first n true ranges, then Wilder smoothing."""
    tr = true_range(bars).to_numpy()
    out = np.full(len(tr), np.nan)
    if len(tr) >= n:
        out[n - 1] = tr[:n].mean()
        for i in range(n, len(tr)):
            out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return pd.Series(out, index=bars.index)


@indicator("atr", "Average True Range", "volatility", ["atr"], False, [length(14, hi=200)])
def atr(bars, length):
    return pd.DataFrame({"atr": atr_series(bars, length)})


@indicator(
    "bbands",
    "Bollinger Bands",
    "volatility",
    ["upper", "middle", "lower"],
    True,
    [length(20), Param("std", "float", 2.0, 0.5, 5.0)],
)
def bbands(bars, length, std):
    c = bars["close"]
    mid = c.rolling(length).mean()
    sd = c.rolling(length).std(ddof=0)
    return pd.DataFrame({"upper": mid + std * sd, "middle": mid, "lower": mid - std * sd})


@indicator(
    "keltner",
    "Keltner Channels",
    "volatility",
    ["upper", "middle", "lower"],
    True,
    [length(20), Param("multiplier", "float", 2.0, 0.5, 5.0), length(10, name="atr_length")],
)
def keltner(bars, length, multiplier, atr_length):
    mid = bars["close"].ewm(span=length, adjust=False, min_periods=length).mean()
    a = atr_series(bars, atr_length)
    return pd.DataFrame(
        {"upper": mid + multiplier * a, "middle": mid, "lower": mid - multiplier * a}
    )


@indicator(
    "donchian", "Donchian Channels", "volatility", ["upper", "middle", "lower"], True, [length(20)]
)
def donchian(bars, length):
    up = bars["high"].rolling(length).max()
    lo = bars["low"].rolling(length).min()
    return pd.DataFrame({"upper": up, "middle": (up + lo) / 2, "lower": lo})
