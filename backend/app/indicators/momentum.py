import pandas as pd
import ta

from app.indicators.registry import indicator, length


def _rsi(close: pd.Series, n: int) -> pd.Series:
    return ta.momentum.RSIIndicator(close, n).rsi()


@indicator("rsi", "Relative Strength Index", "momentum", ["rsi"], False, [length(14, hi=200)])
def rsi(bars, length):
    return pd.DataFrame({"rsi": _rsi(bars["close"], length)})


@indicator(
    "stoch",
    "Stochastic Oscillator",
    "momentum",
    ["k", "d"],
    False,
    [length(14, hi=200), length(3, lo=1, hi=50, name="smooth")],
)
def stoch(bars, length, smooth):
    lo = bars["low"].rolling(length).min()
    hi = bars["high"].rolling(length).max()
    k = 100 * (bars["close"] - lo) / (hi - lo)
    return pd.DataFrame({"k": k, "d": k.rolling(smooth).mean()})


@indicator(
    "stochrsi",
    "Stochastic RSI",
    "momentum",
    ["k", "d"],
    False,
    [
        length(14, hi=200),
        length(3, lo=1, hi=50, name="smooth_k"),
        length(3, lo=1, hi=50, name="smooth_d"),
    ],
)
def stochrsi(bars, length, smooth_k, smooth_d):
    r = _rsi(bars["close"], length)
    lo, hi = r.rolling(length).min(), r.rolling(length).max()
    raw = 100 * (r - lo) / (hi - lo)
    k = raw.rolling(smooth_k).mean()
    return pd.DataFrame({"k": k, "d": k.rolling(smooth_d).mean()})


@indicator("willr", "Williams %R", "momentum", ["willr"], False, [length(14, hi=200)])
def willr(bars, length):
    hi = bars["high"].rolling(length).max()
    lo = bars["low"].rolling(length).min()
    return pd.DataFrame({"willr": -100 * (hi - bars["close"]) / (hi - lo)})


@indicator("cci", "Commodity Channel Index", "momentum", ["cci"], False, [length(20, hi=200)])
def cci(bars, length):
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    sma = tp.rolling(length).mean()
    mad = tp.rolling(length).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
    return pd.DataFrame({"cci": (tp - sma) / (0.015 * mad)})


@indicator("roc", "Rate of Change", "momentum", ["roc"], False, [length(12, lo=1)])
def roc(bars, length):
    return pd.DataFrame({"roc": bars["close"].pct_change(length) * 100})


@indicator("mom", "Momentum", "momentum", ["mom"], False, [length(10, lo=1)])
def mom(bars, length):
    return pd.DataFrame({"mom": bars["close"].diff(length)})
