import numpy as np
import pandas as pd

from app.indicators.registry import indicator, length


@indicator("obv", "On-Balance Volume", "volume", ["obv"], False)
def obv(bars):
    direction = np.sign(bars["close"].diff()).fillna(0)
    return pd.DataFrame({"obv": (direction * bars["volume"]).cumsum()})


@indicator("cmf", "Chaikin Money Flow", "volume", ["cmf"], False, [length(20)])
def cmf(bars, length):
    h, lo, c, v = bars["high"], bars["low"], bars["close"], bars["volume"]
    rng = (h - lo).replace(0, np.nan)
    mfv = ((c - lo) - (h - c)) / rng * v
    return pd.DataFrame({"cmf": mfv.fillna(0).rolling(length).sum() / v.rolling(length).sum()})


@indicator("mfi", "Money Flow Index", "volume", ["mfi"], False, [length(14, hi=200)])
def mfi(bars, length):
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    flow = tp * bars["volume"]
    delta = tp.diff()
    pos = flow.where(delta > 0, 0.0).rolling(length).sum()
    neg = flow.where(delta < 0, 0.0).rolling(length).sum()
    mfi = 100 - 100 / (1 + pos / neg)
    mfi.iloc[:length] = np.nan
    return pd.DataFrame({"mfi": mfi})
