"""Generate a deterministic synthetic daily-bar fixture for offline tests.

Usage: uv run python scripts/make_synthetic_fixture.py tests/fixtures/synthetic_daily.csv
"""

import sys

import numpy as np
import pandas as pd


def make_bars(start: str = "2014-01-02", periods: int = 2520, seed: int = 7, s0: float = 100.0):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=periods, name="date")
    # Regime-switching drift so trends exist for indicators to find.
    drift = np.repeat(rng.choice([-0.0008, 0.0002, 0.0012], size=periods // 60 + 1), 60)[:periods]
    rets = drift + rng.normal(0, 0.015, periods)
    close = s0 * np.exp(np.cumsum(rets))
    prev = np.concatenate([[s0], close[:-1]])
    open_ = prev * np.exp(rng.normal(0, 0.004, periods))
    spread = np.abs(rng.normal(0, 0.008, periods)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(1_000_000, 5_000_000, periods)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    ).round(6)


if __name__ == "__main__":
    make_bars().to_csv(sys.argv[1])
