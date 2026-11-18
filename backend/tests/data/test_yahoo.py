import numpy as np
import pandas as pd
import pytest

from app.data import providers
from app.data.providers import SymbolNotFound, YahooProvider
from app.data.universe import load_universe


class FakeTicker:
    def __init__(self, frame):
        self.frame = frame

    def history(self, **kwargs):
        return self.frame


def _patch(monkeypatch, frame):
    monkeypatch.setattr(providers.yf, "Ticker", lambda symbol: FakeTicker(frame))


def test_yahoo_drops_nan_rows_and_normalizes(monkeypatch):
    idx = pd.date_range("2024-01-02", periods=3, freq="B", tz="America/New_York")
    frame = pd.DataFrame(
        {
            "Open": [1.0, np.nan, 3.0],
            "High": [1.5, np.nan, 3.5],
            "Low": [0.5, np.nan, 2.5],
            "Close": [1.2, np.nan, 3.2],
            "Volume": [10, 0, 30],
            "Dividends": [0, 0, 0],
            "Stock Splits": [0, 0, 0],
        },
        index=idx,
    )
    _patch(monkeypatch, frame)
    df = YahooProvider().get_daily_bars("AAPL")
    assert list(df.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert df.index.tz is None
    assert df.index.name == "date"
    assert len(df) == 2
    assert (df["adj_close"] == df["close"]).all()


def test_yahoo_unknown_symbol_raises(monkeypatch):
    _patch(monkeypatch, pd.DataFrame())
    with pytest.raises(SymbolNotFound) as err:
        YahooProvider().get_daily_bars("AAPLL")
    assert err.value.symbol == "AAPLL"


def test_universe_has_quick_picks():
    u = load_universe()
    assert u["quick_picks"] == ["SPY", "AAPL", "MSFT", "NVDA", "AMZN"]
    assert len(u["symbols"]) >= 45
    assert set(u["quick_picks"]) <= set(u["symbols"])


def test_yahoo_network_errors_become_data_source_error(monkeypatch):
    from app.data.providers import DataSourceError

    class Boom:
        def history(self, **kwargs):
            raise RuntimeError("CONNECT tunnel failed")

    monkeypatch.setattr(providers.yf, "Ticker", lambda symbol: Boom())
    with pytest.raises(DataSourceError):
        YahooProvider().get_daily_bars("AAPL")
