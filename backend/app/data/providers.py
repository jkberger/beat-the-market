from datetime import date
from typing import Protocol

import pandas as pd
import yfinance as yf

COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class SymbolNotFound(Exception):
    def __init__(self, symbol: str):
        super().__init__(f"Unknown symbol: {symbol}")
        self.symbol = symbol


class DataSourceError(Exception):
    """The price source could not be reached or answered with an error."""


class PriceProvider(Protocol):
    def get_daily_bars(
        self, symbol: str, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame: ...


def slice_bars(df: pd.DataFrame, start: date | None, end: date | None) -> pd.DataFrame:
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


class YahooProvider:
    """Daily bars from Yahoo Finance. Unofficial; personal use only."""

    def get_daily_bars(
        self, symbol: str, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        try:
            raw = yf.Ticker(symbol).history(period="max", auto_adjust=True)
        except Exception as e:  # yfinance raises many transport-specific types
            raise DataSourceError(f"Couldn't reach Yahoo Finance for {symbol}: {e}") from e
        if raw is None or raw.empty:
            raise SymbolNotFound(symbol)
        df = raw.rename(columns=str.lower)
        index = pd.DatetimeIndex(df.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        df.index = index.normalize()
        df.index.name = "date"
        df = df.dropna(subset=["open", "high", "low", "close"])
        if df.empty:
            raise SymbolNotFound(symbol)
        df["adj_close"] = df["close"]
        df = df[COLUMNS].sort_index()
        return slice_bars(df, start, end)
