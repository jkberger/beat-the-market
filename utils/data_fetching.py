# utils/data_fetching.py
import os
import pandas as pd
import yfinance as yf
from datetime import datetime
import ta  # Technical analysis library
from cachetools import TTLCache, cached

# In-memory cache: up to 100 items for 1 hour
data_cache = TTLCache(maxsize=100, ttl=3600)

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df["Symbol"].tolist()
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        return tickers
    except Exception as e:
        print("Error fetching S&P 500 list:", e)
        return []

# Note: We include 'interval' as part of the cache key.
@cached(cache=data_cache, key=lambda symbol, start, end, interval="5m": f"{symbol}_{start}_{end}_{interval}")
def fetch_data(symbol, start, end, interval="5m"):
    """
    Download historical data for a given ticker using yfinance.
    This version checks for alternative date column names and renames them to 'Date'.
    """
    df = yf.download(symbol, start=start, end=end, progress=False, interval=interval)
    df.reset_index(inplace=True)
    
    # Check and rename alternative date columns if necessary
    if "Datetime" in df.columns and "Date" not in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)
    if "date" in df.columns and "Date" not in df.columns:
        df.rename(columns={"date": "Date"}, inplace=True)
    
    # If columns are a MultiIndex, flatten them; otherwise, clean up column names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(map(str, col)).strip() for col in df.columns.values]
    else:
        df.columns = [str(col).strip() for col in df.columns]
    
    # Simplify column names: if a column equals (ignoring case) 'date', rename it to 'Date'
    cleaned_columns = []
    for col in df.columns:
        if col.lower() == "date":
            cleaned_columns.append("Date")
        else:
            cleaned_columns.append(col.split()[0])
    df.columns = cleaned_columns
    
    # Remove any duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Ensure the 'Date' column exists
    if "Date" not in df.columns:
        raise ValueError(f"Ticker {symbol}: 'Date' column not found.")
    
    # Convert 'Date' column to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def prepare_base_data(df):
    """
    Prepare raw data for model training:
      - Clean column names
      - Ensure essential columns (Date, Close, High, Low, Volume) exist
      - Flatten any multi-dimensional columns.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(map(str, col)).strip() for col in df.columns.values]
    else:
        df.columns = [str(col).strip() for col in df.columns]
    
    df.columns = [col.split()[0] for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Flatten columns if necessary
    cols_to_check = ["Date", "Close", "Adj", "High", "Low", "Volume"]
    for col in cols_to_check:
        if col in df.columns and isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]
    
    if "Date" in df.columns:
        df.loc[:, "Date"] = pd.to_datetime(df.loc[:, "Date"])
    
    if "Close" not in df.columns:
        if "Adj" in df.columns:
            df["Close"] = df["Adj"]
        else:
            raise ValueError("Neither 'Close' nor 'Adj' found in data.")
    
    df = df.sort_values("Date")
    return df

# Helper for computing STOCH so that we always return a 1D Series.
def compute_stoch(df, period):
    """
    Compute the Stochastic Oscillator (%K) as a 1D Series.
    """
    result = ta.momentum.stoch(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=period,
        smooth_window=3
    )
    if isinstance(result, pd.DataFrame):
        return result.iloc[:, 0]
    return pd.Series(result).squeeze()

# Technical Indicator Functions with default intraday-friendly values
indicator_functions = {
    "SMA": lambda df, period: ta.trend.sma_indicator(df["Close"], window=period),
    "EMA": lambda df, period: ta.trend.ema_indicator(df["Close"], window=period),
    "RSI": lambda df, period: ta.momentum.rsi(df["Close"], window=period),
    "MACD": lambda df, period: ta.trend.macd(df["Close"], window_fast=period, window_slow=period*2, window_sign=max(1, period//2)),
    "ATR": lambda df, period: ta.volatility.average_true_range(high=df["High"], low=df["Low"], close=df["Close"], window=period),
    "CCI": lambda df, period: ta.trend.cci(high=df["High"], low=df["Low"], close=df["Close"], window=period, constant=0.015),
    "ADX": lambda df, period: ta.trend.adx(high=df["High"], low=df["Low"], close=df["Close"], window=period),
    "OBV": lambda df, period: ta.volume.on_balance_volume(df["Close"], df["Volume"]),
    "STOCH": lambda df, period: compute_stoch(df, period)
}
