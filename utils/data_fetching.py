import os
import pandas as pd
import yfinance as yf
from datetime import datetime
import ta  # Technical analysis library
from cachetools import TTLCache, cached

# Create an in-memory cache that holds up to 100 items for 1 hour (3600 seconds)
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

@cached(cache=data_cache, key=lambda symbol, start, end: f"{symbol}_{start}_{end}")
def fetch_data(symbol, start, end):
    """
    Download daily historical data for a given ticker using yfinance.
    Reset the index and clean column names.
    """
    df = yf.download(symbol, start=start, end=end, progress=False, interval="1d")
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(map(str, col)).strip() for col in df.columns.values]
    else:
        df.columns = [str(col).strip() for col in df.columns]
    # Clean column names: if a column contains whitespace, take the first token.
    cleaned_columns = []
    for col in df.columns:
        if col.lower() == "date":
            cleaned_columns.append("Date")
        else:
            cleaned_columns.append(col.split()[0])
    df.columns = cleaned_columns
    print(f"Ticker {symbol} cleaned columns: {df.columns.tolist()}")
    if "Date" not in df.columns:
        raise ValueError(f"Ticker {symbol}: 'Date' column not found.")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_base_data(df):
    """
    Prepare raw data for custom model training.
    - Ensure essential columns ('Date', 'Close', etc.) are present.
    - Flatten any multi-dimensional columns.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(map(str, col)).strip() for col in df.columns.values]
    else:
        df.columns = [str(col).strip() for col in df.columns]
    
    df.columns = [col.split()[0] for col in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    
    # List of columns we care about
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

# Helper for computing STOCH so that we always get a 1D Series.
def compute_stoch(df, period):
    """
    Computes the Stochastic Oscillator (%K) for the given DataFrame and period.
    If the ta function returns a DataFrame, selects the first column.
    """
    result = ta.momentum.stoch(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=period,
        smooth_window=3
    )
    # If result is a DataFrame (or has more than one column), select the first column
    if isinstance(result, pd.DataFrame):
        return result.iloc[:, 0]
    # Also, if result is a numpy array with more than one dimension, squeeze it.
    return pd.Series(result).squeeze()

# Technical Indicator Functions using ta
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
