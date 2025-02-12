# utils/backtesting.py
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from utils.model_training import load_model
from utils.data_fetching import fetch_data, prepare_base_data, indicator_functions

def backtest_model(symbol, model_name, interval="5m"):
    """
    Backtest the custom model using intraday data (default 5m) over the last 59 days.
    Also computes benchmarks for buy-and-hold stock and buy-and-hold SPY.
    """
    model, feature_cols = load_model(model_name)
    end = datetime.today()
    start = end - timedelta(days=59)
    df = fetch_data(symbol, start, end, interval=interval)
    df = prepare_base_data(df)
    for feat in feature_cols:
        if feat == "Close":
            continue
        if feat not in df.columns:
            try:
                indicator, period_str = feat.split("_")
                period = int(period_str)
            except Exception:
                continue
            if indicator.upper() in indicator_functions:
                df[feat] = indicator_functions[indicator.upper()](df, period)
    df = df.dropna(subset=feature_cols)
    X = df[feature_cols].values
    df["Prediction"] = model.predict(X)
    initial_capital = 10000.0
    position = 0
    cash = initial_capital
    portfolio_values = []
    actions = []
    for idx, row in df.iterrows():
        price = float(row["Close"])
        pred = int(row["Prediction"])
        if pred == 1 and cash >= price:
            position += 1
            cash -= price
            actions.append({
                "Date": row["Date"].strftime("%Y-%m-%d %H:%M"),
                "Action": "BUY",
                "Close": price,
                "Indicators": {col: row[col] for col in feature_cols if col != "Close"}
            })
        elif pred == 0 and position > 0:
            cash += position * price
            actions.append({
                "Date": row["Date"].strftime("%Y-%m-%d %H:%M"),
                "Action": "SELL",
                "Close": price,
                "Indicators": {col: row[col] for col in feature_cols if col != "Close"}
            })
            position = 0
        portfolio_values.append(cash + position * price)
    df["Portfolio Value"] = portfolio_values

    # Benchmark 1: Buy and hold the chosen stock.
    starting_stock_price = df["Close"].iloc[0]
    buy_hold_stock = [initial_capital * (price / starting_stock_price) for price in df["Close"]]

    # Benchmark 2: Buy and hold SPY.
    spy_df = fetch_data("SPY", start, end, interval=interval)
    spy_df = prepare_base_data(spy_df)
    # Reindex SPY data to match df dates (forward-fill missing values)
    spy_df = spy_df.set_index("Date").reindex(df["Date"], method='ffill').reset_index()
    starting_spy_price = spy_df["Close"].iloc[0]
    buy_hold_spy = [initial_capital * (price / starting_spy_price) for price in spy_df["Close"]]

    return df, actions, buy_hold_stock, buy_hold_spy
