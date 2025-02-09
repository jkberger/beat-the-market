# utils/backtesting.py
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from utils.model_training import load_model
from utils.data_fetching import fetch_data, prepare_base_data, indicator_functions

def backtest_model(symbol, model_name):
    """
    Backtest the selected custom model on a given stock.
    """
    model, feature_cols = load_model(model_name)
    end = datetime.today().date()
    start = end - timedelta(days=365)
    df = fetch_data(symbol, start, end)
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
                "Date": row["Date"].strftime("%Y-%m-%d"),
                "Action": "BUY",
                "Close": price,
                "Indicators": {col: row[col] for col in feature_cols if col != "Close"}
            })
        elif pred == 0 and position > 0:
            cash += position * price
            actions.append({
                "Date": row["Date"].strftime("%Y-%m-%d"),
                "Action": "SELL",
                "Close": price,
                "Indicators": {col: row[col] for col in feature_cols if col != "Close"}
            })
            position = 0
        portfolio_values.append(cash + position * price)
    df["Portfolio Value"] = portfolio_values
    return df, actions
