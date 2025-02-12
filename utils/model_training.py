# utils/model_training.py

import os
import pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import concurrent.futures

from utils.data_fetching import get_sp500_tickers, fetch_data, prepare_base_data, indicator_functions

def check_condition(row, col_name, op, threshold):
    try:
        value = float(row[col_name])
    except Exception:
        return False
    try:
        threshold = float(threshold)
    except Exception:
        return False
    if op == ">":
        return value > threshold
    elif op == "<":
        return value < threshold
    elif op == ">=":
        return value >= threshold
    elif op == "<=":
        return value <= threshold
    elif op == "=":
        return value == threshold
    else:
        return False

DEFAULT_PERIODS = {
    "SMA": 20,
    "EMA": 20,
    "RSI": 14,
    "MACD": 12,
    "ATR": 14,
    "CCI": 20,
    "ADX": 14,
    "OBV": 14,
    "STOCH": 14
}

def train_custom_model(buy_conditions, sell_conditions, model_name="Custom", interval="5m"):
    """
    Train a custom model using intraday data (default interval 5m).
    Uses the last 59 days of data.
    Expects buy_conditions and sell_conditions to be lists of tuples:
        (indicator, period, operator, threshold)
    """
    tickers = get_sp500_tickers()
    end = datetime.today()
    start = end - timedelta(days=59)
    required_features = {"Close"}
    for cond in buy_conditions + sell_conditions:
        indicator, period, op, thr = cond
        required_features.add(f"{indicator}_{period}")
    
    def process_ticker(ticker):
        try:
            df = fetch_data(ticker, start, end, interval=interval)
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")
            return None
        try:
            df = prepare_base_data(df)
        except Exception as e:
            print(f"Error preparing data for {ticker}: {e}")
            return None
        
        if df.empty or "Close" not in df.columns:
            print(f"{ticker}: insufficient data. Skipping.")
            return None
        
        for feat in required_features:
            if feat == "Close":
                continue
            if feat not in df.columns:
                try:
                    indicator, period_str = feat.split("_")
                    period = int(period_str)
                except Exception:
                    continue
                if indicator.upper() in indicator_functions:
                    try:
                        df[feat] = indicator_functions[indicator.upper()](df, period)
                    except Exception as e:
                        print(f"Error computing {feat} for {ticker}: {e}")
                        continue
        df = df.dropna(subset=list(required_features))
        if df.empty:
            print(f"{ticker}: All rows dropped. Skipping.")
            return None

        def label_row(row):
            buy_details = []
            for (ind, period, op, thr) in buy_conditions:
                col = f"{ind}_{period}"
                value = row[col] if col in row else None
                cond_met = check_condition(row, col, op, thr)
                buy_details.append((ind, period, value, op, thr, cond_met))
            sell_details = []
            for (ind, period, op, thr) in sell_conditions:
                col = f"{ind}_{period}"
                value = row[col] if col in row else None
                cond_met = check_condition(row, col, op, thr)
                sell_details.append((ind, period, value, op, thr, cond_met))
            
            print(f"Ticker {ticker} | Date: {row['Date']}")
            print(f"  Buy details: {buy_details}")
            print(f"  Sell details: {sell_details}")
            
            buy_flags = [detail[5] for detail in buy_details]
            sell_flags = [detail[5] for detail in sell_details]
            
            if buy_flags and (any(buy_flags) and not all(buy_flags)):
                print(f"  Partial Buy Conditions met: {buy_details}")
            if sell_flags and (any(sell_flags) and not all(sell_flags)):
                print(f"  Partial Sell Conditions met: {sell_details}")
            
            if buy_flags and sell_flags:
                if all(buy_flags) and not any(sell_flags):
                    final_decision = 1
                elif all(sell_flags) and not any(buy_flags):
                    final_decision = 0
                else:
                    final_decision = None
            elif buy_flags:
                final_decision = 1 if all(buy_flags) else None
            elif sell_flags:
                final_decision = 0 if all(sell_flags) else None
            else:
                final_decision = None
            
            print(f"  Final decision: {final_decision}")
            return final_decision

        df["CustomLabel"] = df.apply(label_row, axis=1)
        count_labeled = df["CustomLabel"].notna().sum()
        print(f"{ticker}: Number of labeled rows: {count_labeled}")
        df = df.dropna(subset=["CustomLabel"])
        if df.empty:
            print(f"{ticker}: no rows met conditions. Skipping.")
            return None
        
        df["CustomLabel"] = df["CustomLabel"].astype(int)
        df["Ticker"] = ticker
        print(f"{ticker}: Collected {len(df)} rows after labeling.")
        return df

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_ticker, tickers))
    valid_dfs = [df for df in results if df is not None]
    if not valid_dfs:
        raise ValueError("No valid data collected for custom training.")
    combined_df = pd.concat(valid_dfs, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")
    features_list = list(required_features)
    X = combined_df[features_list].values
    y = combined_df["CustomLabel"].values
    if np.isnan(y).any():
        raise ValueError("Custom target contains NaN values.")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    model_data = {"model": model, "features": features_list}
    if not os.path.exists("models"):
        os.makedirs("models")
    model_filename = f"models/{model_name}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model_data, f)
    print(f"Model trained and saved as '{model_filename}'.")
    return model_data, combined_df

def load_custom_model(model_name="Custom"):
    model_file = f"models/{model_name}.pkl"
    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            model_data = pickle.load(f)
        return model_data
    else:
        raise ValueError("Custom model not found. Please train it first.")

def load_model(model_name):
    model_data = load_custom_model(model_name)
    return model_data["model"], model_data["features"]
