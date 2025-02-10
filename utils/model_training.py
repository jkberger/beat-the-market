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
    """Return True if the value in row[col_name] meets the condition op threshold; otherwise, False."""
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

def train_custom_model(buy_conditions, sell_conditions, model_name="Custom"):
    """
    Train a custom model using user-defined conditions.
    Processes each ticker in parallel to improve performance.
    """
    tickers = get_sp500_tickers()
    end = datetime.today().date()
    start = end - timedelta(days=365 * 2)
    required_features = {"Close"}
    for cond in buy_conditions + sell_conditions:
        indicator, period, op, thr = cond
        required_features.add(f"{indicator}_{period}")
    
    def process_ticker(ticker):
        try:
            df = fetch_data(ticker, start, end)
            df = prepare_base_data(df)
            if df.empty:
                print(f"{ticker}: No data after preparing base data. Skipping.")
                return None
            if "Close" not in df.columns:
                print(f"{ticker}: 'Close' not found. Skipping.")
                return None
            # Compute required indicator features
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
                        df[feat] = indicator_functions[indicator.upper()](df, period)
                    else:
                        print(f"{ticker}: Indicator {indicator} not supported. Skipping condition.")
                        continue
            df = df.dropna(subset=list(required_features))
            if df.empty:
                print(f"{ticker}: All rows dropped after computing required features. Skipping.")
                return None
            
            # Label rows based on buy and sell conditions
            def label_row(row):
                buy_met = all(check_condition(row, f"{ind}_{period}", op, thr)
                              for (ind, period, op, thr) in buy_conditions)
                sell_met = all(check_condition(row, f"{ind}_{period}", op, thr)
                               for (ind, period, op, thr) in sell_conditions)
                if buy_met and not sell_met:
                    return 1
                elif sell_met and not buy_met:
                    return 0
                else:
                    return None
            df["CustomLabel"] = df.apply(label_row, axis=1)
            df = df.dropna(subset=["CustomLabel"])
            if df.empty:
                print(f"{ticker}: No rows met custom conditions. Skipping.")
                return None
            df["CustomLabel"] = df["CustomLabel"].astype(int)
            df["Ticker"] = ticker
            print(f"{ticker}: Collected {len(df)} custom-labeled rows.")
            return df
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            return None

    # Use a ThreadPoolExecutor to process tickers in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(process_ticker, tickers))
    
    valid_dfs = [df for df in results if df is not None]
    
    if not valid_dfs:
        raise ValueError("No valid data collected for custom training.")
    
    combined_df = pd.concat(valid_dfs, ignore_index=True)
    print(f"Combined custom training data shape: {combined_df.shape}")
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
    print(f"Custom model trained and saved as '{model_filename}'.")
    return model_data, combined_df

def load_custom_model(model_name="Custom"):
    model_file = f"models/{model_name}.pkl"
    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            model_data = pickle.load(f)
        return model_data  # dict with keys "model" and "features"
    else:
        raise ValueError("Custom model not found. Please train the custom model first.")

def load_model(model_name):
    model_data = load_custom_model(model_name)
    return model_data["model"], model_data["features"]
