import os
import pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import concurrent.futures

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

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
    Original custom training using fixed hyperparameters.
    Expects buy_conditions and sell_conditions as lists of tuples:
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
    # Train using RandomForest with fixed hyperparameters
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    model_data = {"model": model, "features": features_list, "model_type": "RandomForest"}
    if not os.path.exists("models"):
        os.makedirs("models")
    model_filename = f"models/{model_name}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(model_data, f)
    print(f"Model trained and saved as '{model_filename}'.")
    return model_data, combined_df

def train_model_with_tuning(buy_conditions, sell_conditions, model_name="TunedModel", model_type="RandomForest", interval="5m"):
    """
    Train a custom model using intraday data (default interval 5m) with hyperparameter tuning.
    Uses the last 59 days of data and automatically selects optimal hyperparameters for the chosen model type.
    Evaluates the model on a held-out test set and reports performance metrics.
    
    model_type can be one of: "RandomForest", "GradientBoosting", "LogisticRegression"
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
        raise ValueError("No valid data collected for training.")
    combined_df = pd.concat(valid_dfs, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")
    features_list = list(required_features)
    X = combined_df[features_list].values
    y = combined_df["CustomLabel"].values
    if np.isnan(y).any():
        raise ValueError("Custom target contains NaN values.")
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define estimator and parameter grid based on model_type
    if model_type == "RandomForest":
        estimator = RandomForestClassifier(random_state=42)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
        }
    elif model_type == "GradientBoosting":
        estimator = GradientBoostingClassifier(random_state=42)
        param_grid = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        }
    elif model_type == "LogisticRegression":
        estimator = LogisticRegression(max_iter=1000, random_state=42)
        param_grid = {
            "C": [0.1, 1, 10],
        }
    else:
        raise ValueError("Unsupported model type")
    
    grid_search = GridSearchCV(estimator, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_score = grid_search.best_score_
    
    # Evaluate on the test set
    y_pred = best_estimator.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Best Parameters:", best_params)
    print("Cross-Validation Score:", cv_score)
    print("Test Accuracy:", test_accuracy)
    print("Classification Report:\n", report)
    
    # Train best estimator on full dataset
    best_estimator.fit(X, y)
    
    model_data = {
        "model": best_estimator,
        "features": features_list,
        "model_type": model_type,
        "best_params": best_params,
        "cv_score": cv_score,
        "test_accuracy": test_accuracy,
        "classification_report": report,
    }
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

def parse_indicator_json(indicator_json):
    """
    Parse the JSON object for indicators.
    Expects a dictionary with keys "buy" and "sell", where each maps to a dict of indicator strings.
    Returns:
        buy_conditions: list of tuples (indicator, period, operator, threshold)
        sell_conditions: list of tuples (indicator, period, operator, threshold)
    Uses default operators and thresholds:
        For BUY: operator "<", threshold "30"
        For SELL: operator ">", threshold "70"
    """
    buy_conditions = []
    sell_conditions = []
    default_buy_operator = "<"
    default_buy_threshold = "30"
    default_sell_operator = ">"
    default_sell_threshold = "70"
    if "buy" in indicator_json:
        for key, value in indicator_json["buy"].items():
            if value:
                try:
                    indicator, period_str = key.split("_")
                    period = int(period_str)
                    buy_conditions.append((indicator, period, default_buy_operator, default_buy_threshold))
                except Exception as e:
                    print(f"Error parsing buy indicator {key}: {e}")
    if "sell" in indicator_json:
        for key, value in indicator_json["sell"].items():
            if value:
                try:
                    indicator, period_str = key.split("_")
                    period = int(period_str)
                    sell_conditions.append((indicator, period, default_sell_operator, default_sell_threshold))
                except Exception as e:
                    print(f"Error parsing sell indicator {key}: {e}")
    return buy_conditions, sell_conditions
