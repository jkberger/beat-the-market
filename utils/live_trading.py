# utils/live_trading.py
import time
import threading
from datetime import datetime, timedelta
import numpy as np
from utils.data_fetching import fetch_data, prepare_base_data, indicator_functions
from utils.model_training import load_model
from utils.alpaca_bot import AlpacaBot

# Shared state variables for live trading
live_trading_thread = None
trading_stop_event = threading.Event()
trade_history = []
account_history = []
initial_equity = None

def live_trading_loop(symbol, sleep_interval=60):
    global initial_equity, account_history, trade_history
    print(f"Starting live trading loop for {symbol}")
    bot = AlpacaBot()
    model, feature_cols = load_model("Custom")
    account = bot.get_account()
    initial_equity = float(account.equity)
    account_history.clear()
    trade_history.clear()
    while not trading_stop_event.is_set():
        try:
            if not bot.is_market_open():
                print("Market closed. Waiting...")
                time.sleep(sleep_interval)
                continue
            print(f"Live trading iteration for {symbol} at {datetime.now()}")
            end_date = datetime.today().date()
            start_date = end_date - timedelta(days=10)
            df = fetch_data(symbol, start_date, end_date)
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
            if df.empty:
                print("No data fetched. Skipping iteration.")
                time.sleep(sleep_interval)
                continue
            last_row = df.iloc[-1]
            X = np.array([[last_row[col] for col in feature_cols]])
            prediction = int(model.predict(X)[0])
            print(f"Prediction for {symbol} is {prediction}")
            current_position = bot.get_position(symbol)
            price = float(last_row["Close"])
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if prediction == 1:
                if current_position == 0:
                    print("Bullish signal. Buying one share.")
                    bot.place_order(symbol, qty=1, side="buy")
                    trade_history.append({"timestamp": timestamp, "action": "BUY", "qty": 1, "price": price, "symbol": symbol})
                else:
                    print("Bullish signal but already in position.")
            else:
                if current_position > 0:
                    print("Bearish signal. Selling all shares.")
                    bot.place_order(symbol, qty=current_position, side="sell")
                    trade_history.append({"timestamp": timestamp, "action": "SELL", "qty": current_position, "price": price, "symbol": symbol})
                else:
                    print("Bearish signal but no position.")
            account = bot.get_account()
            account_history.append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "equity": float(account.equity),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value)
            })
        except Exception as e:
            print("Error in live trading loop:", e)
        time.sleep(sleep_interval)
    print("Live trading loop stopped.")
