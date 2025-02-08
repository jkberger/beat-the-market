#!/usr/bin/env python3
"""
trading_app.py

An extended trading system demo that:
- Dynamically fetches a list of S&P 500 stocks.
- Trains a universal model using historical data from all S&P 500 stocks.
- Backtests the universal model on a selected stock.
- Connects to Alpaca for paper trading.
- Provides a dashboard (the home page) with options to:
    • Train the universal model,
    • View a backtest chart (with a dropdown to select a stock),
    • and access live trading (with a form to enter a stock symbol).
- The live trading dashboard shows account info, an equity-over-time chart,
  profit/loss percentage, and a trade history.
"""

import os
import pickle
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for
from sklearn.ensemble import RandomForestClassifier
import alpaca_trade_api as tradeapi

# ----------------------------
# Utility: Get S&P 500 tickers from Wikipedia
# ----------------------------
def get_sp500_tickers():
    """
    Fetch a list of S&P 500 ticker symbols from Wikipedia.
    Returns a list of tickers.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df["Symbol"].tolist()
        # Replace any '.' with '-' (e.g., BRK.B becomes BRK-B)
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        return tickers
    except Exception as e:
        print("Error fetching S&P 500 list:", e)
        return []

# ----------------------------
# Data Fetching and Column Cleaning
# ----------------------------
def fetch_data(symbol, start, end):
    """
    Download historical data for a given ticker using yfinance.
    We explicitly use a daily interval. After downloading, we flatten
    any MultiIndex columns (if present) and then "clean" the column names.
    
    In many cases, yfinance returns columns like "Close MMM" or "Open MMM".
    This function renames those columns to standard names: "Date", "Open",
    "High", "Low", "Close", "Volume", etc.
    """
    # Request daily data explicitly.
    df = yf.download(symbol, start=start, end=end, progress=False, interval="1d")
    df.reset_index(inplace=True)
    
    # Flatten MultiIndex columns if present.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(map(str, col)).strip() for col in df.columns.values]
    else:
        df.columns = [str(col).strip() for col in df.columns]
    
    # Clean column names: if a column name contains a space, take the first word.
    # This should convert "Close MMM" to "Close", "Open MMM" to "Open", etc.
    cleaned_columns = []
    for col in df.columns:
        # Special case: if the column is already "Date", leave it.
        if col.lower() == "date":
            cleaned_columns.append("Date")
        else:
            # Split on whitespace and take the first token.
            token = col.split()[0]
            cleaned_columns.append(token)
    df.columns = cleaned_columns

    # Print the cleaned columns for debugging.
    print(f"Ticker {symbol} cleaned columns: {df.columns.tolist()}")
    
    if "Date" not in df.columns:
        raise ValueError(f"Ticker {symbol}: 'Date' column not found after cleaning.")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# ----------------------------
# Feature Engineering
# ----------------------------
def add_features(df):
    """
    Compute the 20-day and 50-day simple moving averages and create a binary signal.
    Use the "Close" column (or "Adj Close" if "Close" is missing).
    Signal = 1 if SMA_20 > SMA_50, else 0.
    """
    # Use "Close" if available; if not, try "Adj" (for Adj Close)
    if "Close" not in df.columns:
        if "Adj" in df.columns:
            df["Close"] = df["Adj"]
        else:
            raise ValueError("Neither 'Close' nor 'Adj' found in data.")
    if isinstance(df["Close"], pd.DataFrame):
        df["Close"] = df["Close"].iloc[:, 0]
    df = df.sort_values("Date")
    df["SMA_20"] = df["Close"].rolling(window=20, min_periods=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50, min_periods=50).mean()
    df = df.dropna(subset=["SMA_20", "SMA_50"])
    df["Signal"] = (df["SMA_20"] > df["SMA_50"]).astype(int)
    return df

# ----------------------------
# Universal Model Training and Loading
# ----------------------------
def train_universal_model():
    """
    Train a universal model using historical data from all S&P 500 stocks.
    For each ticker:
      - Fetch data,
      - Compute features,
      - Select only the desired columns,
      - Drop rows with NaN in 'Signal'.
    Then, concatenate all valid DataFrames and train a RandomForest model on
    three features: "Close", "SMA_20", and "SMA_50".
    """
    tickers = get_sp500_tickers()
    valid_dfs = []
    end = datetime.today().date()
    start = end - timedelta(days=365 * 2)
    
    expected_columns = ["Date", "Close", "SMA_20", "SMA_50", "Signal"]
    valid_count = 0
    
    for ticker in tickers:
        try:
            df = fetch_data(ticker, start, end)
            if df.empty:
                print(f"{ticker}: No data returned. Skipping.")
                continue
            df = add_features(df)
            if df.empty:
                print(f"{ticker}: No rows after computing features. Skipping.")
                continue
            # Check if all expected columns exist.
            if not set(expected_columns).issubset(set(df.columns)):
                missing = set(expected_columns) - set(df.columns)
                print(f"{ticker}: Missing columns {missing}. Skipping.")
                continue
            # Select only the expected columns.
            df = df[expected_columns].copy()
            # Drop any rows with NaN in "Signal".
            df = df.dropna(subset=["Signal"])
            if df.empty:
                print(f"{ticker}: All rows dropped after dropna. Skipping.")
                continue
            df["Ticker"] = ticker
            valid_dfs.append(df)
            valid_count += 1
            print(f"{ticker}: Collected {len(df)} rows.")
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    if not valid_dfs:
        raise ValueError(f"No valid data collected for universal training from {valid_count} tickers.")
    
    combined_df = pd.concat(valid_dfs, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")
    
    X = combined_df[["Close", "SMA_20", "SMA_50"]].values
    y = combined_df["Signal"].values
    
    if np.isnan(y).any():
        raise ValueError("Target y contains NaN values.")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    if not os.path.exists("models"):
        os.makedirs("models")
    with open("models/Universal.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Universal model trained and saved as 'models/Universal.pkl'.")
    return model, combined_df

def load_universal_model():
    """
    Load the universal model from disk.
    Raise an error if the model's expected number of features is not 3.
    """
    model_file = "models/Universal.pkl"
    expected_features = 3
    if os.path.exists(model_file):
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        if getattr(model, "n_features_in_", None) != expected_features:
            raise ValueError("Universal model does not have the expected number of features (3). Please retrain the model.")
    else:
        raise ValueError("Universal model not found. Please train the universal model first.")
    return model

# ----------------------------
# Backtesting with Universal Model
# ----------------------------
def backtest_universal_model(symbol):
    """
    Backtest the universal model on a selected stock.
    """
    universal_model = load_universal_model()
    end = datetime.today().date()
    start = end - timedelta(days=365)
    df = fetch_data(symbol, start, end)
    df = add_features(df)
    
    X = df[["Close", "SMA_20", "SMA_50"]].values
    df["Prediction"] = universal_model.predict(X)
    
    initial_capital = 10000.0
    position = 0
    cash = initial_capital
    portfolio_values = []
    
    for i, row in df.iterrows():
        price = float(row["Close"])
        pred = int(row["Prediction"])
        if pred == 1 and cash >= price:
            position += 1
            cash -= price
        elif pred == 0 and position > 0:
            cash += position * price
            position = 0
        portfolio_values.append(cash + position * price)
    df["Portfolio Value"] = portfolio_values
    return df

# ----------------------------
# Alpaca Trading Bot
# ----------------------------
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

class AlpacaBot:
    def __init__(self):
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise ValueError("Alpaca API credentials not set.")
        self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2")
    
    def get_account(self):
        return self.api.get_account()
    
    def get_position(self, symbol):
        try:
            position = self.api.get_position(symbol)
            return float(position.qty)
        except Exception:
            return 0.0
    
    def is_market_open(self):
        clock = self.api.get_clock()
        return clock.is_open
    
    def place_order(self, symbol, qty, side, order_type="market", time_in_force="gtc"):
        order = self.api.submit_order(symbol=symbol, qty=qty, side=side,
                                      type=order_type, time_in_force=time_in_force)
        return order

# ----------------------------
# Live Trading Loop with Universal Model
# ----------------------------
live_trading_thread = None
trading_stop_event = threading.Event()
trade_history = []    # List to store trade logs
account_history = []  # List to store account equity logs
initial_equity = None

def live_trading_loop(symbol, sleep_interval=60):
    """
    Continuously trade live on a selected stock using the universal model.
    Records account equity and trade history.
    """
    global initial_equity, account_history, trade_history
    print(f"Starting live trading loop for {symbol}")
    bot = AlpacaBot()
    universal_model = load_universal_model()
    
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
            df = add_features(df)
            if df.empty:
                print("No data fetched. Skipping iteration.")
                time.sleep(sleep_interval)
                continue
            
            last_row = df.iloc[-1]
            X = np.array([[last_row["Close"], last_row["SMA_20"], last_row["SMA_50"]]])
            prediction = int(universal_model.predict(X)[0])
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
            account_history.append({"timestamp": datetime.now().strftime("%H:%M:%S"),
                                    "equity": float(account.equity),
                                    "cash": float(account.cash),
                                    "portfolio_value": float(account.portfolio_value)})
            
        except Exception as e:
            print("Error in live trading loop:", e)
        
        time.sleep(sleep_interval)
    
    print("Live trading loop stopped.")

# ----------------------------
# Flask Application and Routes
# ----------------------------
load_dotenv()

app = Flask(__name__)

@app.route("/")
def index():
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/train_universal")
def train_universal_route():
    try:
        train_universal_model()
        message = "Universal model successfully trained using data from the S&P 500."
    except Exception as e:
        message = f"Error training universal model: {e}"
    return render_template("message.html", message=message)

@app.route("/backtest_chart", methods=["GET", "POST"])
def backtest_chart_route():
    tickers = get_sp500_tickers()
    symbol = ""
    if request.method == "POST":
        symbol = request.form.get("symbol", "").upper()
    else:
        symbol = request.args.get("symbol", "").upper()
    
    if symbol == "":
        return render_template("backtest_chart.html", tickers=tickers, symbol=None)
    else:
        df = backtest_universal_model(symbol)
        start_date = df["Date"].iloc[0].strftime("%Y-%m-%d")
        end_date = df["Date"].iloc[-1].strftime("%Y-%m-%d")
        portfolio_dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()
        portfolio_values = df["Portfolio Value"].tolist()
        return render_template("backtest_chart.html",
                               tickers=tickers,
                               symbol=symbol,
                               portfolio_dates=portfolio_dates,
                               portfolio_values=portfolio_values,
                               start_date=start_date,
                               end_date=end_date)

@app.route("/live_trading_dashboard", methods=["GET", "POST"])
def live_trading_dashboard():
    symbol = ""
    if request.method == "POST":
        symbol = request.form.get("symbol", "").upper()
    else:
        symbol = request.args.get("symbol", "").upper()
    
    if symbol == "":
        return render_template("live_trading_dashboard.html", symbol=None)
    
    bot = AlpacaBot()
    try:
        account = bot.get_account()
        account_info = {
            "Equity": account.equity,
            "Cash": account.cash,
            "Buying Power": account.buying_power,
            "Portfolio Value": account.portfolio_value,
            "Status": account.status
        }
        error_msg = None
    except Exception as e:
        account_info = None
        error_msg = f"Error retrieving account info: {e}"
    
    trading_running = live_trading_thread is not None and live_trading_thread.is_alive()
    
    profit_loss = None
    if initial_equity is not None:
        try:
            current_equity = float(account.equity)
            profit_loss = ((current_equity - initial_equity) / initial_equity) * 100
        except Exception:
            profit_loss = None
    
    return render_template("live_trading_dashboard.html",
                           symbol=symbol,
                           account_info=account_info,
                           error_msg=error_msg,
                           trading_running=trading_running,
                           account_history=account_history,
                           profit_loss=profit_loss,
                           trade_history=trade_history)

@app.route("/start_live_trading")
def start_live_trading_route():
    global live_trading_thread, trading_stop_event
    symbol = request.args.get("symbol", "").upper()
    if symbol == "":
        return redirect(url_for("live_trading_dashboard"))
    if live_trading_thread is None or not live_trading_thread.is_alive():
        trading_stop_event.clear()
        live_trading_thread = threading.Thread(target=live_trading_loop, args=(symbol,), daemon=True)
        live_trading_thread.start()
    return redirect(url_for("live_trading_dashboard", symbol=symbol))

@app.route("/stop_live_trading")
def stop_live_trading_route():
    global trading_stop_event
    if live_trading_thread is not None and live_trading_thread.is_alive():
        trading_stop_event.set()
    symbol = request.args.get("symbol", "").upper()
    return redirect(url_for("live_trading_dashboard", symbol=symbol))

@app.route("/message")
def message():
    msg = request.args.get("msg", "No message provided.")
    return render_template("message.html", message=msg)

if __name__ == "__main__":
    app.run(debug=True)

