# trading_app.py
import os
import threading
from flask import Flask, render_template, request, redirect, url_for
from utils.data_fetching import get_sp500_tickers
from utils.model_training import train_custom_model
from utils.backtesting import backtest_model
from utils.alpaca_bot import AlpacaBot
from utils.live_trading import live_trading_loop, live_trading_thread, trading_stop_event, trade_history, account_history, initial_equity

app = Flask(__name__)

# Route: Home redirects to Dashboard
@app.route("/")
def index():
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# Route: Train Model Parameters page
@app.route("/train_model_parameters")
def train_model_parameters():
    available_indicators = ["SMA", "EMA", "RSI", "MACD", "ATR", "CCI", "ADX", "OBV", "STOCH"]
    operators = [">", "<", ">=", "<=", "="]
    supported_note = "Indicators computed via ta: SMA, EMA, RSI, MACD, ATR, CCI, ADX, OBV, STOCH."
    return render_template("train_model_parameters.html",
                           available_indicators=available_indicators,
                           operators=operators,
                           supported_note=supported_note)

# Route: Train Custom Model (Process form submission)
@app.route("/train_model_custom", methods=["POST"])
def train_model_custom_route():
    buy_conditions = []
    sell_conditions = []
    # Process Buy Conditions (expecting keys like "buy_indicator_1", etc.)
    buy_keys = sorted([key for key in request.form if key.startswith("buy_indicator_")],
                      key=lambda k: int(k.split("_")[2]))
    for key in buy_keys:
        suffix = key.split("_")[2]
        ind = request.form.get(f"buy_indicator_{suffix}", "").strip()
        period = request.form.get(f"buy_period_{suffix}", "").strip()
        op = request.form.get(f"buy_operator_{suffix}", "").strip()
        thr = request.form.get(f"buy_threshold_{suffix}", "").strip()
        if ind and period and op and thr:
            try:
                period_val = int(period)
            except ValueError:
                continue
            buy_conditions.append((ind.upper(), period_val, op, thr))
    # Process Sell Conditions
    sell_keys = sorted([key for key in request.form if key.startswith("sell_indicator_")],
                       key=lambda k: int(k.split("_")[2]))
    for key in sell_keys:
        suffix = key.split("_")[2]
        ind = request.form.get(f"sell_indicator_{suffix}", "").strip()
        period = request.form.get(f"sell_period_{suffix}", "").strip()
        op = request.form.get(f"sell_operator_{suffix}", "").strip()
        thr = request.form.get(f"sell_threshold_{suffix}", "").strip()
        if ind and period and op and thr:
            try:
                period_val = int(period)
            except ValueError:
                continue
            sell_conditions.append((ind.upper(), period_val, op, thr))
    model_name = request.form.get("model_name", "Custom").strip()
    if not buy_conditions or not sell_conditions:
        message = "You must provide at least one condition for both BUY and SELL actions."
        return render_template("message.html", message=message)
    try:
        train_custom_model(buy_conditions, sell_conditions, model_name=model_name)
        message = f"Custom model '{model_name}' successfully trained using your parameters."
    except Exception as e:
        message = f"Error training custom model: {e}"
    return render_template("message.html", message=message)

# Route: Backtest Chart
@app.route("/backtest_chart", methods=["GET", "POST"])
def backtest_chart_route():
    available_models = []
    models_dir = "models"
    if os.path.exists(models_dir):
        files = os.listdir(models_dir)
        available_models = [os.path.splitext(f)[0] for f in files if f.endswith(".pkl")]
    tickers = get_sp500_tickers()
    if request.method == "POST":
        selected_model = request.form.get("model", "Custom").strip()
        symbol = request.form.get("symbol", "").upper()
    else:
        selected_model = request.args.get("model", "Custom").strip()
        symbol = request.args.get("symbol", "").upper()
    if not selected_model:
        selected_model = "Custom"
    if symbol == "":
        return render_template("backtest_chart.html",
                               available_models=available_models,
                               selected_model=selected_model,
                               tickers=tickers,
                               symbol=None)
    else:
        df, actions = backtest_model(symbol, selected_model)
        start_date = df["Date"].iloc[0].strftime("%Y-%m-%d")
        end_date = df["Date"].iloc[-1].strftime("%Y-%m-%d")
        portfolio_dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()
        portfolio_values = df["Portfolio Value"].tolist()
        return render_template("backtest_chart.html",
                               available_models=available_models,
                               selected_model=selected_model,
                               tickers=tickers,
                               symbol=symbol,
                               start_date=start_date,
                               end_date=end_date,
                               portfolio_dates=portfolio_dates,
                               portfolio_values=portfolio_values,
                               actions=actions)

# Route: Live Trading Dashboard
@app.route("/live_trading_dashboard", methods=["GET", "POST"])
def live_trading_dashboard():
    global initial_equity
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
    trading_running = False
    if 'live_trading_thread' in globals() and live_trading_thread is not None:
        trading_running = live_trading_thread.is_alive()
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

# Route: Start Live Trading
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

# Route: Stop Live Trading
@app.route("/stop_live_trading")
def stop_live_trading_route():
    global trading_stop_event
    if live_trading_thread is not None and live_trading_thread.is_alive():
        trading_stop_event.set()
    symbol = request.args.get("symbol", "").upper()
    return redirect(url_for("live_trading_dashboard", symbol=symbol))

# Route: Message Display
@app.route("/message")
def message():
    msg = request.args.get("msg", "No message provided.")
    return render_template("message.html", message=msg)

if __name__ == "__main__":
    app.run(debug=True)
