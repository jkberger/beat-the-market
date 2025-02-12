import os
import threading
#from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for
from utils.data_fetching import get_sp500_tickers
from utils.model_training import train_custom_model, train_model_with_tuning, DEFAULT_PERIODS
from utils.backtesting import backtest_model
from utils.alpaca_bot import AlpacaBot
from utils.live_trading import live_trading_loop, live_trading_thread, trading_stop_event, trade_history, account_history, initial_equity
from utils.helpers import get_available_models
import json

#load_dotenv()

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

# Combined Training Route: Uses either standard training or tuned training based on the selected algorithm.
@app.route("/train_model_custom", methods=["POST"])
def train_model_custom_route():
    print("Form submission:", dict(request.form))
    
    buy_conditions = []
    sell_conditions = []
    
    # Extract all BUY conditions
    for key in request.form:
        if key.startswith("buy_indicator_"):
            suffix = key.split("_")[-1]
            ind = request.form.get(f"buy_indicator_{suffix}", "").strip()
            op = request.form.get(f"buy_operator_{suffix}", "").strip()
            thr = request.form.get(f"buy_threshold_{suffix}", "").strip()
            if ind and op and thr:
                from utils.model_training import DEFAULT_PERIODS
                period = DEFAULT_PERIODS.get(ind.upper(), 14)
                buy_conditions.append((ind, period, op, thr))
    
    # Extract all SELL conditions
    for key in request.form:
        if key.startswith("sell_indicator_"):
            suffix = key.split("_")[-1]
            ind = request.form.get(f"sell_indicator_{suffix}", "").strip()
            op = request.form.get(f"sell_operator_{suffix}", "").strip()
            thr = request.form.get(f"sell_threshold_{suffix}", "").strip()
            if ind and op and thr:
                from utils.model_training import DEFAULT_PERIODS
                period = DEFAULT_PERIODS.get(ind.upper(), 14)
                sell_conditions.append((ind, period, op, thr))
    
    print("Buy conditions parsed:", buy_conditions)
    print("Sell conditions parsed:", sell_conditions)
    
    model_name = request.form.get("model_name", "Custom").strip()
    training_type = request.form.get("training_type", "standard")
    
    try:
        if training_type == "tuned":
            model_type = request.form.get("model_type", "RandomForest")
            from utils.model_training import train_model_with_tuning
            model_data, df = train_model_with_tuning(buy_conditions, sell_conditions, model_name=model_name, model_type=model_type, interval="5m")
            # Format metrics for display as percentages.
            cv_score_percent = model_data.get('cv_score', 0) * 100
            test_accuracy_percent = model_data.get('test_accuracy', 0) * 100
            
            # Render the model_performance template with all performance details.
            return render_template(
                "model_performance.html",
                model_name=model_name,
                model_type=model_type,
                best_params=model_data.get('best_params', {}),
                cv_score_percent=f"{cv_score_percent:.2f}",
                test_accuracy_percent=f"{test_accuracy_percent:.2f}",
                classification_report=model_data.get('classification_report', '')
            )
        else:
            from utils.model_training import train_custom_model
            train_custom_model(buy_conditions, sell_conditions, model_name=model_name, interval="5m")
            message = f"<h3>Custom model '{model_name}' trained successfully using standard training.</h3>"
            message += '<br><a href="/dashboard" class="btn btn-secondary">Back to Dashboard</a>'
            return render_template("message.html", message=message)
    except Exception as e:
        message = f"Error training custom model: {e}"
        return render_template("message.html", message=message)

# (Other routes remain unchanged.)

@app.route("/backtest_chart", methods=["GET", "POST"])
def backtest_chart_route():
    available_models = get_available_models()
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
        try:
            df, actions, hold_stock, hold_spy = backtest_model(symbol, selected_model, interval="5m")
        except Exception as e:
            return render_template("message.html", message=f"Error in backtesting: {e}")
        
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
                               hold_stock=hold_stock,
                               hold_spy=hold_spy,
                               actions=actions)

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
