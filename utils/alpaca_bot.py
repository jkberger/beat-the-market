# utils/alpaca_bot.py
import os
import alpaca_trade_api as tradeapi

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
