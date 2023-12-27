import yfinance as yf
import pandas as pd

def get_stock_data(symbol, start_date, end_date):
    # Download historical stock data
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

def main():
    # Define the stock symbol (TSLA for Tesla)
    stock_symbol = "TSLA"

    # Define the date range for the last 5 years
    end_date = pd.to_datetime('today')
    start_date = end_date - pd.DateOffset(years=5)

    # Get historical stock data
    stock_data = get_stock_data(stock_symbol, start_date, end_date)

    # Display the retrieved data
    print("Historical Stock Data for", stock_symbol)
    print(stock_data.head())

if __name__ == "__main__":
    main()