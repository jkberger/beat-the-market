# Universal Trading App

This is a Flask-based web application that:
- Trains a universal machine learning model on historical data from all S&P 500 stocks.
- Backtests the trained model on individual stocks.
- Integrates with Alpaca’s paper trading API for live trading simulation.
- Provides a web dashboard with options to train the model, view backtest charts, and run live trading.

## Prerequisites

- **Python 3.7 or higher**  
- **Alpaca Paper Trading Account:**  
  You need API credentials (an API key and secret) from Alpaca.  
- **Internet Connection:**  
  The app fetches historical data via the yfinance API and interacts with Alpaca.

## Setup Instructions

### 1. Clone the Repository

Clone the repository from GitHub (or download the source code):

```bash
> git clone <repository_url>
> cd <repository_directory>

### 2. Create a Virtual Environment
```MacOs or linux
> python3 -m venv venv
> source venv/bin/activate

```Windows
> python -m venv venv
> venv\Scripts\activate

### 3. Install Dependencies
A requirements.txt file is provided. Install the required packages:
> pip install -r requirements.txt

If for some reason you dont have the requirements.txt file you can run: 
> pip install flask numpy pandas scikit-learn yfinance alpaca-trade-api plotly

### 4. Configure Alpaca API Credentials
Create a .env file at the root level of this repository and add the 
following two lines:
> ALPACA_API_KEY=your_api_key_here
> ALPACA_SECRET_KEY=your_secret_key_here

### 5. Run the application
> python3 trading_app.py
Open the app to try out features and debug testing at http://127.0.0.1:5000 in your web browser
