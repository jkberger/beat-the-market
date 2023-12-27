import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Todo: Change this to take in list of symbols as first parameter
def download_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data


def preprocess_data(data):
    # Feature engineering - here we use the 'Close' price as the target variable and shift it by 1 day
    data['Target'] = data['Close'].shift(-1)

    # Drop rows with missing values
    data = data.dropna()

    # Select features and target variable
    features = data[['Open', 'High', 'Low', 'Volume']]
    target = data['Target']

    return features, target


def train_model(features, target):
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Initialize a RandomForestRegressor (you can experiment with different models)
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(x_train, y_train)

    # Predictions on the test set
    predictions = model.predict(x_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    return model


def visualize_results(data, predictions):
    plt.figure(figsize=(12, 6))

    # Plot the actual stock prices
    plt.plot(data.index[-len(predictions):], data['Close'].tail(len(predictions)), label='Actual Prices')

    # Plot the predicted prices
    plt.plot(data.index[-len(predictions):], predictions, label='Predicted Prices', linestyle='dashed')

    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


def main():
    # Download historical stock data
    # Todo: Change this to stock_symbols to take in a list
    stock_symbol = 'TSLA'  # Change this to the desired stock symbol
    start_date = '2018-01-01'
    end_date = '2023-01-01'
    # Todo: Change this function to send a list of stock symbols
    stock_data = download_stock_data(stock_symbol, start_date, end_date)

    # Preprocess the data
    features, target = preprocess_data(stock_data)

    # Train the machine learning model
    model = train_model(features, target)

    # Make predictions on the last 20% of the data
    predictions = model.predict(features[-int(len(features) * 0.2):])

    # Visualize the results
    visualize_results(stock_data, predictions)


if __name__ == "__main__":
    main()
