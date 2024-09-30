import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

# Set up logging
log_file = 'logs/evaluation.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(model, X_test, y_test):
    try:
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        
        logging.info(f"Evaluation Metrics - MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

        # Plot true vs predicted values
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='True Values', color='blue')
        plt.plot(predictions, label='Predictions', color='orange')
        plt.title('True Values vs Predictions')
        plt.xlabel('Time Step')
        plt.ylabel('Stock Price')
        plt.legend()
        
        # Save the plot
        plt.savefig('logs/evaluation_plot.png')
        plt.show()

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")


from keras.models import load_model

def predict_prices(model, data, time_step=10):
    last_data = data[-time_step:].reshape(1, time_step, 1)
    predicted_price = model.predict(last_data)
    return predicted_price




# Load the trained model
model = load_model('D:/Deep Learning for Risk Management in Financial Markets/lstm_model.h5')

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Step 1: Fetch Current Price
ticker_symbol = 'AAPL'
current_data = yf.download(ticker_symbol, period='1d', interval='1m')
current_price = current_data['Close'].iloc[-1]
print(f"Current Price of {ticker_symbol}: {current_price:.2f}")

# Step 2: Fetch Historical Data
historical_data = yf.download(ticker_symbol, start='2023-01-01', end='2024-09-01')
#closing_prices = historical_data['Close'].values.reshape(-1, 1)

features = historical_data[['Open', 'High', 'Low', 'Close', 'Adj Close','Volume']].values


# Step 3: Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features)

# Step 4: Prepare Data for Prediction
def create_input_sequence(data, time_step=60):
    # Reshape the last `time_step` data points for prediction
    return data[-time_step:].reshape(1, time_step, data.shape[1])  # Change the last 1 to the number of features


# Create input sequence for prediction (adjust the shape based on features)
input_sequence = create_input_sequence(scaled_data)

# Load your trained LSTM model
model = load_model('D:/Deep Learning for Risk Management in Financial Markets/lstm_model.h5')

model.summary()

print("Input sequence shape:", input_sequence.shape)

predicted_price_scaled = model.predict(input_sequence)

# Check the shape of predicted_price_scaled
print("Predicted price shape:", predicted_price_scaled.shape)

if predicted_price_scaled.shape[1] == 1:  # If it outputs only one feature
    predicted_price_scaled = np.concatenate((predicted_price_scaled, np.zeros((1, 5))), axis=1)
# Step 5: Make Prediction


# Inverse transform to get the actual price
predicted_price = scaler.inverse_transform(predicted_price_scaled)
print(f"Predicted Price for the next period: {predicted_price[0][3]:.2f}")  # Assuming index 3 is 'Close'
