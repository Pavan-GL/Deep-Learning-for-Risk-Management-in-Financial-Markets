import pandas as pd
import numpy as np
import yfinance as yf
import logging
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from data_preprocessing import load_data,preprocess_data
import os

# Set up logging
log_file = 'training.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_model(input_shape):
    try:
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(1))  # Output layer
        model.compile(optimizer='adam', loss='mean_squared_error')
        logging.info("Model created successfully.")
        return model
    except Exception as e:
        logging.error(f"Error creating model: {e}")
        raise

def train_model(model, training_data):
    try:
        X_train, y_train = training_data
        model.fit(X_train, y_train, epochs=200, batch_size=16)
        logging.info("Model training completed successfully.")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

def save_model(model, model_name):
    try:
        model.save(model_name)
        logging.info(f"Model saved successfully to {model_name}.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def prepare_training_data(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])  # Features
        y.append(data[i + time_step])       # Target
    return np.array(X), np.array(y)








def main():
    ticker = 'AAPL'  # Example ticker
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    
    try:
        df = load_data(ticker, start_date, end_date)
        scaled_data, scaler = preprocess_data(df)
        print(scaled_data)
    

        # Prepare training data
        time_step = 60
        if len(scaled_data) < time_step:
            raise ValueError(f"Not enough data points: {len(scaled_data)} to use time step of {time_step}")
        X, y = prepare_training_data(scaled_data.values, time_step)  # Ensure to use .values
        
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
        
      
        
        # Create and train model
        model = create_model((X.shape[1], X.shape[2]))
        train_model(model, (X, y))
        
        # Save the model
        save_model(model, 'lstm_model.h5')

    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")

if __name__ == "__main__":
    if not os.path.exists('logs'):
        os.makedirs('logs')
    main()
