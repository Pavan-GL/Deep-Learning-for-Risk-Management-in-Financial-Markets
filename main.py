import os
import logging
import numpy as np
from src.data_preprocessing import load_data, preprocess_data
#from src.data_preprocessing import load_data, preprocess_data
from src.feature_eng import add_technical_indicators
from src.model import create_model
from src.model import train_model
from src.evaluate import evaluate_model
from src.dashboard import app
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Set up logging
log_file = 'logs/training.log'
if not os.path.exists('logs'):
    os.makedirs('logs')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_training_data(scaled_data, time_step=60):
    X, y = [], []
    for i in range(len(scaled_data) - time_step):
        X.append(scaled_data[i:i + time_step, 0])  # Input sequence
        y.append(scaled_data[i + time_step, 0])    # Next value
    return np.array(X), np.array(y)

if __name__ == '__main__':
    try:
        # Load and preprocess data
        df = load_data('AAPL', '2020-01-01', '2024-01-01')
        logging.info(f"DataFrame shape after loading: {df.shape}")
        
        df = preprocess_data(df)
        logging.info(f"DataFrame shape after preprocessing: {df.shape}")
        
        df = add_technical_indicators(df)
        logging.info(f"DataFrame shape after adding technical indicators: {df.shape}")

        

        # Prepare data for training/testing
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['Close']].values)
        X, y = prepare_training_data(scaled_data)
        
        # Reshape X for LSTM input
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info(f"Training data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")

        # Create and train model
        model = create_model((X_train.shape[1], 1))
        train_model(model, (X_train, y_train))

        # Evaluate model
        evaluate_model(model, X_test, y_test)

        # Start the dashboard
        app.run_server(debug=True)

    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")
