import pandas as pd
import numpy as np
import yfinance as yf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(ticker, start_date, end_date):
    try:
        logging.info(f"Fetching data for {ticker} from {start_date} to {end_date}.")
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            raise ValueError("No data fetched. Please check the ticker symbol and date range.")
        logging.info("Data fetched successfully.")
        return df
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        raise

def preprocess_data(df):
    try:
        logging.info("Starting data preprocessing.")
        
        # Handle missing values
        if df.isnull().values.any():
            logging.warning("Missing values detected. Filling missing values with forward fill.")
            df.fillna(method='ffill', inplace=True)

        # Normalize data (Min-Max normalization)
        logging.info("Normalizing data.")
        normalized_df = (df - df.min()) / (df.max() - df.min())
        
        logging.info("Data preprocessing completed successfully.")
        print(normalized_df,"normal")
        return normalized_df , None
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

def main():
    ticker = 'AAPL'  # Example ticker
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    
    #try:
    df = load_data(ticker, start_date, end_date)
    processed_data = preprocess_data(df)
    logging.info("Processed data:")
    logging.info(processed_data.head())
    # except Exception as e:
    #     logging.error(f"An error occurred in the main execution: {e}")

if __name__ == "__main__":
    main()
