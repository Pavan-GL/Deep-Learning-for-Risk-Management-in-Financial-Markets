import pandas as pd
import numpy as np
import yfinance as yf
import logging
from textblob import TextBlob
from data_preprocessing import load_data,preprocess_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def add_technical_indicators(df):
    try:
        logging.info("Calculating technical indicators.")
        
        # Calculate Moving Average (MA)
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Calculate Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        logging.info("Technical indicators added successfully.")
        return df
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        raise

def sentiment_analysis(news_data):
    try:
        logging.info("Performing sentiment analysis on news articles.")
        
        # Assuming news_data is a DataFrame with a column 'article'
        sentiments = []
        for article in news_data['article']:
            analysis = TextBlob(article)
            sentiments.append(analysis.sentiment.polarity)

        news_data['sentiment'] = sentiments
        logging.info("Sentiment analysis completed successfully.")
        return news_data
    except Exception as e:
        logging.error(f"Error during sentiment analysis: {e}")
        raise

def main():
    ticker = 'AAPL'  # Example ticker
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    
    try:
        df = load_data(ticker, start_date, end_date)
        processed_data = preprocess_data(df)
        processed_data_with_indicators = add_technical_indicators(processed_data)
        
        logging.info("Processed data with technical indicators:")
        logging.info(processed_data_with_indicators.head())
        
        # Example news data for sentiment analysis
        news_data = pd.DataFrame({
            'article': [
                "Apple's stock rises as demand increases.",
                "Analysts predict a downturn in the tech sector.",
                "New product launch boosts investor confidence."
            ]
        })
        
        analyzed_news = sentiment_analysis(news_data)
        logging.info("Sentiment analysis results:")
        logging.info(analyzed_news)
        
    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")

if __name__ == "__main__":
    main()
