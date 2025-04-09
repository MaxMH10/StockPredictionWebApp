# Stock Prediction WebApp
Stock Price Predictor with LSTM
This project is a Streamlit-based web application that predicts future stock prices using Long Short-Term Memory (LSTM) neural networks. It incorporates technical indicators like 50-day and 200-day moving averages to enhance prediction accuracy. The app fetches real-time stock data from Yahoo Finance, trains an LSTM model, and provides visualizations of historical trends, model predictions, and future forecasts.

Features
Real-time Data Integration: Fetches historical stock prices using the Yahoo Finance API.

Technical Indicators: Calculates and visualizes 50-day and 200-day moving averages.

LSTM Model: Implements a deep learning model with dropout layers to prevent overfitting.

Visualizations: Displays historical prices, model predictions, and future forecasts (up to 90 days) using Matplotlib.

30-Day Trend Analysis: Provides a summary of expected price movement (up/down) and percentage change over the next 30 days.

Customizable Inputs: Allows users to select stock tickers, date ranges, and prediction horizons.

Setup Instructions:

Prerequisites:
Python 3.8 or later and pip package manager

Run the following in terminal:

Git clone https://github.com/MaxMH10/StockPredictionWebApp.git

cd StockPredictionWebApp

pip install -r requirements.txt

streamlit run StockPredictionWebApp.py
