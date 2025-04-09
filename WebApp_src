import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from datetime import datetime, timedelta

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# css
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput input {
        color: #FFFFFF !important;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .prediction-up {
        color: #4CAF50;
        font-weight: bold;
        font-size: 1.2em;
    }
    .prediction-down {
        color: #F44336;
        font-weight: bold;
        font-size: 1.2em;
    }
</style>
""", unsafe_allow_html=True)

# title
st.title('Stock Price Predictor')

# sidebar
st.sidebar.header('User Input Parameters')

def get_input():
    ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper();
    start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365*5));
    end_date = st.sidebar.date_input("End Date", datetime.now());
    future_days = st.sidebar.slider("Days to Predict Ahead", 1, 90, 30);
    return ticker, start_date, end_date, future_days;

# load and process stock data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end);
        if data.empty:
            st.error("No data found for this ticker. Please try another one.");
            return None;
        
        # moving averages
        data['MA_50'] = data['Close'].rolling(window=50).mean();
        data['MA_200'] = data['Close'].rolling(window=200).mean();
        data = data.dropna();
        
        return data;
    except Exception as e:
        st.error(f"Error fetching data: {e}");
        return None;

# predict future prices
def predict_future(model, last_sequence, scaler, future_days):
    future_predictions = [];
    current_sequence = last_sequence.copy();
    
    for _ in range(future_days):
        # next day
        next_pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1]))[0, 0];
        future_predictions.append(next_pred);
        
        # update sequence
        current_sequence = np.roll(current_sequence, -1, axis=0);
        current_sequence[-1, 0] = next_pred;
        current_sequence[-1, 1] = current_sequence[-2, 1];
        current_sequence[-1, 2] = current_sequence[-2, 2];
    
    # closer scaler (inverse transform)
    future_prices = scaler.inverse_transform(
        np.concatenate((
            np.array(future_predictions).reshape(-1, 1),
            np.zeros((len(future_predictions), 2))
        ), axis=1)
    )[:, 0];
    
    return future_prices;

# main
def main():
    ticker, start_date, end_date, future_days = get_input();
    
    if st.sidebar.button('Predict'):
        with st.spinner('Fetching data and making predictions...'):
            data = load_data(ticker, start_date, end_date);
            if data is None:
                return;
            
            # raw data
            st.subheader(f'Raw Data for {ticker}');
            st.write(data.tail());
            
            # plot for closing price and moving averages
            st.subheader('Closing Price with Moving Averages');
            fig, ax = plt.subplots(figsize=(12,6));
            ax.plot(data['Close'], label='Close Price', alpha=0.8);
            ax.plot(data['MA_50'], 'r', label='50-day MA', alpha=0.6);
            ax.plot(data['MA_200'], 'g', label='200-day MA', alpha=0.6);
            ax.set_xlabel('Date');
            ax.set_ylabel('Price ($)');
            ax.legend();
            st.pyplot(fig);
            
            # prepare data for training with moving averages
            df = data[['Close', 'MA_50', 'MA_200']];
            
            # split data
            train_size = int(len(df) * 0.70);
            training = df.iloc[:train_size];
            testing = df.iloc[train_size:];
            
            # scale data
            scaler = MinMaxScaler(feature_range=(0,1));
            training_scaled = scaler.fit_transform(training);
            testing_scaled = scaler.transform(testing);
            
            # create training sequences
            x_train, y_train = [], [];
            for i in range(100, training_scaled.shape[0]):
                x_train.append(training_scaled[i-100:i]);
                y_train.append(training_scaled[i, 0]);
            
            x_train, y_train = np.array(x_train), np.array(y_train);
            
            # create and train model
            model = Sequential();
            model.add(LSTM(units=50, activation='relu', return_sequences=True, 
                         input_shape=(x_train.shape[1], x_train.shape[2])));
            model.add(Dropout(0.2));
            model.add(LSTM(units=60, activation='relu'));
            model.add(Dropout(0.3));
            model.add(Dense(units=1));
            model.compile(optimizer='adam', loss='mean_squared_error');
            model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0);
            
            # prepare test data
            past100days = training.tail(100);
            df_final = pd.concat([past100days, testing], ignore_index=True);
            input_data = scaler.transform(df_final);
            
            # create test sequences
            x_test, y_test = [], [];
            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i-100:i]);
                y_test.append(input_data[i, 0]);
            
            x_test, y_test = np.array(x_test), np.array(y_test);
            
            # predictions
            predicted_prices = model.predict(x_test);
            
            # inverse transform predictions (only close price)
            predicted_prices = scaler.inverse_transform(
                np.concatenate((
                    predicted_prices, 
                    np.zeros((predicted_prices.shape[0], 2))
                ), axis=1)
            )[:, 0];
            
            y_test = scaler.inverse_transform(
                np.concatenate((
                    y_test.reshape(-1,1), 
                    np.zeros((y_test.shape[0], 2))
                ), axis=1)
            )[:, 0];

            def calculate_accuracy(y_true, y_pred):
                correct = 0;
                for i in range(1, len(y_true)):
                    if (y_true[i] > y_true[i-1]) == (y_pred[i] > y_pred[i-1]):
                        correct += 1;
                return (correct / (len(y_true)-1)) * 100;

            accuracy = calculate_accuracy(y_test.flatten(), predicted_prices.flatten());
            st.metric("Model Directional Accuracy", f"{accuracy:.1f}%");
            
            # predict future prices
            last_sequence = input_data[-100:];
            future_prices = predict_future(model, last_sequence, scaler, future_days);
            
            # create date range for future predictions
            last_date = data.index[-1];
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=future_days);
            
            # combine actual and predicted data for plotting
            all_dates = np.concatenate([data.index[-len(y_test):], future_dates]);
            all_actual = np.concatenate([y_test.flatten(), [np.nan]*future_days]);
            all_predicted = np.concatenate([predicted_prices.flatten(), future_prices]);
            
            # plot predictions with future forecast
            st.subheader('Price Prediction with Future Forecast');
            fig, ax = plt.subplots(figsize=(12,6));
            
            # convert to datetime
            all_dates = pd.to_datetime(all_dates);
            
            # plot actual prices
            ax.plot(all_dates[:len(y_test)], all_actual[:len(y_test)], 'b-', linewidth=2, label='Actual Price');
            
            # plot predicted prices
            ax.plot(all_dates[:len(y_test)], all_predicted[:len(y_test)], 'r-', linewidth=2, label='Predicted Price by Model)');
            
            # plot future predictions
            ax.plot(all_dates[len(y_test):], all_predicted[len(y_test):], 'g--', linewidth=2, label=f'Future Prediction ({future_days} days)');
            
            # vertical line to separate
            ax.axvline(x=last_date, color='k', linestyle='--', alpha=0.5);
            
            # format plot
            ax.set_xlabel('Date');
            ax.set_ylabel('Price ($)');
            ax.legend(loc='upper left');
            ax.grid(True, which='both', linestyle='--', alpha=0.5);
            plt.xticks(rotation=45);
            
            # x-axis
            if len(all_dates) > 10:
                step = max(1, len(all_dates) // 10);
                ax.set_xticks(all_dates[::step]);
            
            st.pyplot(fig);
            
            # 30-day trend for up or down
            if future_days >= 30:
                future_30 = future_prices[:30];
                price_change = future_30[-1] - future_30[0];
                percent_change = (price_change / future_30[0]) * 100;
                
                if price_change > 0:
                    trend = "UP";
                    trend_class = "prediction-up";
                    arrow = "↑";
                else:
                    trend = "DOWN";
                    trend_class = "prediction-down";
                    arrow = "↓";
                
                st.markdown(f"""
                ### 30-Day Price Trend Prediction
                <div class="{trend_class}">
                    Expected trend: {trend} {arrow}<br>
                    Predicted change: {abs(percent_change):.2f}%
                </div>
                """, unsafe_allow_html=True);
            
            st.success('Prediction completed!');

if __name__ == '__main__':
    main();
