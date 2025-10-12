# app.py

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta # Used for fetching latest data

# --- Configuration & Loading Assets ---

# Load the trained LSTM model
@st.cache_resource # Caches the model so it loads only once
def get_model():
    # Make sure this file name matches what you saved!
    model = load_model('lstm_model_final.keras') 
    return model

# Load the trained scaler object
@st.cache_resource
def get_scaler():
    # Make sure this file name matches what you saved!
    with open('scaler_object.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return scaler

# Constants from your training:
TIME_STEP = 60 # Model needs 60 days of history for one prediction
TODAY = date.today()

model = get_model()
scaler = get_scaler()

# --- Streamlit Interface Code ---

st.title('📈 Stock Price Prediction using Deep Learning (LSTM)')
st.markdown('***A Final Year Project Demonstration***')

# 1. User Input for Ticker
# Use a sidebar for user inputs
st.sidebar.header('User Input')
ticker_symbol = st.sidebar.text_input('Enter Stock Ticker (e.g., AAPL)', 'AAPL')

# 2. Prediction Button
if st.sidebar.button('Predict Next Day Price'):
    
    # 3. Data Retrieval
    end_date = TODAY.strftime('%Y-%m-%d')
    start_date = (TODAY - timedelta(days=TIME_STEP + 30)).strftime('%Y-%m-%d') 
    # Fetch 90 days of data to ensure we have enough for the 60-day window
    
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    
    if data.empty:
        st.error(f"Could not retrieve data for ticker: {ticker_symbol}. Check the symbol.")
    else:
        st.subheader(f'Prediction for {ticker_symbol}')
        
        # 4. Preprocessing (MUST match training!)
        data_close = data['Close'].values.reshape(-1, 1) # Reshape to (X, 1)
        scaled_data = scaler.transform(data_close) # Use the loaded scaler!
        
        # Get the last 60 days for prediction
        X_predict = scaled_data[-TIME_STEP:].reshape(1, TIME_STEP, 1) # Shape (1, 60, 1)
        
        # 5. Prediction
        predicted_scaled_price = model.predict(X_predict)
        
        # 6. Inverse Transform (Convert back to USD)
        predicted_price_usd = scaler.inverse_transform(predicted_scaled_price)[0, 0]
        
        # 7. Display Results
        st.success(f"**Predicted Closing Price for Tomorrow:** ${predicted_price_usd:.2f}")
        st.markdown(f"*(Based on the last {TIME_STEP} trading days)*")
        
        # Simple visualization of the last 60 days
        st.line_chart(data['Close'].tail(TIME_STEP))
        st.markdown('---')
        st.caption('Disclaimer: This is a project demonstration for educational purposes only.')