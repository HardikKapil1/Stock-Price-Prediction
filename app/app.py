import streamlit as st
import yfinance as yf

st.title("📈 Stock Price Prediction (LSTM)")

# Step 1: Input
ticker = st.text_input("Enter Stock Symbol", "AAPL")

# Step 2: Fetch Data
if st.button("Load Data"):
    df = yf.download(ticker, start="2015-01-01")

    st.write("### Raw Data")
    st.dataframe(df.tail())

    # Step 3: Plot
    st.write("### Closing Price Chart")
    st.line_chart(df["Close"])