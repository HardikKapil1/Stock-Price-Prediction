import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import yaml
from pathlib import Path
import yfinance as yf

st.set_page_config(
    page_title="Stock Market Prediction | LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card h3 {
        color: #e94560;
        font-size: 14px;
        margin-bottom: 5px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card p {
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        color: #00d4ff;
    }

    .header-container {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 25px 30px;
        border-radius: 15px;
        margin-bottom: 20px;
        border: 1px solid #533483;
    }
    .header-container h1 {
        color: #ffffff;
        font-weight: 700;
        margin: 0;
        font-size: 28px;
    }
    .header-container p {
        color: #b0b0b0;
        margin: 5px 0 0 0;
        font-size: 14px;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 500;
    }

    .info-box {
        background: #1a1a2e;
        border-left: 4px solid #e94560;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        color: #e0e0e0;
    }

    div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_config():
    config_path = Path(__file__).parent.parent / 'config.yaml'
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return None


@st.cache_data(ttl=300)
def fetch_live_data(ticker, period="1y"):
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def load_metrics():
    metrics_path = Path(__file__).parent.parent / 'outputs' / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


def load_predictions():
    pred_path = Path(__file__).parent.parent / 'outputs' / 'predictions.csv'
    if pred_path.exists():
        return pd.read_csv(pred_path)
    return None


def load_training_history():
    hist_path = Path(__file__).parent.parent / 'outputs' / 'training_history.json'
    if hist_path.exists():
        with open(hist_path) as f:
            return json.load(f)
    return None


st.markdown("""
<div class="header-container">
    <h1>📈 Stock Market Prediction Using LSTM Networks</h1>
    <p>Final Year Project — Stacked LSTM Model for Close Price Forecasting | Meerut Institute of Engineering & Technology (MIET)</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    config = load_config()

    if config:
        ticker = st.text_input("Stock Ticker", value=config.get('Ticker', 'RELIANCE.NS'))
        st.markdown("---")
        st.markdown("### 📋 Model Parameters")
        st.markdown(f"""
        - **Sequence Length:** {config.get('SequenceLength', 60)} days
        - **LSTM Units:** {config.get('LSTMUnits', 50)}
        - **LSTM Layers:** {config.get('NumLayers', 2)}
        - **Dropout:** {config.get('Dropout', 0.2)}
        - **Optimizer:** Adam
        - **Loss:** MSE
        - **Train/Test:** 80/20
        """)
    else:
        ticker = st.text_input("Stock Ticker", value="RELIANCE.NS")

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This application demonstrates a **Long Short-Term Memory (LSTM)**
    neural network for predicting stock closing prices.

    **Authors:**
    - Arpash Singh
    - Dhruv Rathi
    - Hardik Kapil

    **Guide:** Mr. Hemant Kumar Baranval

    *Dept. of Data Science, MIET*
    """)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Market Data",
    "🧠 Predictions",
    "📈 Model Performance",
    "📉 Training History",
    "🏗️ Architecture"
])

with tab1:
    st.markdown("### 📊 Live Market Data")

    col1, col2 = st.columns([3, 1])
    with col2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

    try:
        df = fetch_live_data(ticker, period)
        if df.empty:
            st.error(f"No data available for {ticker}")
        else:
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
            price_change = float(latest['Close']) - float(prev['Close'])
            pct_change = (price_change / float(prev['Close'])) * 100

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"₹{float(latest['Close']):.2f}",
                          f"{price_change:+.2f} ({pct_change:+.2f}%)")
            with col2:
                st.metric("Day High", f"₹{float(latest['High']):.2f}")
            with col3:
                st.metric("Day Low", f"₹{float(latest['Low']):.2f}")
            with col4:
                st.metric("Volume", f"{int(latest['Volume']):,}")

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.7, 0.3], vertical_spacing=0.05)

            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='OHLC', increasing_line_color='#00C853',
                decreasing_line_color='#FF1744'
            ), row=1, col=1)

            if len(df) > 20:
                ma20 = df['Close'].rolling(20).mean()
                fig.add_trace(go.Scatter(x=df.index, y=ma20, name='SMA 20',
                                         line=dict(color='#FF9800', width=1.5)), row=1, col=1)
            if len(df) > 50:
                ma50 = df['Close'].rolling(50).mean()
                fig.add_trace(go.Scatter(x=df.index, y=ma50, name='SMA 50',
                                         line=dict(color='#2196F3', width=1.5)), row=1, col=1)

            colors = ['#00C853' if c >= o else '#FF1744'
                      for c, o in zip(df['Close'], df['Open'])]
            fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume',
                                  marker_color=colors, opacity=0.6), row=2, col=1)

            fig.update_layout(
                title=f'{ticker} — Price Chart',
                template='plotly_dark',
                height=600,
                xaxis_rangeslider_visible=False,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=60, b=30)
            )
            fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Raw Data"):
                st.dataframe(df.tail(20).style.format("{:.2f}"), use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching data: {e}")

with tab2:
    st.markdown("### 🧠 LSTM Predictions — Actual vs Predicted")

    predictions = load_predictions()

    if predictions is not None:
        fig = go.Figure()

        if 'Date' in predictions.columns:
            x_axis = pd.to_datetime(predictions['Date'])
        else:
            x_axis = list(range(len(predictions)))

        fig.add_trace(go.Scatter(
            x=x_axis, y=predictions['Actual_Close'],
            name='Actual Close Price', mode='lines',
            line=dict(color='#2196F3', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=x_axis, y=predictions['Predicted_Close'],
            name='Predicted Close Price', mode='lines',
            line=dict(color='#FF5722', width=2, dash='dash')
        ))

        fig.update_layout(
            title='Stock Close Price — Actual vs LSTM Predicted',
            template='plotly_dark',
            height=500,
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=60, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Error Distribution")
            fig_err = px.histogram(predictions, x='Error', nbins=50,
                                    title='Prediction Error Distribution',
                                    template='plotly_dark',
                                    color_discrete_sequence=['#9C27B0'])
            fig_err.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_err, use_container_width=True)

        with col2:
            st.markdown("#### Scatter: Predicted vs Actual")
            fig_scatter = px.scatter(predictions, x='Actual_Close', y='Predicted_Close',
                                      title='Predicted vs Actual Close Price',
                                      template='plotly_dark',
                                      color_discrete_sequence=['#4CAF50'])
            min_val = min(predictions['Actual_Close'].min(), predictions['Predicted_Close'].min())
            max_val = max(predictions['Actual_Close'].max(), predictions['Predicted_Close'].max())
            fig_scatter.add_shape(type='line', x0=min_val, y0=min_val,
                                   x1=max_val, y1=max_val,
                                   line=dict(color='red', dash='dash'))
            st.plotly_chart(fig_scatter, use_container_width=True)

        with st.expander("📋 Prediction Data"):
            st.dataframe(predictions.style.format({
                'Actual_Close': '₹{:.2f}',
                'Predicted_Close': '₹{:.2f}',
                'Error': '₹{:.2f}',
                'Abs_Error': '₹{:.2f}'
            }), use_container_width=True)
    else:
        st.info("⚠️ No predictions available. Run `python train.py` and `python evaluate.py` first.")

with tab3:
    st.markdown("### 📈 Model Performance — Evaluation Metrics")

    metrics = load_metrics()

    if metrics is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>RMSE</h3>
                <p>₹{metrics['test_rmse']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>MAE</h3>
                <p>₹{metrics['test_mae']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>R² Score</h3>
                <p>{metrics['test_r2']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Training Set Metrics")
            fig_train = go.Figure(data=[go.Bar(
                x=['RMSE', 'MAE'],
                y=[metrics['train_rmse'], metrics['train_mae']],
                marker_color=['#2196F3', '#2196F3'],
                text=[f"₹{metrics['train_rmse']:.2f}", f"₹{metrics['train_mae']:.2f}"],
                textposition='auto'
            )])
            fig_train.update_layout(template='plotly_dark', height=350, title="Training Error Metrics")
            st.plotly_chart(fig_train, use_container_width=True)

        with col2:
            st.markdown("#### Test Set Metrics")
            fig_test = go.Figure(data=[go.Bar(
                x=['RMSE', 'MAE'],
                y=[metrics['test_rmse'], metrics['test_mae']],
                marker_color=['#FF5722', '#FF5722'],
                text=[f"₹{metrics['test_rmse']:.2f}", f"₹{metrics['test_mae']:.2f}"],
                textposition='auto'
            )])
            fig_test.update_layout(template='plotly_dark', height=350, title="Test Error Metrics")
            st.plotly_chart(fig_test, use_container_width=True)

        st.markdown("#### R² Score (Coefficient of Determination)")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=metrics['test_r2'],
            title={'text': "R² Score (Test Set)"},
            delta={'reference': metrics['train_r2'], 'relative': False},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "#4CAF50"},
                'steps': [
                    {'range': [0, 0.5], 'color': '#ffebee'},
                    {'range': [0.5, 0.8], 'color': '#fff3e0'},
                    {'range': [0.8, 1.0], 'color': '#e8f5e9'},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.95
                }
            }
        ))
        fig_gauge.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)

        with st.expander("📋 Detailed Metrics"):
            metrics_df = pd.DataFrame({
                'Metric': ['RMSE', 'MAE', 'R²', 'RMSE', 'MAE', 'R²'],
                'Set': ['Test', 'Test', 'Test', 'Train', 'Train', 'Train'],
                'Value': [
                    metrics['test_rmse'], metrics['test_mae'], metrics['test_r2'],
                    metrics['train_rmse'], metrics['train_mae'], metrics['train_r2']
                ]
            })
            st.dataframe(metrics_df.style.format({'Value': '{:.4f}'}), use_container_width=True)
    else:
        st.info("⚠️ No metrics available. Run `python train.py` and `python evaluate.py` first.")

with tab4:
    st.markdown("### 📉 Training History — Loss Curves")

    history = load_training_history()

    if history is not None:
        epochs = list(range(1, len(history['loss']) + 1))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs, y=history['loss'],
            name='Training Loss', mode='lines+markers',
            line=dict(color='#2196F3', width=2),
            marker=dict(size=4)
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=history['val_loss'],
            name='Validation Loss', mode='lines+markers',
            line=dict(color='#FF5722', width=2),
            marker=dict(size=4)
        ))

        best_epoch = int(np.argmin(history['val_loss'])) + 1
        best_loss = min(history['val_loss'])

        fig.add_annotation(
            x=best_epoch, y=best_loss,
            text=f"Best: {best_loss:.6f}<br>Epoch {best_epoch}",
            showarrow=True, arrowhead=2,
            font=dict(size=12, color='red'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='red'
        )

        fig.update_layout(
            title='Training & Validation Loss (MSE)',
            xaxis_title='Epoch',
            yaxis_title='MSE',
            template='plotly_dark',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Epochs", len(history['loss']))
        with col2:
            st.metric("Best Epoch", best_epoch)
        with col3:
            st.metric("Best Val Loss", f"{best_loss:.6f}")

        with st.expander("📋 Epoch-by-Epoch Loss"):
            loss_df = pd.DataFrame({
                'Epoch': epochs,
                'Training Loss': history['loss'],
                'Validation Loss': history['val_loss']
            })
            st.dataframe(loss_df.style.format({
                'Training Loss': '{:.6f}',
                'Validation Loss': '{:.6f}'
            }), use_container_width=True)
    else:
        st.info("⚠️ No training history. Run `python train.py` first.")

with tab5:
    st.markdown("### 🏗️ LSTM Model Architecture")

    st.markdown("""
    <div class="info-box">
    The model uses a stacked LSTM architecture with two layers (50 units each) and
    dropout (0.2) to prevent overfitting. The LSTM gates model long-term dependencies
    in financial time series data.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Model Architecture Diagram")
        st.markdown("""
        ```
        ┌─────────────────────────────────────────────┐
        │              INPUT LAYER                     │
        │         (60 timesteps × 1 feature)           │
        └──────────────────┬──────────────────────────┘
                           │
        ┌──────────────────▼──────────────────────────┐
        │           LSTM Layer 1                       │
        │         (50 units, return_sequences=True)    │
        └──────────────────┬──────────────────────────┘
                           │
        ┌──────────────────▼──────────────────────────┐
        │           Dropout (0.2)                      │
        └──────────────────┬──────────────────────────┘
                           │
        ┌──────────────────▼──────────────────────────┐
        │           LSTM Layer 2                       │
        │         (50 units, return_sequences=False)   │
        └──────────────────┬──────────────────────────┘
                           │
        ┌──────────────────▼──────────────────────────┐
        │           Dropout (0.2)                      │
        └──────────────────┬──────────────────────────┘
                           │
        ┌──────────────────▼──────────────────────────┐
        │           Dense Layer (1 neuron)              │
        └──────────────────┬──────────────────────────┘
                           │
                    Predicted Price
        ```
        """)

    with col2:
        st.markdown("#### Hyperparameters")
        params_df = pd.DataFrame({
            'Parameter': [
                'LSTM Layers', 'Units/Layer', 'Dropout Rate',
                'Optimizer', 'Loss Function', 'Learning Rate',
                'Sequence Length', 'Batch Size', 'Epochs',
                'Train/Test Split'
            ],
            'Value': [
                '2 (Stacked)', '50', '0.2',
                'Adam', 'MSE', '0.001',
                '60 days', '32', '50',
                '80/20'
            ]
        })
        st.dataframe(params_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🔄 Pipeline")
    st.markdown("""
    ```
    ┌──────────────────┐
    │  Data Collection │  ← NSE Historical Stock Data (Yahoo Finance)
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │  Preprocessing   │  ← Missing Values, MinMaxScaler
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │    Sequences     │  ← 60-day sliding windows
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │  Model Training  │  ← Stacked LSTM, Adam, MSE, 80/20 Split
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │    Prediction    │  ← Close Price Forecasting
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │   Evaluation     │  ← RMSE, MAE, R²
    └────────┬─────────┘
             │
    ┌────────▼─────────┐
    │   This Dashboard │
    └──────────────────┘
    ```
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 13px;">
    📊 Stock Market Prediction Using LSTM Networks — Final Year Project<br>
    Meerut Institute of Engineering and Technology (MIET) — Dept. of Data Science<br>
    <em>This is an educational project. Not financial advice.</em>
</div>
""", unsafe_allow_html=True)