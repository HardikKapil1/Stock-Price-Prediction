"""
🚀 STOCK DIRECTION PREDICTION SYSTEM
Professional Streamlit Application (No external AI dependencies)
Final Year Project - Production Ready
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score
from stock_predictor.features import engineer_features, ENGINEERED_FEATURES

# Page config
st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
    }
    .up-prediction {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .down-prediction {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# Load models
@st.cache_resource
def load_models():
    """Load trained models and scalers"""
    try:
        base = 'models'
        with open(f'{base}/final_rf.pkl', 'rb') as f:
            rf = pickle.load(f)
        with open(f'{base}/final_gb.pkl', 'rb') as f:
            gb = pickle.load(f)
        with open(f'{base}/final_xgb.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        with open(f'{base}/final_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open(f'{base}/final_features.pkl', 'rb') as f:
            features = pickle.load(f)
        with open(f'{base}/final_weights.pkl', 'rb') as f:
            weights = pickle.load(f)
        return rf, gb, xgb_model, scaler, features, weights
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, None, None

# Feature engineering function
 # Feature engineering imported from shared module (engineer_features)

# (Removed external AI insight generation to simplify dependencies)

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 Stock Direction Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Machine Learning Ensemble (Technical Indicators Only)")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/stock-share.png", width=80)
        st.title("⚙️ Settings")
        
        ticker = st.text_input("Stock Ticker", value="TSLA", help="Enter stock symbol (e.g., TSLA, AAPL)")
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info("""
        **Ensemble Model**
        - Random Forest (500 trees)
        - Gradient Boosting (300 est.)
        - XGBoost (300 est.)
        
        **Features:** 29 technical indicators
        **Accuracy:** 51.9%
        **Training:** 2018-2023 data
        """)
        
        st.markdown("---")
        st.markdown("### 🎓 Final Year Project")
        st.success("Machine Learning for Stock Direction Prediction")
    
    # Load models
    rf, gb, xgb_model, scaler, features, weights = load_models()
    
    if rf is None:
        st.error("❌ Models not loaded. Please run train_final.py first!")
        return
    
    # Fetch data
    with st.spinner(f'📡 Fetching {ticker} data...'):
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=365)
            raw_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns = raw_data.columns.get_level_values(0)
            
            if len(raw_data) < 60:
                st.error(f"❌ Not enough data for {ticker}")
                return
                
        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
            return
    
    # Engineer features
    data = engineer_features(raw_data)
    data = data.dropna()
    
    if len(data) == 0:
        st.error("❌ No data after feature engineering")
        return
    
    # Get latest data point
    latest = data.iloc[-1]
    X_latest = latest[features].values.reshape(1, -1)
    X_scaled = scaler.transform(X_latest)
    
    # Make predictions
    rf_prob = rf.predict_proba(X_scaled)[0, 1]
    gb_prob = gb.predict_proba(X_scaled)[0, 1]
    xgb_prob = xgb_model.predict_proba(X_scaled)[0, 1]
    
    ensemble_prob = (rf_prob * weights[0] + gb_prob * weights[1] + xgb_prob * weights[2])
    ensemble_pred = 1 if ensemble_prob > 0.5 else 0
    
    current_price = latest['Close']
    
    # Display prediction
    st.markdown("## 🎯 Prediction for Next Trading Day")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if ensemble_pred == 1:
            st.markdown(f'''
            <div class="prediction-box up-prediction">
                📈 BULLISH SIGNAL<br>
                <span style="font-size:3rem;">↗ UP</span><br>
                Confidence: {ensemble_prob*100:.1f}%
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="prediction-box down-prediction">
                📉 BEARISH SIGNAL<br>
                <span style="font-size:3rem;">↘ DOWN</span><br>
                Confidence: {(1-ensemble_prob)*100:.1f}%
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        daily_change = ((latest['Close'] / data['Close'].iloc[-2]) - 1) * 100
        st.metric("Today's Change", f"{daily_change:+.2f}%", delta=f"{daily_change:+.2f}%")
    with col3:
        st.metric("RSI", f"{latest['rsi']:.1f}")
    with col4:
        st.metric("Volume Ratio", f"{latest['volume_ratio']:.2f}x")
    
    # AI Insights
    # Removed AI-powered insights section (no external API key required)
    
    # Charts
    st.markdown("## 📊 Technical Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Chart", "🎯 Model Votes", "📊 Feature Importance", "📉 Indicators"])
    
    with tab1:
        # Interactive price chart
        fig = go.Figure()
        
        last_30 = data.tail(30)
        fig.add_trace(go.Candlestick(
            x=last_30.index,
            open=last_30['Open'],
            high=last_30['High'],
            low=last_30['Low'],
            close=last_30['Close'],
            name='Price'
        ))
        
        fig.add_trace(go.Scatter(
            x=last_30.index, y=last_30['sma20'],
            name='SMA 20', line=dict(color='blue', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=last_30.index, y=last_30['bb_upper'],
            name='BB Upper', line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=last_30.index, y=last_30['bb_lower'],
            name='BB Lower', line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title=f'{ticker} - Last 30 Days',
            yaxis_title='Price ($)',
            xaxis_title='Date',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Model votes
        votes_data = pd.DataFrame({
            'Model': ['Random Forest', 'Gradient Boosting', 'XGBoost'],
            'UP Probability': [rf_prob*100, gb_prob*100, xgb_prob*100],
            'DOWN Probability': [(1-rf_prob)*100, (1-gb_prob)*100, (1-xgb_prob)*100],
            'Weight': weights * 100
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=votes_data['Model'],
            y=votes_data['UP Probability'],
            name='UP',
            marker_color='#38ef7d'
        ))
        fig.add_trace(go.Bar(
            x=votes_data['Model'],
            y=votes_data['DOWN Probability'],
            name='DOWN',
            marker_color='#f45c43'
        ))
        
        fig.update_layout(
            title='Individual Model Predictions',
            yaxis_title='Probability (%)',
            barmode='group',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(votes_data.style.format({'UP Probability': '{:.1f}%', 
                                               'DOWN Probability': '{:.1f}%',
                                               'Weight': '{:.1f}%'}),
                    use_container_width=True)
    
    with tab3:
        # Feature importance
        try:
            feat_imp = pd.read_csv('outputs/feature_importance.csv').head(15)
            fig = px.barh(feat_imp, x='Importance', y='Feature', 
                         title='Top 15 Most Important Features',
                         color='Importance',
                         color_continuous_scale='viridis')
            fig.update_layout(height=500, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.warning("Feature importance file not found")
    
    with tab4:
        # Technical indicators
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.tail(30).index, y=data.tail(30)['rsi'],
                                    name='RSI', line=dict(color='purple', width=2)))
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
            fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
            fig.update_layout(title='RSI (14)', height=300, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # MACD
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.tail(30).index, y=data.tail(30)['macd'],
                                    name='MACD', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=data.tail(30).index, y=data.tail(30)['macd_signal'],
                                    name='Signal', line=dict(color='red')))
            fig.update_layout(title='MACD', height=300, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p><strong>Disclaimer:</strong> This is an educational project. Not financial advice. 
        Always do your own research before making investment decisions.</p>
        <p>Final Year Project • Machine Learning • Stock Prediction • 2024</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
