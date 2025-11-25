"""Feature engineering functions for stock direction prediction.

All feature transformations must remain consistent across training and
inference (Streamlit app). The engineer_features function returns a DataFrame
with the original columns plus engineered features and a 'target' column
(if include_target=True).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterable

__all__ = ["ENGINEERED_FEATURES", "engineer_features"]

ENGINEERED_FEATURES: Iterable[str] = [
    'returns', 'log_returns', 'gap', 'daily_range',
    'momentum_3', 'momentum_5', 'momentum_10', 'momentum_14',
    'volume_change_3', 'volume_change_5', 'volume_change_10',
    'rsi', 'rsi_oversold', 'rsi_overbought',
    'macd_diff', 'macd_crossover',
    'bb_position', 'bb_squeeze',
    'volatility_10', 'volatility_30',
    'volume_ratio', 'volume_spike',
    'price_to_sma5', 'price_to_sma20', 'sma_trend',
    'near_high', 'near_low',
    'consecutive_up', 'consecutive_down'
]

def engineer_features(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Create engineered features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data with columns: Open, High, Low, Close, Volume.
    include_target : bool, default True
        Whether to include the next-day direction target column.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features and optional 'target'.
    """
    data = df.copy()

    # Price-based
    data['returns'] = data['Close'].pct_change()
    data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))

    # Momentum & volume change
    for period in [3, 5, 10, 14]:
        data[f'momentum_{period}'] = data['Close'].pct_change(period)
        data[f'volume_change_{period}'] = data['Volume'].pct_change(period)

    # Gap
    data['gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
    data['gap_up'] = (data['gap'] > 0.01).astype(int)

    # Intraday range & shadows
    data['daily_range'] = (data['High'] - data['Low']) / data['Close']
    data['upper_shadow'] = (data['High'] - data[['Open', 'Close']].max(axis=1)) / data['Close']
    data['lower_shadow'] = (data[['Open', 'Close']].min(axis=1) - data['Low']) / data['Close']

    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    data['rsi'] = 100 - (100 / (1 + rs))
    data['rsi_oversold'] = (data['rsi'] < 30).astype(int)
    data['rsi_overbought'] = (data['rsi'] > 70).astype(int)

    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['macd'] = exp1 - exp2
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_diff'] = data['macd'] - data['macd_signal']
    data['macd_crossover'] = ((data['macd'] > data['macd_signal']) &
                              (data['macd'].shift(1) <= data['macd_signal'].shift(1))).astype(int)

    # Bollinger Bands
    sma20 = data['Close'].rolling(20).mean()
    std20 = data['Close'].rolling(20).std()
    data['bb_upper'] = sma20 + (2 * std20)
    data['bb_lower'] = sma20 - (2 * std20)
    data['bb_position'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'] + 1e-10)
    data['bb_squeeze'] = (data['bb_upper'] - data['bb_lower']) / sma20

    # Volatility
    data['volatility_10'] = data['returns'].rolling(10).std()
    data['volatility_30'] = data['returns'].rolling(30).std()

    # Volume analysis
    data['volume_ma5'] = data['Volume'].rolling(5).mean()
    data['volume_ma20'] = data['Volume'].rolling(20).mean()
    data['volume_ratio'] = data['Volume'] / data['volume_ma20']
    data['volume_spike'] = (data['volume_ratio'] > 2).astype(int)

    # Moving averages
    data['sma5'] = data['Close'].rolling(5).mean()
    data['sma20'] = data['Close'].rolling(20).mean()
    data['sma50'] = data['Close'].rolling(50).mean()
    data['price_to_sma5'] = data['Close'] / data['sma5']
    data['price_to_sma20'] = data['Close'] / data['sma20']
    data['sma_trend'] = (data['sma5'] > data['sma20']).astype(int)

    # Highs / Lows proximity
    data['near_high'] = (data['Close'] >= data['High'].rolling(14).max() * 0.98).astype(int)
    data['near_low'] = (data['Close'] <= data['Low'].rolling(14).min() * 1.02).astype(int)

    # Consecutive patterns
    data['consecutive_up'] = (data['returns'] > 0).rolling(3).sum()
    data['consecutive_down'] = (data['returns'] < 0).rolling(3).sum()

    if include_target:
        data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    return data
