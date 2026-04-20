import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from stock_predictor.logging_utils import get_logger

logger = get_logger()


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Forward fill gaps then drop any remaining NaNs."""
    missing_before = df.isnull().sum().sum()
    df = df.ffill()
    df = df.dropna()
    logger.info(f"Missing values: {missing_before} -> {df.isnull().sum().sum()} after cleaning")
    return df


def normalize_data(data: np.ndarray, scaler_path: str = None,
                   scaler: MinMaxScaler = None) -> Tuple[np.ndarray, MinMaxScaler]:
    """Scale data to [0, 1] using MinMaxScaler."""
    values = data.reshape(-1, 1)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        logger.info(f"Scaler fitted | Min: {scaler.data_min_[0]:.2f}, Max: {scaler.data_max_[0]:.2f}")
    else:
        scaled = scaler.transform(values)

    if scaler_path:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved: {scaler_path}")

    return scaled.flatten(), scaler


def create_sequences(data: np.ndarray, seq_len: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding window sequences for LSTM input."""
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    logger.info(f"Sequences: {len(X)} | X: {X.shape} | y: {y.shape}")
    return X, y


def split_data(X: np.ndarray, y: np.ndarray,
               test_size: float = 0.20) -> Tuple[np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray]:
    """Chronological 80/20 train-test split."""
    split_idx = int(len(X) * (1 - test_size))

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def load_scaler(scaler_path: str) -> MinMaxScaler:
    """Load a saved MinMaxScaler."""
    scaler = joblib.load(scaler_path)
    logger.info(f"Scaler loaded: {scaler_path}")
    return scaler
