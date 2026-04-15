"""LSTM preprocessing helpers: scaling and sequence creation."""
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


def create_sequences(values: np.ndarray, seq_len: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Create rolling sequences for LSTM.

    Args:
        values: 1D numpy array of target values (e.g., closing prices).
        seq_len: length of input sequence.

    Returns:
        X: (n_samples, seq_len, 1)
        y: (n_samples,)
    """
    X, y = [], []
    for i in range(seq_len, len(values)):
        X.append(values[i - seq_len:i])
        y.append(values[i])
    X = np.array(X)
    y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y


def scale_series(series: pd.Series, scaler_path: str = None) -> Tuple[np.ndarray, MinMaxScaler]:
    """Scale a pandas Series to [0,1] and optionally save the scaler.

    Returns scaled values (1D numpy) and scaler object.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = series.values.reshape(-1, 1)
    scaled = scaler.fit_transform(values).flatten()
    if scaler_path:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
    return scaled, scaler


def load_scaler(scaler_path: str) -> MinMaxScaler:
    return joblib.load(scaler_path)
