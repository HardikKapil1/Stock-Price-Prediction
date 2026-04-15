"""
Data Preprocessing Module for Stock Market Prediction.

Paper Section III.B - Data Preprocessing:
    "Missing values were processed, feature normalization (min-max scaling),
    and time sequences were processed. The information was systematized to
    enable effective processing and analysis."

This module provides:
    - Missing value handling (forward fill + drop)
    - MinMaxScaler normalization to [0, 1]
    - Sliding-window sequence creation for LSTM input
    - Chronological train/test splitting
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from stock_predictor.logging_utils import get_logger

logger = get_logger()


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset by handling missing values.

    Uses forward-fill followed by dropping any remaining NaN rows.
    This preserves time-series continuity while removing incomplete data.

    Args:
        df: Raw OHLCV DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame with no missing values.
    """
    missing_before = df.isnull().sum().sum()
    df = df.ffill()           # Forward fill for continuity
    df = df.dropna()          # Drop any remaining NaNs
    missing_after = df.isnull().sum().sum()

    logger.info(f"Missing values: {missing_before} -> {missing_after} after cleaning")
    logger.info(f"Clean data shape: {df.shape}")

    return df


def normalize_data(data: np.ndarray, scaler_path: str = None,
                   scaler: MinMaxScaler = None) -> Tuple[np.ndarray, MinMaxScaler]:
    """Normalize data to [0, 1] range using MinMaxScaler.

    Paper specifies MinMaxScaler for data normalization to bring all
    values into a uniform [0, 1] range for LSTM training.

    Args:
        data: 1D numpy array of values (e.g., closing prices).
        scaler_path: Optional path to save the fitted scaler.
        scaler: Optional pre-fitted scaler for transform-only mode.

    Returns:
        Tuple of (scaled_data, scaler).
    """
    values = data.reshape(-1, 1)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        logger.info(f"MinMaxScaler fitted | Min: {scaler.data_min_[0]:.2f}, Max: {scaler.data_max_[0]:.2f}")
    else:
        scaled = scaler.transform(values)

    if scaler_path:
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to: {scaler_path}")

    return scaled.flatten(), scaler


def create_sequences(data: np.ndarray, seq_len: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences for LSTM input.

    For each position i >= seq_len, creates:
        X[i]: data[i-seq_len : i]  (the past seq_len values)
        y[i]: data[i]              (the next value to predict)

    Args:
        data: 1D numpy array of normalized values.
        seq_len: Length of the input sequence (lookback window).

    Returns:
        X: shape (n_samples, seq_len, 1) - LSTM input sequences.
        y: shape (n_samples,) - target values.
    """
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])

    X = np.array(X)
    y = np.array(y)

    # Reshape X to (samples, timesteps, features) for LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))

    logger.info(f"Created {len(X)} sequences with lookback={seq_len}")
    logger.info(f"X shape: {X.shape} | y shape: {y.shape}")

    return X, y


def split_data(X: np.ndarray, y: np.ndarray,
               test_size: float = 0.20) -> Tuple[np.ndarray, np.ndarray,
                                                   np.ndarray, np.ndarray]:
    """Split data chronologically into train and test sets.

    Paper Section III.D specifies 80/20 train-test split.
    Chronological split is critical for time-series to prevent look-ahead bias.

    Args:
        X: Input sequences.
        y: Target values.
        test_size: Fraction of data to use for testing.

    Returns:
        X_train, X_test, y_train, y_test
    """
    split_idx = int(len(X) * (1 - test_size))

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    logger.info(f"Train/Test split at index {split_idx}")
    logger.info(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def load_scaler(scaler_path: str) -> MinMaxScaler:
    """Load a previously saved MinMaxScaler.

    Args:
        scaler_path: Path to the saved scaler file.

    Returns:
        MinMaxScaler: The loaded scaler.
    """
    scaler = joblib.load(scaler_path)
    logger.info(f"Loaded scaler from: {scaler_path}")
    return scaler
