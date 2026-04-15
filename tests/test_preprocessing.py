"""Tests for the preprocessing module."""

import numpy as np
import pandas as pd
import pytest
from stock_predictor.preprocessing import (
    handle_missing_values,
    normalize_data,
    create_sequences,
    split_data
)


class TestHandleMissingValues:
    """Test missing value handling."""

    def test_forward_fill(self):
        """Missing values should be forward-filled."""
        df = pd.DataFrame({
            'Close': [100, np.nan, 102, np.nan, 104],
            'Volume': [1000, 1100, np.nan, 1300, 1400]
        })
        result = handle_missing_values(df)
        assert result.isnull().sum().sum() == 0
        assert len(result) == 5

    def test_no_missing(self):
        """DataFrame with no missing values should pass through."""
        df = pd.DataFrame({'Close': [100, 101, 102]})
        result = handle_missing_values(df)
        assert len(result) == 3


class TestNormalizeData:
    """Test MinMaxScaler normalization."""

    def test_range_01(self):
        """Normalized values should be in [0, 1]."""
        data = np.array([100, 200, 150, 300, 250])
        scaled, scaler = normalize_data(data)
        assert scaled.min() >= 0.0
        assert scaled.max() <= 1.0

    def test_inverse_transform(self):
        """Inverse transform should recover original values."""
        data = np.array([100.0, 200.0, 150.0])
        scaled, scaler = normalize_data(data)
        recovered = scaler.inverse_transform(scaled.reshape(-1, 1)).flatten()
        np.testing.assert_array_almost_equal(recovered, data, decimal=5)


class TestCreateSequences:
    """Test sliding window sequence creation."""

    def test_sequence_shape(self):
        """Output shapes should match expected dimensions."""
        data = np.arange(100, dtype=float)
        X, y = create_sequences(data, seq_len=10)
        assert X.shape == (90, 10, 1)
        assert y.shape == (90,)

    def test_sequence_values(self):
        """First sequence should be first seq_len values."""
        data = np.arange(20, dtype=float)
        X, y = create_sequences(data, seq_len=5)
        expected_X = np.array([0, 1, 2, 3, 4]).reshape(5, 1)
        np.testing.assert_array_equal(X[0], expected_X)
        assert y[0] == 5.0

    def test_last_sequence(self):
        """Last target should be the last value in data."""
        data = np.arange(20, dtype=float)
        X, y = create_sequences(data, seq_len=5)
        assert y[-1] == 19.0


class TestSplitData:
    """Test chronological train/test split."""

    def test_split_sizes(self):
        """Split should produce correct sizes."""
        X = np.random.randn(100, 10, 1)
        y = np.random.randn(100)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_no_shuffle(self):
        """Split should be chronological (no shuffling)."""
        X = np.arange(50).reshape(50, 1, 1)
        y = np.arange(50, dtype=float)
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        # Train should be first 40, test should be last 10
        np.testing.assert_array_equal(y_train, np.arange(40, dtype=float))
        np.testing.assert_array_equal(y_test, np.arange(40, 50, dtype=float))
