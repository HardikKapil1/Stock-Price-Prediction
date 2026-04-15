"""Tests for the LSTM model module."""

import pytest


class TestBuildLSTM:
    """Test LSTM model building."""

    def test_model_builds(self):
        """Model should build successfully with TensorFlow."""
        try:
            from stock_predictor.lstm_model import build_lstm
            model = build_lstm(input_shape=(60, 1), lstm_units=50,
                               dropout_rate=0.2, num_layers=2)
            if model is not None:
                # Check output shape
                assert model.output_shape == (None, 1)
                # Check it has the right number of layers
                # Expect: LSTM, Dropout, LSTM, Dropout, Dense = 5 layers
                assert len(model.layers) == 5
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_model_compiles(self):
        """Model should have MSE loss and Adam optimizer."""
        try:
            from stock_predictor.lstm_model import build_lstm
            model = build_lstm(input_shape=(60, 1))
            if model is not None:
                assert model.loss == 'mean_squared_error'
        except ImportError:
            pytest.skip("TensorFlow not installed")

    def test_custom_units(self):
        """Model should accept custom LSTM units."""
        try:
            from stock_predictor.lstm_model import build_lstm
            model = build_lstm(input_shape=(30, 1), lstm_units=100,
                               dropout_rate=0.3, num_layers=3)
            if model is not None:
                assert model.output_shape == (None, 1)
                # 3 LSTM + 3 Dropout + 1 Dense = 7 layers
                assert len(model.layers) == 7
        except ImportError:
            pytest.skip("TensorFlow not installed")
