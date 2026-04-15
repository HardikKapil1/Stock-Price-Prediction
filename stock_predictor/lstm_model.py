"""
LSTM Model Architecture for Stock Market Prediction.

Paper Section III.C - LSTM Model Architecture:
    "The model will consist of stacked LSTM with two layers (50 units each)
    and dropout (0.2). The LSTM gates (input, forget, output) allow accurate
    prediction of the financial time series by modeling nonlinear patterns
    and long-term dependencies."

Architecture:
    Input -> LSTM(50, return_sequences=True) -> Dropout(0.2)
          -> LSTM(50, return_sequences=False) -> Dropout(0.2)
          -> Dense(1)

    Optimizer: Adam (learning_rate configurable, default 0.001)
    Loss: Mean Squared Error (MSE)
"""

from typing import Tuple, Any, Optional
from stock_predictor.logging_utils import get_logger

logger = get_logger()


def build_lstm(input_shape: Tuple[int, int],
               lstm_units: int = 50,
               dropout_rate: float = 0.2,
               num_layers: int = 2,
               learning_rate: float = 0.001) -> Any:
    """Build and compile a stacked LSTM model as specified in the paper.

    The architecture uses:
    - Stacked LSTM layers (2 layers by default) for capturing complex
      temporal patterns in financial time series
    - Dropout regularization to prevent overfitting
    - Dense output layer with single neuron for price prediction
    - Adam optimizer with MSE loss

    Args:
        input_shape: (timesteps, features) - e.g., (60, 1).
        lstm_units: Number of LSTM units per layer (paper: 50).
        dropout_rate: Dropout rate (paper: 0.2).
        num_layers: Number of stacked LSTM layers (paper: 2).
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled Keras Sequential model, or None if TensorFlow unavailable.
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        logger.error("TensorFlow not installed. Please install: pip install tensorflow")
        return None

    model = Sequential()

    # First LSTM layer - returns sequences for stacking
    model.add(LSTM(
        units=lstm_units,
        return_sequences=(num_layers > 1),  # True if more layers follow
        input_shape=input_shape
    ))
    model.add(Dropout(dropout_rate))

    # Additional LSTM layers (stacked architecture)
    for i in range(1, num_layers):
        return_seq = (i < num_layers - 1)  # Last LSTM layer returns single output
        model.add(LSTM(units=lstm_units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))

    # Output layer - single neuron for price prediction (regression)
    model.add(Dense(1))

    # Compile with Adam optimizer and MSE loss (Paper Section III.D)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error'
    )

    logger.info("=" * 60)
    logger.info("LSTM Model Architecture (Paper Section III.C)")
    logger.info("=" * 60)
    model.summary(print_fn=logger.info)
    logger.info(f"Optimizer: Adam (lr={learning_rate})")
    logger.info(f"Loss: Mean Squared Error (MSE)")
    logger.info(f"LSTM Layers: {num_layers} x {lstm_units} units")
    logger.info(f"Dropout: {dropout_rate}")
    logger.info("=" * 60)

    return model


def save_model(model: Any, path: str) -> None:
    """Save a trained Keras model to disk.

    Args:
        model: Trained Keras model.
        path: File path to save (e.g., 'models/lstm_model.keras').
    """
    try:
        from pathlib import Path as P
        P(path).parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        logger.info(f"Model saved to: {path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")


def load_model(path: str) -> Optional[Any]:
    """Load a trained Keras model from disk.

    Args:
        path: File path of the saved model.

    Returns:
        Loaded Keras model, or None if loading fails.
    """
    try:
        from tensorflow.keras.models import load_model as _load
        model = _load(path)
        logger.info(f"Model loaded from: {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None
