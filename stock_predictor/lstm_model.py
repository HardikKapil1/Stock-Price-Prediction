from typing import Tuple, Any, Optional
from stock_predictor.logging_utils import get_logger

logger = get_logger()


def build_lstm(input_shape: Tuple[int, int],
               lstm_units: int = 50,
               dropout_rate: float = 0.2,
               num_layers: int = 2,
               learning_rate: float = 0.001) -> Any:
    """Build and compile the stacked LSTM model."""
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        logger.error("TensorFlow not installed.")
        return None

    model = Sequential()

    # first LSTM layer
    model.add(LSTM(
        units=lstm_units,
        return_sequences=(num_layers > 1),
        input_shape=input_shape
    ))
    model.add(Dropout(dropout_rate))

    # remaining stacked layers
    for i in range(1, num_layers):
        return_seq = (i < num_layers - 1)
        model.add(LSTM(units=lstm_units, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error'
    )

    model.summary(print_fn=logger.info)
    logger.info(f"Layers: {num_layers} x {lstm_units} | Dropout: {dropout_rate} | LR: {learning_rate}")

    return model


def save_model(model: Any, path: str) -> None:
    """Save model to disk."""
    try:
        from pathlib import Path as P
        P(path).parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        logger.info(f"Model saved: {path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")


def load_model(path: str) -> Optional[Any]:
    """Load model from disk."""
    try:
        from tensorflow.keras.models import load_model as _load
        model = _load(path)
        logger.info(f"Model loaded: {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None
