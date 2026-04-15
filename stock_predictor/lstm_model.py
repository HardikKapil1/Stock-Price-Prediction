"""LSTM model builder and save/load helpers."""
from typing import Tuple, Any


def build_lstm(input_shape: Tuple[int, int]) -> Any:
    """Build and compile a simple LSTM model.

    Args:
        input_shape: (timesteps, features)
    Returns:
        Compiled Keras model (if TensorFlow installed).
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
    except Exception:
        return None

    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return model


def save_model(model: Any, path: str):
    try:
        model.save(path)
    except Exception:
        pass


def load_model(path: str) -> Any:
    try:
        from tensorflow.keras.models import load_model as _load
        return _load(path)
    except Exception:
        return None
