import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from stock_predictor.data_loader import download_stock_data, load_from_csv
from stock_predictor.preprocessing import (
    handle_missing_values,
    normalize_data,
    create_sequences,
    split_data
)
from stock_predictor.lstm_model import build_lstm, save_model
from stock_predictor.logging_utils import get_logger

logger = get_logger()


def train_lstm(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full training pipeline."""
    print("Starting LSTM training...")

    ticker = config['Ticker']
    start = str(config['StartDate'])
    end = str(config['EndDate'])

    # check for cached data
    ticker_clean = ticker.replace('.', '_').lower()
    cache_path = f"data/raw/{ticker_clean}_{start.replace('-','')}_{end.replace('-','')}.csv"

    if Path(cache_path).exists():
        logger.info(f"Using cached data: {cache_path}")
        df = load_from_csv(cache_path)
    else:
        df = download_stock_data(ticker, start, end)

    print(f"Data: {ticker} | {df.index[0].date()} to {df.index[-1].date()} | {len(df)} rows")

    # preprocessing
    df = handle_missing_values(df)
    close_prices = df['Close'].values.astype(float)
    print(f"Close price range: {close_prices.min():.2f} - {close_prices.max():.2f}")

    scaled_data, scaler = normalize_data(close_prices, scaler_path="models/scaler.pkl")

    # create sequences
    seq_len = config.get('SequenceLength', 60)
    X, y = create_sequences(scaled_data, seq_len=seq_len)
    print(f"Sequences: {X.shape[0]} samples | lookback: {seq_len} days")

    # train/test split
    test_size = float(config.get('TestSize', 0.20))
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # build model
    lstm_units = config.get('LSTMUnits', 50)
    dropout_rate = float(config.get('Dropout', 0.2))
    num_layers = config.get('NumLayers', 2)
    learning_rate = float(config.get('LearningRate', 0.001))

    model = build_lstm(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        num_layers=num_layers,
        learning_rate=learning_rate
    )

    if model is None:
        raise RuntimeError("Failed to build LSTM model. Is TensorFlow installed?")

    # train
    epochs = config.get('Epochs', 50)
    batch_size = config.get('BatchSize', 32)
    print(f"Training for {epochs} epochs, batch size {batch_size}...")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # save everything
    model_path = "models/lstm_model.keras"
    save_model(model, model_path)

    Path("outputs").mkdir(parents=True, exist_ok=True)
    history_dict = {
        'loss': [float(v) for v in history.history['loss']],
        'val_loss': [float(v) for v in history.history['val_loss']],
    }
    with open("outputs/training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    np.savez(
        "data/processed/test_data.npz",
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train
    )

    test_dates = df.index[seq_len + len(X_train):][:len(X_test)]
    pd.Series(test_dates).to_csv("data/processed/test_dates.csv", index=False)

    with open("outputs/training_config.json", "w") as f:
        json.dump({k: str(v) for k, v in config.items()}, f, indent=2)

    # summary
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    best_val_loss = min(history.history['val_loss'])
    best_epoch = history.history['val_loss'].index(best_val_loss) + 1

    print(f"\nTraining done!")
    print(f"  Train Loss: {final_train_loss:.6f}")
    print(f"  Val Loss:   {final_val_loss:.6f}")
    print(f"  Best Val:   {best_val_loss:.6f} (epoch {best_epoch})")

    return {
        'model_path': model_path,
        'scaler_path': 'models/scaler.pkl',
        'history': history_dict,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
    }
