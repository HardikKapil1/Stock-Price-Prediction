"""
LSTM Training Module for Stock Market Prediction.

Paper Section III.D - Model Training:
    "The Adam optimizer is applied with MSE loss and the train-test split
    is 80/20 to train the model. Multi-epoch sequential training and
    validation tracking enhance accuracy, reduce overfitting, and guarantee
    predictive accuracy on unknown stock data."

This module orchestrates the full training pipeline:
    1. Load and clean data
    2. Normalize with MinMaxScaler
    3. Create sequences
    4. Split into train/test
    5. Build and train stacked LSTM
    6. Save model, scaler, and training history
"""

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
    """Execute the full LSTM training pipeline.

    Follows the paper's methodology:
        Data Collection -> Data Preprocessing -> Feature Extraction
        -> Model Training -> Save Artifacts

    Args:
        config: Dictionary with configuration parameters:
            - Ticker, StartDate, EndDate
            - SequenceLength, LSTMUnits, Dropout, NumLayers
            - Epochs, BatchSize, LearningRate
            - TestSize

    Returns:
        Dictionary with training results and file paths.
    """
    print("=" * 80)
    print("STOCK MARKET PREDICTION USING LSTM NETWORKS")
    print("Training Pipeline - Final Year Project")
    print("=" * 80)

    # -- Step 1: Data Collection (Paper Section III.A) --
    logger.info("STEP 1: Data Collection")
    ticker = config['Ticker']
    start = str(config['StartDate'])
    end = str(config['EndDate'])

    # Check for cached data first
    ticker_clean = ticker.replace('.', '_').lower()
    start_compact = start.replace('-', '')
    end_compact = end.replace('-', '')
    cache_path = f"data/raw/{ticker_clean}_{start_compact}_{end_compact}.csv"

    if Path(cache_path).exists():
        logger.info(f"Loading cached data from {cache_path}")
        df = load_from_csv(cache_path)
    else:
        df = download_stock_data(ticker, start, end)

    print(f"\n[DATA] {ticker} | {df.index[0].date()} to {df.index[-1].date()} | {len(df)} rows")

    # -- Step 2: Data Preprocessing (Paper Section III.B) --
    logger.info("STEP 2: Data Preprocessing")
    df = handle_missing_values(df)

    # Extract closing prices (the primary feature for LSTM prediction)
    close_prices = df['Close'].values.astype(float)
    print(f"[PRICE] Close price range: Rs.{close_prices.min():.2f} - Rs.{close_prices.max():.2f}")

    # Normalize to [0, 1] using MinMaxScaler
    scaled_data, scaler = normalize_data(
        close_prices,
        scaler_path="models/scaler.pkl"
    )
    print(f"[OK] MinMaxScaler applied | Range: [{scaled_data.min():.4f}, {scaled_data.max():.4f}]")

    # -- Step 3: Feature Extraction / Sequence Creation --
    logger.info("STEP 3: Creating time sequences")
    seq_len = config.get('SequenceLength', 60)
    X, y = create_sequences(scaled_data, seq_len=seq_len)
    print(f"[SEQ] Sequences: {X.shape[0]} samples | Lookback: {seq_len} days | Features: {X.shape[2]}")

    # -- Step 4: Train/Test Split (80/20 chronological) --
    test_size = float(config.get('TestSize', 0.20))
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
    print(f"[SPLIT] Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # -- Step 5: Model Building (Paper Section III.C) --
    logger.info("STEP 4: Building LSTM model")
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

    print(f"\n[MODEL] Stacked LSTM ({num_layers} layers x {lstm_units} units)")
    print(f"   Dropout: {dropout_rate} | Optimizer: Adam (lr={learning_rate})")
    print(f"   Loss: Mean Squared Error (MSE)")

    # -- Step 6: Model Training (Paper Section III.D) --
    logger.info("STEP 5: Training model")
    epochs = config.get('Epochs', 50)
    batch_size = config.get('BatchSize', 32)

    print(f"\n[TRAIN] Training for {epochs} epochs with batch_size={batch_size}...")
    print("-" * 80)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # -- Step 7: Save Artifacts --
    logger.info("STEP 6: Saving model and artifacts")

    # Save model
    model_path = "models/lstm_model.keras"
    save_model(model, model_path)

    # Save training history
    Path("outputs").mkdir(parents=True, exist_ok=True)
    history_dict = {
        'loss': [float(v) for v in history.history['loss']],
        'val_loss': [float(v) for v in history.history['val_loss']],
    }
    with open("outputs/training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    # Save test data for evaluation
    np.savez(
        "data/processed/test_data.npz",
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train
    )

    # Save dates for plotting
    test_dates = df.index[seq_len + len(X_train):][:len(X_test)]
    pd.Series(test_dates).to_csv("data/processed/test_dates.csv", index=False)

    # Save config used for this training
    with open("outputs/training_config.json", "w") as f:
        serializable_config = {k: str(v) for k, v in config.items()}
        json.dump(serializable_config, f, indent=2)

    # -- Training Summary --
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    best_val_loss = min(history.history['val_loss'])
    best_epoch = history.history['val_loss'].index(best_val_loss) + 1

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"  Final Training Loss (MSE):   {final_train_loss:.6f}")
    print(f"  Final Validation Loss (MSE): {final_val_loss:.6f}")
    print(f"  Best Validation Loss:        {best_val_loss:.6f} (Epoch {best_epoch})")
    print(f"\n  Saved: {model_path}")
    print(f"  Saved: models/scaler.pkl")
    print(f"  Saved: outputs/training_history.json")
    print(f"  Saved: data/processed/test_data.npz")
    print("=" * 80)

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
