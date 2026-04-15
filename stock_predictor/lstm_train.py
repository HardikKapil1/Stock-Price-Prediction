"""Training script for LSTM model.

Usage:
    python -m stock_predictor.lstm_train --data data/raw_tsla_20180101_20241231.csv
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from .lstm_preprocessing import create_sequences, scale_series
from .lstm_model import build_lstm, save_model


def train(data_path: str, seq_len: int, epochs: int, batch_size: int, model_out: str, scaler_out: str):
    df = pd.read_csv(Path(data_path))
    if "close" not in df.columns:
        raise ValueError("Data must contain a 'close' column")

    series = df["close"].astype(float)
    scaled, scaler = scale_series(series, scaler_out)
    X, y = create_sequences(scaled, seq_len=seq_len)

    # simple train/val split
    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_lstm((X_train.shape[1], X_train.shape[2]))
    if model is None:
        print("TensorFlow not available — skipping training")
        return

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    save_model(model, model_out)
    print(f"Saved model to {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_out", default="models/lstm_model.keras")
    parser.add_argument("--scaler_out", default="models/scaler.pkl")
    args = parser.parse_args()

    train(args.data, args.seq_len, args.epochs, args.batch_size, args.model_out, args.scaler_out)
