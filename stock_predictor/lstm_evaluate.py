"""Evaluate trained LSTM model and save predictions/metrics."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

from .lstm_preprocessing import create_sequences, load_scaler
from .lstm_model import load_model


def evaluate(data_path: str, model_path: str, scaler_path: str, seq_len: int, out_predictions: str, out_metrics: str):
    df = pd.read_csv(Path(data_path))
    if "close" not in df.columns:
        raise ValueError("Data must contain a 'close' column")

    series = df["close"].astype(float)
    scaler = load_scaler(scaler_path)
    values = scaler.transform(series.values.reshape(-1, 1)).flatten()

    X, y = create_sequences(values, seq_len=seq_len)
    model = load_model(model_path)
    if model is None:
        print("Model could not be loaded (TensorFlow missing?). Exiting.")
        return

    preds_scaled = model.predict(X).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y.reshape(-1, 1)).flatten()

    mse = mean_squared_error(y_true, preds)
    mae = mean_absolute_error(y_true, preds)

    Path(out_predictions).parent.mkdir(parents=True, exist_ok=True)
    preds_df = pd.DataFrame({"predicted_close": preds})
    preds_df.to_csv(out_predictions, index=False)

    metrics = {"mse": float(mse), "mae": float(mae)}
    Path(out_metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"Saved predictions to {out_predictions} and metrics to {out_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="models/lstm_model.keras")
    parser.add_argument("--scaler", default="models/scaler.pkl")
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--out_predictions", default="outputs/predictions.csv")
    parser.add_argument("--out_metrics", default="outputs/metrics.json")
    args = parser.parse_args()

    evaluate(args.data, args.model, args.scaler, args.seq_len, args.out_predictions, args.out_metrics)
