import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from stock_predictor.lstm_model import load_model
from stock_predictor.preprocessing import load_scaler
from stock_predictor.logging_utils import get_logger

logger = get_logger()


def evaluate_lstm(model_path: str = "models/lstm_model.keras",
                  scaler_path: str = "models/scaler.pkl",
                  test_data_path: str = "data/processed/test_data.npz",
                  history_path: str = "outputs/training_history.json",
                  output_dir: str = "outputs") -> Dict[str, Any]:
    """Evaluate the trained LSTM model and generate plots."""
    print("Running evaluation...")

    model = load_model(model_path)
    if model is None:
        raise RuntimeError(f"Cannot load model from {model_path}")

    scaler = load_scaler(scaler_path)
    test_data = np.load(test_data_path)
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    X_train = test_data['X_train']
    y_train = test_data['y_train']

    with open(history_path, 'r') as f:
        history = json.load(f)

    dates_path = "data/processed/test_dates.csv"
    if Path(dates_path).exists():
        test_dates = pd.to_datetime(pd.read_csv(dates_path).iloc[:, 0])
    else:
        test_dates = None

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()
    y_train_pred_scaled = model.predict(X_train, verbose=0).flatten()

    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_train_pred = scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred))
    mae = mean_absolute_error(y_test_actual, y_test_pred)
    r2 = r2_score(y_test_actual, y_test_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred))
    train_mae = mean_absolute_error(y_train_actual, y_train_pred)
    train_r2 = r2_score(y_train_actual, y_train_pred)

    metrics = {
        'test_rmse': float(rmse),
        'test_mae': float(mae),
        'test_r2': float(r2),
        'train_rmse': float(train_rmse),
        'train_mae': float(train_mae),
        'train_r2': float(train_r2),
        'test_samples': int(len(y_test)),
        'train_samples': int(len(y_train)),
    }

    print(f"\nTest  — RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.6f}")
    print(f"Train — RMSE: {train_rmse:.2f} | MAE: {train_mae:.2f} | R2: {train_r2:.6f}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame({
        'Actual_Close': y_test_actual,
        'Predicted_Close': y_test_pred,
        'Error': y_test_actual - y_test_pred,
        'Abs_Error': np.abs(y_test_actual - y_test_pred),
    })
    if test_dates is not None and len(test_dates) >= len(pred_df):
        pred_df.insert(0, 'Date', test_dates[:len(pred_df)].values)
    pred_df.to_csv(f"{output_dir}/predictions.csv", index=False)

    plots_dir = f"{output_dir}/plots"
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    sns.set_style('whitegrid')
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'figure.dpi': 100})

    _plot_actual_vs_predicted(y_test_actual, y_test_pred, test_dates, plots_dir)
    _plot_training_loss(history, plots_dir)
    _plot_scatter(y_test_actual, y_test_pred, r2, plots_dir)
    _plot_error_distribution(y_test_actual, y_test_pred, plots_dir)
    _plot_residuals(y_test_actual, y_test_pred, test_dates, plots_dir)
    _plot_comprehensive(y_test_actual, y_test_pred, y_train_actual, y_train_pred,
                        history, test_dates, metrics, plots_dir)

    print(f"Plots saved to: {plots_dir}/")
    return metrics


def _plot_actual_vs_predicted(y_actual, y_pred, dates, output_dir):
    fig, ax = plt.subplots(figsize=(14, 6))
    x_axis = dates[:len(y_actual)] if dates is not None and len(dates) >= len(y_actual) else range(len(y_actual))
    ax.plot(x_axis, y_actual, label='Actual', color='#2196F3', linewidth=1.5, alpha=0.9)
    ax.plot(x_axis, y_pred, label='Predicted', color='#FF5722', linewidth=1.5, alpha=0.8, linestyle='--')
    ax.set_title('Actual vs Predicted Close Price (LSTM)', fontweight='bold', pad=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price (Rs.)')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.fill_between(x_axis, y_actual, y_pred, alpha=0.1, color='gray')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/actual_vs_predicted.png", dpi=300, bbox_inches='tight')
    plt.close()


def _plot_training_loss(history, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(history['loss']) + 1)
    ax.plot(epochs, history['loss'], label='Train Loss', color='#2196F3', linewidth=2, marker='o', markersize=3)
    ax.plot(epochs, history['val_loss'], label='Val Loss', color='#FF5722', linewidth=2, marker='s', markersize=3)
    ax.set_title('Training Loss (MSE)', fontweight='bold', pad=15)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    best_epoch = np.argmin(history['val_loss']) + 1
    best_loss = min(history['val_loss'])
    ax.annotate(f'Best: {best_loss:.6f}\n(Epoch {best_epoch})',
                xy=(best_epoch, best_loss),
                xytext=(best_epoch + len(epochs) * 0.1,
                        best_loss + (max(history['val_loss']) - min(history['val_loss'])) * 0.2),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=10, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='red'))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_loss.png", dpi=300, bbox_inches='tight')
    plt.close()


def _plot_scatter(y_actual, y_pred, r2, output_dir):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_actual, y_pred, alpha=0.4, s=15, color='#4CAF50', edgecolors='#2E7D32', linewidth=0.3,
               label=f'Predictions (n={len(y_actual)})')
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_title(f'Predicted vs Actual\nR² = {r2:.4f}', fontweight='bold', pad=15)
    ax.set_xlabel('Actual Price (Rs.)')
    ax.set_ylabel('Predicted Price (Rs.)')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.text(0.95, 0.05, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scatter_plot.png", dpi=300, bbox_inches='tight')
    plt.close()


def _plot_error_distribution(y_actual, y_pred, output_dir):
    errors = y_actual - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=50, color='#9C27B0', edgecolor='black', alpha=0.75, linewidth=0.8)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax.axvline(x=errors.mean(), color='orange', linestyle='-', linewidth=2,
               label=f'Mean: Rs.{errors.mean():.2f}')
    ax.set_title('Prediction Error Distribution', fontweight='bold', pad=15)
    ax.set_xlabel('Error (Actual - Predicted) in Rs.')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    stats_text = f'Mean: Rs.{errors.mean():.2f}\nStd: Rs.{errors.std():.2f}\nMedian: Rs.{np.median(errors):.2f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()


def _plot_residuals(y_actual, y_pred, dates, output_dir):
    residuals = y_actual - y_pred
    fig, ax = plt.subplots(figsize=(14, 5))
    x_axis = dates[:len(residuals)] if dates is not None and len(dates) >= len(residuals) else range(len(residuals))
    ax.scatter(x_axis, residuals, alpha=0.4, s=10, color='#FF9800', edgecolors='#E65100', linewidth=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Residuals Over Time', fontweight='bold', pad=15)
    ax.set_xlabel('Date')
    ax.set_ylabel('Residual (Rs.)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/residuals.png", dpi=300, bbox_inches='tight')
    plt.close()


def _plot_comprehensive(y_test_actual, y_test_pred, y_train_actual, y_train_pred,
                        history, test_dates, metrics, output_dir):
    """Multi-panel summary figure."""
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    fig.suptitle('Stock Market Prediction — LSTM Model Evaluation',
                 fontsize=18, fontweight='bold', y=0.98)

    r2 = metrics['test_r2']
    rmse = metrics['test_rmse']
    mae = metrics['test_mae']

    ax1 = fig.add_subplot(gs[0, :])
    x_axis = test_dates[:len(y_test_actual)] if test_dates is not None and len(test_dates) >= len(y_test_actual) else range(len(y_test_actual))
    ax1.plot(x_axis, y_test_actual, label='Actual', color='#2196F3', linewidth=1.5)
    ax1.plot(x_axis, y_test_pred, label='Predicted', color='#FF5722', linewidth=1.5, linestyle='--')
    ax1.set_title('Actual vs Predicted Close Price (Test Set)', fontweight='bold')
    ax1.set_ylabel('Price (Rs.)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    epochs = range(1, len(history['loss']) + 1)
    ax2.plot(epochs, history['loss'], label='Train', color='#2196F3', linewidth=2)
    ax2.plot(epochs, history['val_loss'], label='Val', color='#FF5722', linewidth=2)
    ax2.set_title('Training Loss (MSE)', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE')
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(y_test_actual, y_test_pred, alpha=0.4, s=10, color='#4CAF50')
    min_v = min(y_test_actual.min(), y_test_pred.min())
    max_v = max(y_test_actual.max(), y_test_pred.max())
    ax3.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2)
    ax3.set_title(f'Predicted vs Actual (R²={r2:.4f})', fontweight='bold')
    ax3.set_xlabel('Actual (Rs.)')
    ax3.set_ylabel('Predicted (Rs.)')
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 2])
    errors = y_test_actual - y_test_pred
    ax4.hist(errors, bins=40, color='#9C27B0', edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax4.set_title('Error Distribution', fontweight='bold')
    ax4.set_xlabel('Error (Rs.)')
    ax4.set_ylabel('Frequency')
    ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs[2, 0])
    metric_names = ['RMSE', 'MAE', 'R²']
    metric_values = [rmse, mae, r2]
    colors = ['#F44336', '#FF9800', '#4CAF50']
    bars = ax5.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=2)
    ax5.set_title('Evaluation Metrics', fontweight='bold')
    for bar, val in zip(bars, metric_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_values)*0.02,
                f'{val:.4f}', ha='center', fontweight='bold', fontsize=10)
    ax5.grid(axis='y', alpha=0.3)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.scatter(range(len(errors)), errors, alpha=0.4, s=8, color='#FF9800')
    ax6.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax6.set_title('Residuals', fontweight='bold')
    ax6.set_xlabel('Sample')
    ax6.set_ylabel('Residual (Rs.)')
    ax6.grid(alpha=0.3)

    ax7 = fig.add_subplot(gs[2, 2])
    comp_metrics = ['RMSE', 'MAE']
    train_vals = [metrics['train_rmse'], metrics['train_mae']]
    test_vals = [metrics['test_rmse'], metrics['test_mae']]
    x = np.arange(len(comp_metrics))
    width = 0.35
    ax7.bar(x - width/2, train_vals, width, label='Train', color='#2196F3', edgecolor='black')
    ax7.bar(x + width/2, test_vals, width, label='Test', color='#FF5722', edgecolor='black')
    ax7.set_title('Train vs Test', fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(comp_metrics)
    ax7.legend()
    ax7.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comprehensive_evaluation.png", dpi=300, bbox_inches='tight')
    plt.close()
