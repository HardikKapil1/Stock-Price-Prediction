"""
Main Evaluation Script — Stock Market Prediction Using LSTM Networks.

This script runs comprehensive evaluation on the trained LSTM model,
computing RMSE, MAE, R² metrics and generating publication-quality
visualizations for the research paper.

Usage:
    python evaluate.py

Prerequisites:
    - Run train.py first to generate model artifacts.
"""

from stock_predictor.lstm_evaluate import evaluate_lstm


def main():
    """Run full LSTM model evaluation."""
    metrics = evaluate_lstm()

    print("\n✨ Evaluation complete!")
    print(f"   📊 Test RMSE: ₹{metrics['test_rmse']:.2f}")
    print(f"   📊 Test MAE:  ₹{metrics['test_mae']:.2f}")
    print(f"   📊 Test R²:   {metrics['test_r2']:.4f}")
    print(f"\n   📁 Check outputs/ directory for all plots and metrics.")
    print(f"   🚀 Next step: streamlit run app/app.py")


if __name__ == '__main__':
    main()
