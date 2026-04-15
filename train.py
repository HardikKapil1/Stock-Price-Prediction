"""
Main Training Script -- Stock Market Prediction Using LSTM Networks.

This script orchestrates the complete training pipeline as described
in the research paper. Run this first before evaluation.

Usage:
    python train.py

Configuration is read from config.yaml.
"""

import yaml
from stock_predictor.lstm_train import train_lstm


def main():
    """Load configuration and run the LSTM training pipeline."""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    results = train_lstm(config)

    print("\n>> Training pipeline complete!")
    print("   Next step: python evaluate.py")


if __name__ == '__main__':
    main()
