from stock_predictor.lstm_evaluate import evaluate_lstm


def main():
    metrics = evaluate_lstm()
    print(f"\nTest RMSE: Rs.{metrics['test_rmse']:.2f}")
    print(f"Test MAE:  Rs.{metrics['test_mae']:.2f}")
    print(f"Test R2:   {metrics['test_r2']:.4f}")
    print("Check outputs/ for plots and metrics.")


if __name__ == '__main__':
    main()
