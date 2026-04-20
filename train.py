import yaml
from stock_predictor.lstm_train import train_lstm


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_lstm(config)
    print("Training done. Run: python evaluate.py")


if __name__ == '__main__':
    main()
