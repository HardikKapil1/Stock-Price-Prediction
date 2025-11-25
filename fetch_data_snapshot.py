"""Utility script to archive a raw OHLCV data snapshot for reproducibility.

Creates `data/raw_tsla_2018_2024.csv` (or for custom ticker/date range from config.yaml).
Run before submission so evaluators can verify identical source data.
"""

import os
import yaml
import yfinance as yf
import pandas as pd

def main():
    with open('config.yaml', 'r') as cf:
        cfg = yaml.safe_load(cf)
    ticker = cfg['Ticker']
    start = cfg['StartDate']
    end = cfg['EndDate']

    print(f"Downloading raw data for {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    os.makedirs('data', exist_ok=True)
    out_path = f"data/raw_{ticker.lower()}_{start.replace('-', '')}_{end.replace('-', '')}.csv"
    df.to_csv(out_path)
    print(f"Saved snapshot: {out_path} ({len(df)} rows)")
    print("Suggested: compute checksum -> certutil -hashfile " + out_path + " SHA256")

if __name__ == '__main__':
    main()
