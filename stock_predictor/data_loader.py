"""
Data Collection Module for Stock Market Prediction.

Paper Section III.A - Data Collection:
    "Historical stock data of NSE was used to collect the data. Other
    characteristics like past close, last traded price and trading data
    were also taken to capture detailed market behavior and trends with time."

This module handles downloading historical stock data from Yahoo Finance
and loading/saving CSV snapshots for reproducibility.
"""

import os
import pandas as pd
import yfinance as yf
from pathlib import Path
from stock_predictor.logging_utils import get_logger

logger = get_logger()


def download_stock_data(ticker: str, start: str, end: str,
                        save_dir: str = "data/raw") -> pd.DataFrame:
    """Download historical OHLCV stock data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., 'RELIANCE.NS' for NSE).
        start: Start date in 'YYYY-MM-DD' format.
        end: End date in 'YYYY-MM-DD' format.
        save_dir: Directory to save the raw CSV snapshot.

    Returns:
        pd.DataFrame: OHLCV data with DatetimeIndex.
    """
    logger.info(f"Downloading {ticker} data from {start} to {end}...")

    df = yf.download(ticker, start=start, end=end, progress=False)

    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}. Check ticker symbol and date range.")

    # Save raw snapshot for reproducibility
    os.makedirs(save_dir, exist_ok=True)
    start_compact = str(start).replace('-', '')
    end_compact = str(end).replace('-', '')
    ticker_clean = ticker.replace('.', '_').lower()
    save_path = os.path.join(save_dir, f"{ticker_clean}_{start_compact}_{end_compact}.csv")
    df.to_csv(save_path)

    logger.info(f"Downloaded {len(df)} rows | Date range: {df.index[0].date()} to {df.index[-1].date()}")
    logger.info(f"Saved raw data snapshot: {save_path}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Missing values:\n{df.isnull().sum().to_string()}")

    return df


def load_from_csv(path: str) -> pd.DataFrame:
    """Load stock data from a saved CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        pd.DataFrame: Stock data with DatetimeIndex.
    """
    df = pd.read_csv(Path(path), index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df
