import os
import pandas as pd
import yfinance as yf
from pathlib import Path
from stock_predictor.logging_utils import get_logger

logger = get_logger()


def download_stock_data(ticker: str, start: str, end: str,
                        save_dir: str = "data/raw") -> pd.DataFrame:
    """Download stock data from Yahoo Finance and save a local copy."""
    logger.info(f"Downloading {ticker} from {start} to {end}...")

    df = yf.download(ticker, start=start, end=end, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise ValueError(f"No data for {ticker}. Check ticker and date range.")

    os.makedirs(save_dir, exist_ok=True)
    start_compact = str(start).replace('-', '')
    end_compact = str(end).replace('-', '')
    ticker_clean = ticker.replace('.', '_').lower()
    save_path = os.path.join(save_dir, f"{ticker_clean}_{start_compact}_{end_compact}.csv")
    df.to_csv(save_path)

    logger.info(f"Downloaded {len(df)} rows | saved to {save_path}")
    return df


def load_from_csv(path: str) -> pd.DataFrame:
    """Load stock data from a CSV file."""
    df = pd.read_csv(Path(path), index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df
