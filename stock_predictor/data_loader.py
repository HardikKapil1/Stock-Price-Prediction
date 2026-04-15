"""Data loading utilities for stock predictor."""
import pandas as pd
from pathlib import Path


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV into a DataFrame."""
    return pd.read_csv(Path(path))
