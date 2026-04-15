"""Utility helpers for feature engineering and I/O."""


def ensure_dir(path: str):
    """Ensure a directory exists."""
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)
