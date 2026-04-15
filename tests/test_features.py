import pandas as pd
import numpy as np
from stock_predictor.features import engineer_features, ENGINEERED_FEATURES

def test_engineer_features_basic():
    # Minimal synthetic OHLCV data
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    df = pd.DataFrame({
        'Open': np.linspace(100, 120, 60),
        'High': np.linspace(101, 121, 60),
        'Low': np.linspace(99, 119, 60),
        'Close': np.linspace(100, 120, 60) + np.random.normal(0, 0.5, 60),
        'Volume': np.random.randint(1000000, 2000000, 60)
    }, index=dates)

    out = engineer_features(df, include_target=True).dropna()

    # Ensure all engineered features present
    for col in ENGINEERED_FEATURES:
        assert col in out.columns, f"Missing feature {col}"

    # Target should exist and be binary
    assert 'target' in out.columns
    assert set(out['target'].unique()).issubset({0,1})

    # No infinite values
    assert not np.isinf(out[ENGINEERED_FEATURES]).values.any()

def test_feature_row_alignment():
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'Open': 100 + np.random.rand(100).cumsum(),
        'High': 101 + np.random.rand(100).cumsum(),
        'Low': 99 + np.random.rand(100).cumsum(),
        'Close': 100 + np.random.rand(100).cumsum(),
        'Volume': np.random.randint(500000, 1500000, 100)
    }, index=dates)
    out = engineer_features(df, include_target=True)
    # After dropna we should have fewer rows
    cleaned = out.dropna()
    assert len(cleaned) < len(out)
