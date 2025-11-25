import os
import pickle

def test_model_artifacts_exist():
    required = [
        'models/final_rf.pkl',
        'models/final_gb.pkl',
        'models/final_xgb.pkl',
        'models/final_scaler.pkl',
        'models/final_features.pkl',
        'models/final_weights.pkl'
    ]
    for path in required:
        assert os.path.exists(path), f"Missing artifact: {path}"

def test_can_load_models():
    with open('models/final_rf.pkl', 'rb') as f:
        rf = pickle.load(f)
    assert hasattr(rf, 'predict_proba')
