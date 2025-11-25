## Final Year Project Submission Checklist

This checklist ensures the repository is complete, reproducible, and academically robust at submission time.

### 1. Core Artifacts
- [ ] `train_final.py` (training pipeline with logging)
- [ ] `evaluate_final.py` (comprehensive evaluation figure generation)
- [ ] `app_final.py` (Streamlit dashboard)
- [ ] `stock_predictor/features.py` (centralized feature engineering)
- [ ] `stock_predictor/logging_utils.py` (standardized logging)
- [ ] `config.yaml` (ticker/date/test split parameters)
- [ ] `models/` (6 pkl artifacts: rf, gb, xgb, scaler, features list, weights)
- [ ] `outputs/final_predictions.csv`
- [ ] `outputs/feature_importance.csv`
- [ ] `outputs/final_evaluation_comprehensive.png`
- [ ] `README.md` (updated usage, reproducibility, disclaimer)
- [ ] `PROJECT_SUMMARY.md` (high-level narrative / abstract)
- [ ] `SUBMISSION_CHECKLIST.md` (this document)

### 2. Reproducibility
- [ ] Python version recorded (e.g. `python --version` in appendix)
- [ ] Dependency lock file: `requirements_locked.txt` (create with `pip freeze > requirements_locked.txt`)
- [ ] Optional raw data snapshot: `data/raw_tsla_20180101_20241231.csv` (use `fetch_data_snapshot.py`)
- [ ] Random seeds fixed (`random_state=42` in all models + SMOTE)
- [ ] No hidden state or environment variables required
- [ ] Raw data snapshot SHA256 checksum recorded: `0A9BC9B8F6E284E2A4A83373ED0C28680473C57F1815CFA8747077D5605A2EB4`

### 3. Verification
- [ ] All tests pass: `pytest -vv` screenshot/log
- [ ] Training run log captured (copy console output into appendix)
- [ ] Evaluation figure visually matches claims in report
- [ ] Feature importance top 10 included in report

### 4. Documentation & Academic Components
- [ ] Research question & hypothesis stated
- [ ] Methodology: data, split, features, models, metrics
- [ ] Baseline comparison (naive vs ensemble)
- [ ] Limitations + future work
- [ ] Ethical / financial disclaimer
- [ ] References / libraries acknowledged

### 5. Packaging
- [ ] Final git commit: `feat: submission-ready project artifacts`
- [ ] Git tag: `v1.0-submission`
- [ ] Optional GitHub Release with assets (`final_evaluation_comprehensive.png`, `requirements_locked.txt`)
- [ ] ZIP archive created (exclude `.venv/`):
  ```powershell
  git clean -xdf -e .venv
  powershell Compress-Archive -Path * -DestinationPath Stock-Price-Prediction-v1.0.zip -Force
  ```

### 6. Optional Enhancements (Not Required)
- CI workflow (GitHub Actions) running tests
- SHAP or permutation feature importance supplement
- CLI wrapper for training/evaluation
- Multi-ticker extension experiment

### 7. Submission Appendix Suggestions
- Raw console logs (training + evaluation)
- Test run output
- Dependency lock file
- Data snapshot checksum (e.g., SHA256 of raw CSV)
- Short rationale for feature set selection

### 8. Quick Commands
```powershell
python train_final.py
python evaluate_final.py
pytest -vv
pip freeze > requirements_locked.txt
python fetch_data_snapshot.py
git add .
git commit -m "feat: submission-ready project artifacts"
git tag v1.0-submission
git push origin main --tags
```

---
Tick every box before packaging. Include this file in your final ZIP/release.
