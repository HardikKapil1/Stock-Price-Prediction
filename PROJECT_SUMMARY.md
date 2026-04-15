# 📊 Stock Market Prediction Using LSTM Networks — Project Summary

## 🎯 Project Status: COMPLETE ✅

---

## Overview

This project implements a **Long Short-Term Memory (LSTM)** neural network for
predicting stock market closing prices. It uses historical NSE stock data
(Reliance Industries) and follows a complete machine learning pipeline:
Data Collection → Preprocessing → Model Training → Evaluation → Dashboard.

## Architecture (Paper Section III.C)

```
Input (60 days × 1 feature)
  → LSTM Layer 1 (50 units, return_sequences=True) → Dropout(0.2)
  → LSTM Layer 2 (50 units, return_sequences=False) → Dropout(0.2)
  → Dense(1) → Predicted Price
```

- **Optimizer:** Adam (lr=0.001)
- **Loss:** Mean Squared Error (MSE)
- **Train/Test Split:** 80/20 (chronological)

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python train.py

# 3. Evaluate performance
python evaluate.py

# 4. Launch dashboard
streamlit run app/app.py
```

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error |
| **MAE** | Mean Absolute Error |
| **R²** | Coefficient of Determination |

## 📁 Key Files

| File | Description |
|------|-------------|
| `train.py` | Main training entry point |
| `evaluate.py` | Evaluation & visualization |
| `app/app.py` | Streamlit web dashboard |
| `config.yaml` | All configuration parameters |
| `stock_predictor/` | Core Python package |

## 👥 Team

- **Arpash Singh** — Student, Dept. of Data Science
- **Dhruv Rathi** — Student, Dept. of Data Science
- **Hardik Kapil** — Student, Dept. of Data Science
- **Mr. Hemant Kumar Baranval** — Guide, Asst. Professor

*Meerut Institute of Engineering and Technology (MIET)*

---

**Final Year Project • 2024-25**
