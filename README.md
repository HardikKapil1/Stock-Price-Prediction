# 📈 Stock Market Prediction Using Long Short-Term Memory (LSTM) Networks

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)

**Final Year Project • Meerut Institute of Engineering and Technology (MIET)**
**Department of Data Science**

---

## 📄 Abstract

Stabilizations of stock market prices vary rapidly and the market is unpredictable and hard to forecast the prices; therefore, the need of robust time-varying models like the LSTM. To enhance the quality of predictions, this study uses a number of financial features to feed the model with more information and richer features. It performs the required preprocessing, including data cleaning, data normalization and the generation of time windows to make sure that the dataset is prepared well to be trained in the LSTM architecture. A stacked LSTM network is then trained to know both the short-term and long-term changes in the price in the market. The model performance is measured by RMSE, MAE and R² in order to obtain clear and precise measurement of performance in terms of prediction accuracy. The results indicate that LSTM is better than traditional models like ARIMA and linear regression because it is more appropriate in predicting stocks in a stock market.

**Index Terms** — Stock Prediction, Long Short-Term Memory (LSTM), Deep Learning, Financial Data Analysis, Python Programming, Machine Learning Techniques.

## 👥 Authors

| Name | Role | Email |
|------|------|-------|
| Arpash Singh | Student, Dept. of Data Science | arpash.singh.cseds.2022@miet.ac.in |
| Dhruv Rathi | Student, Dept. of Data Science | dhruv.rathi.cseds.2022@miet.ac.in |
| Hardik Kapil | Student, Dept. of Data Science | hardik.kapil.cseds.2022@miet.ac.in |
| Mr. Hemant Kumar Baranval | Guide, Asst. Professor | hemant.baranval@miet.ac.in |

## 🏗️ Project Structure

```
Stock-Price-Prediction/
├── config.yaml                    # Configuration (ticker, LSTM params)
├── requirements.txt               # Python dependencies
├── train.py                       # Main training entry point
├── evaluate.py                    # Main evaluation entry point
├── fetch_data_snapshot.py         # Data snapshot for reproducibility
│
├── stock_predictor/               # Core Python package
│   ├── __init__.py
│   ├── data_loader.py             # Data Collection (Section III.A)
│   ├── preprocessing.py           # Data Preprocessing (Section III.B)
│   ├── lstm_model.py              # LSTM Architecture (Section III.C)
│   ├── lstm_train.py              # Model Training (Section III.D)
│   ├── lstm_evaluate.py           # Performance Evaluation (Section III.E)
│   └── logging_utils.py           # Logging utility
│
├── app/
│   └── app.py                     # Streamlit Web Dashboard
│
├── data/
│   ├── raw/                       # Raw OHLCV data
│   └── processed/                 # Processed sequences
│
├── models/
│   ├── lstm_model.keras           # Trained LSTM model
│   └── scaler.pkl                 # MinMaxScaler
│
├── outputs/
│   ├── predictions.csv            # Actual vs Predicted prices
│   ├── metrics.json               # RMSE, MAE, R²
│   ├── training_history.json      # Epoch-by-epoch loss
│   └── plots/                     # Evaluation charts (300 DPI)
│       ├── actual_vs_predicted.png
│       ├── training_loss.png
│       ├── scatter_plot.png
│       ├── error_distribution.png
│       ├── residuals.png
│       └── comprehensive_evaluation.png
│
├── tests/                         # Unit tests
├── README.md                      # This file
└── PROJECT_SUMMARY.md             # Quick summary
```

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/HardikKapil1/Stock-Price-Prediction.git
cd Stock-Price-Prediction
```

### 2. Create Virtual Environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow (~500MB) is required for LSTM model training.

## 🚀 Quick Start

### Step 1: Train the LSTM Model
```bash
python train.py
```
This will:
- Download NSE stock data (Reliance Industries) from 2018-2024
- Apply MinMaxScaler normalization
- Create 60-day sliding window sequences
- Train a Stacked LSTM (2 layers × 50 units, dropout 0.2)
- Save the trained model and scaler

### Step 2: Evaluate Performance
```bash
python evaluate.py
```
This generates:
- **RMSE, MAE, R²** metrics
- **Publication-quality plots** (Actual vs Predicted, Loss Curves, Scatter Plot, etc.)
- All outputs saved to `outputs/` directory

### Step 3: Launch Dashboard
```bash
streamlit run app/app.py
```
Opens an interactive dashboard at `http://localhost:8501` with:
- Live market data visualization
- LSTM predictions overlay
- Performance metrics with gauges
- Training history charts
- Model architecture details

## 🔬 Methodology

### III.A — Data Collection
Historical stock data of NSE (Reliance Industries — RELIANCE.NS) is collected using the Yahoo Finance API. Past close prices, trading volumes, and OHLC data capture detailed market behavior and trends over time (2018–2024).

### III.B — Data Preprocessing
- **Missing Values:** Forward-fill + remaining NaN removal
- **Normalization:** MinMaxScaler to [0, 1] range
- **Sequence Creation:** 60-day sliding windows for LSTM input
- **Split:** 80% training, 20% testing (chronological, no look-ahead bias)

### III.C — LSTM Model Architecture
```
Input (60 timesteps × 1 feature)
    → LSTM Layer 1 (50 units, return_sequences=True)
    → Dropout (0.2)
    → LSTM Layer 2 (50 units, return_sequences=False)
    → Dropout (0.2)
    → Dense (1 neuron — price output)
```
**Optimizer:** Adam | **Loss:** Mean Squared Error (MSE)

### III.D — Model Training
- Adam optimizer with MSE loss
- 80/20 chronological train-test split
- Multi-epoch sequential training with validation tracking
- Training history saved for loss curve analysis

### III.E — Performance Evaluation
| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error — magnitude of prediction errors |
| **MAE** | Mean Absolute Error — average absolute deviation |
| **R²** | Coefficient of Determination — proportion of variance explained |

## 📊 Flowchart

```
Data Collection → Data Preprocessing → Feature Extraction
    → Model Training → Prediction → Model Evaluation → Decision Making
```

## ⚙️ Configuration

Edit `config.yaml` to customize:
```yaml
Ticker: "RELIANCE.NS"     # NSE stock ticker
StartDate: "2018-01-01"
EndDate: "2024-12-31"
TestSize: 0.20             # 80/20 split
SequenceLength: 60         # Lookback window
LSTMUnits: 50             # Units per layer
Dropout: 0.2              # Regularization
Epochs: 50                # Training epochs
BatchSize: 32
LearningRate: 0.001
```

## 📚 References

1. Sidhu, P., Aggarwal, H., & Lal, M. (2021). Stock Market Prediction Using LSTM. *International Journal of Advanced Research in Science, Communication and Technology.*
2. Patil, P.P., Khanbare, N., Nadke, K., Gupta, D., & Sahani, D. (2022). Stock Market Prediction using LSTM. *International Journal of Advanced Research in Science, Communication and Technology.*
3. Sharma, R., Jain, S., Singh, S., & Nitie Ke-rani, M. (2020). Stock Market Prediction Using LSTM. *RealAxis Analytics.*
4. Lu, Z., Yu, H., Xu, J., Liu, J., & Mo, Y. (2021). Stock Market Analysis and Prediction Using LSTM: A Case Study on Technology Stocks. *Innovations in Applied Engineering and Technology.*
5. Guar, V. (2023). Stock Market Price Prediction Using LSTM. *International Journal for Research in Applied Science and Engineering Technology.*
6. Bhatane, P., Barasode, P., & Palaki, P. (2023). Stock Market Prediction Model using LSTM. *International Journal for Research in Applied Science and Engineering Technology.*
7. Chen, Y. (2021). Stock Market Analysis and Prediction Using LSTM. *BCP Business & Management.*
8. Chaudhary, R. (2025). Advanced Stock Market Prediction Using Long Short-Term Memory Networks: A Comprehensive Deep Learning Framework.
9. Sharma, Y., Kumar, A., Dubey, V., & Rai, V. (2023). Stock Price Prediction Using LSTM. *13th International Conference on Computing Communication and Networking Technologies (ICCCNT).*
10. Shale, A. (2025). Predicting Stock Market Trends Using Deep Long Short-Term Memory Networks. *International Journal of Scientific Research in Engineering and Management.*

## ⚠️ Disclaimer

**This is an educational project for academic purposes only.**

Stock prediction involves substantial risk. The model's performance reflects the inherent difficulty of market prediction and should NOT be used for actual trading without extensive additional research and risk management. Always consult licensed financial advisors before making investment decisions.

## 📝 License

MIT License — see [LICENSE](LICENSE) file

---

**Final Year Project • 2024-25 • Meerut Institute of Engineering and Technology**
