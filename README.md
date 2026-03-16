# 🌧️ Monthly Rainfall Trend & Forecasting in Karnataka

> A Time Series Analysis and Forecasting project on Karnataka's monthly rainfall data using classical and modern forecasting techniques.

---

## 📌 Project Overview

This project analyzes **120+ years of monthly rainfall data** across Karnataka (1901–2024) to uncover long-term trends, seasonal patterns, and build accurate forecasting models. The study covers all three rainfall subdivisions of Karnataka — **Coastal Karnataka**, **North Interior Karnataka**, and **South Interior Karnataka** — making it relevant for agriculture, water resource management, and climate policy.

This project is submitted as part of the course requirement for **Time Series Analysis & Forecasting Techniques**.

---

## 🎯 Objectives

- Perform exploratory and descriptive analysis of Karnataka's monthly rainfall time series
- Detect stationarity and decompose the series into Trend, Seasonal, and Residual components
- Build and compare multiple forecasting models — classical and deep learning
- Forecast monthly rainfall for the next 12 months with confidence intervals
- Derive actionable insights for monsoon prediction and agricultural planning

---

## 📂 Project Structure

```
karnataka-rainfall-forecasting/
│
├── data/
│   ├── rainfall_india.csv            # Raw dataset (Kaggle - Rainfall in India)
│   └── bengaluru_rainfall_opencity.csv  # Bengaluru city-level data (OpenCity)
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb   # Cleaning, missing value handling
│   ├── 02_exploratory_analysis.ipynb # Trends, seasonality, anomaly detection
│   ├── 03_stationarity_decomposition.ipynb  # ADF, KPSS, STL decomposition
│   ├── 04_model_arima_sarima.ipynb   # ARIMA and SARIMA modelling
│   ├── 05_model_exponential_smoothing.ipynb  # Holt-Winters model
│   ├── 06_model_prophet.ipynb        # Facebook Prophet model
│   ├── 07_model_lstm.ipynb           # LSTM deep learning model
│   └── 08_forecast_evaluation.ipynb  # Model comparison and final forecast
│
├── outputs/
│   ├── plots/                        # All visualizations
│   └── forecasts/                    # Forecast CSVs and reports
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

| Source | Description | Period | Link |
|--------|-------------|--------|------|
| Kaggle — Rainfall in India | State & subdivision-wise monthly rainfall | 1901–2015 | [Download](https://www.kaggle.com/datasets/rajanand/rainfall-in-india) |
| OpenCity — Bengaluru Rainfall | City-level monthly IMD data with El Niño flags | 1900–2024 | [Download](https://data.opencity.in/dataset/bengaluru-rainfall) |

**Karnataka Subdivisions covered:**
- Coastal Karnataka (Mangaluru, Udupi, Uttara Kannada)
- North Interior Karnataka (Belagavi, Dharwad, Kalaburagi)
- South Interior Karnataka (Bengaluru, Mysuru, Tumakuru)

---

## 🛠️ Techniques & Methods

### 1. Data Preprocessing
- Missing value imputation using forward fill and monthly mean interpolation
- Outlier detection (IQR method)
- Datetime indexing and resampling

### 2. Exploratory Analysis
- Monthly and annual rainfall trends (1901–2024)
- Season-wise distribution (Pre-monsoon, SW Monsoon, NE Monsoon, Winter)
- Regional comparison — Coastal vs Interior Karnataka
- Anomaly years detection (droughts: 2002, 2009 / floods: 1994, 2019)

### 3. Stationarity & Decomposition
- Augmented Dickey-Fuller (ADF) Test
- KPSS Test
- ACF and PACF plots
- STL and Classical Additive Decomposition

### 4. Forecasting Models

| Model | Type | Seasonal Handling |
|-------|------|-------------------|
| Simple Moving Average (SMA) | Classical | No |
| Exponential Smoothing (SES) | Classical | No |
| Holt-Winters | Classical | Yes |
| ARIMA | Statistical | No |
| SARIMA (p,d,q)(P,D,Q)[12] | Statistical | Yes ✅ |
| Facebook Prophet | ML-based | Yes ✅ |
| LSTM (Long Short-Term Memory) | Deep Learning | Yes ✅ |

### 5. Model Evaluation Metrics
- MAE — Mean Absolute Error
- RMSE — Root Mean Square Error
- MAPE — Mean Absolute Percentage Error
- Ljung-Box Test (residual autocorrelation check)
- Train-Test Split: 80% training / 20% testing (walk-forward validation)

---

## 📈 Key Results

- **Best performing model:** SARIMA(1,1,1)(1,1,1)[12] / Prophet (to be updated after experiments)
- **Dominant season:** Southwest Monsoon (June–September) contributes ~70% of annual rainfall
- **Trend observed:** Slight declining trend in annual rainfall over the last 30 years in interior Karnataka
- **12-month forecast:** Included in `outputs/forecasts/`

---

## 🌱 Domain Applications

- **Agricultural planning** — Kharif crop sowing decisions (rice, ragi, sugarcane)
- **Reservoir management** — Inflow prediction for Cauvery, Tungabhadra, Krishna basins
- **Drought/flood early warning** — Probability estimation using forecast confidence intervals
- **Monsoon onset prediction** — Predicted vs IMD normal onset date (June 1 for Karnataka coast)

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/karnataka-rainfall-forecasting.git
cd karnataka-rainfall-forecasting

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### requirements.txt
```
pandas
numpy
matplotlib
seaborn
statsmodels
scikit-learn
prophet
tensorflow
keras
pmdarima
scipy
jupyter
```

---

## 🧠 Course Alignment

| Course Topic | Project Coverage |
|---|---|
| Introduction to Time Series | Monthly rainfall series structure |
| Trend & Seasonality Analysis | STL decomposition, ACF/PACF |
| Stationarity Testing | ADF, KPSS tests |
| Smoothing Methods | SMA, SES, Holt-Winters |
| ARIMA Modelling | ARIMA, auto_arima |
| Seasonal Models | SARIMA with lag-12 seasonality |
| Model Evaluation | MAE, RMSE, MAPE, Ljung-Box |
| Advanced Forecasting | Prophet, LSTM |
| Forecasting Applications | Agriculture, water resources |

---



## 📜 License

This project is for academic purposes only. Dataset credits go to IMD (India Meteorological Department), Kaggle (rajanand), and OpenCity.

---

## 🔗 References

- India Meteorological Department — [mausam.imd.gov.in](https://mausam.imd.gov.in)
- Kaggle Rainfall Dataset — [kaggle.com/datasets/rajanand/rainfall-in-india](https://www.kaggle.com/datasets/rajanand/rainfall-in-india)
- OpenCity Bengaluru — [data.opencity.in](https://data.opencity.in/dataset/bengaluru-rainfall)
- KSNDMC Karnataka — [ksndmc.org](http://ksndmc.org)
- Facebook Prophet Docs — [facebook.github.io/prophet](https://facebook.github.io/prophet)
