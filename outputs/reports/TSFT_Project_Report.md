# Monthly Rainfall Trend & Forecasting in Karnataka
## Time Series & Forecasting — Course Project Report

---

## 1. INTRODUCTION & MOTIVATION

### 1.1 Background

Rainfall is the most critical climate variable for a predominantly agrarian economy like India. Karnataka, a state with diverse geographic zones — from the lush Western Ghats coast to the arid Deccan plateau — experiences significant spatial and temporal variability in rainfall. This variability directly impacts agriculture (Kharif crop planning), water resource management (Cauvery & Tungabhadra reservoir operations), and disaster preparedness (droughts and floods).

### 1.2 Project Objective

The primary objective of this project is to **analyze, decompose, model, and forecast monthly rainfall patterns** across three Karnataka subdivisions using 115 years (1901–2015) of historical data from the India Meteorological Department (IMD). The goal is to:

1. Understand the temporal structure of rainfall — trend, seasonality, and residual components
2. Test for stationarity using formal statistical tests
3. Build and compare multiple time series forecasting models — from classical to deep learning
4. Generate actionable 12-month ahead forecasts with confidence intervals
5. Connect the forecasts to real-world applications (agriculture, reservoirs, drought risk)

### 1.3 Why This Project?

Karnataka's three IMD subdivisions — **Coastal Karnataka**, **North Interior Karnataka**, and **South Interior Karnataka** — offer dramatically different rainfall regimes:

- Coastal Karnataka receives ~3,400 mm/year (heavy Western Ghats orographic rainfall)
- North Interior Karnataka receives only ~718 mm/year (rain-shadow semi-arid zone)
- South Interior Karnataka falls in between at ~1,040 mm/year

This **4.7× contrast** within a single state provides an ideal testbed for comparing how different time series models perform under different data characteristics — high variance vs. low variance, heavy seasonality vs. moderate seasonality.

---

## 2. DATASET

### 2.1 Data Source

- **Source:** India Meteorological Department (IMD) rainfall dataset
- **Platform:** Kaggle — "Rainfall in India" by Rajanand Ilangovan
- **File:** `rainfall in india 1901-2015.csv`
- **Total Records:** 4,116 rows × 19 columns (36 Indian subdivisions)

### 2.2 Dataset Structure

| Column | Description | Type |
|--------|-------------|------|
| SUBDIVISION | Geographic region name | String |
| YEAR | Year of observation (1901–2015) | Integer |
| JAN – DEC | Monthly rainfall in millimeters | Float |
| ANNUAL | Total annual rainfall (mm) | Float |
| Jan-Feb, Mar-May, Jun-Sep, Oct-Dec | Seasonal aggregates (mm) | Float |

### 2.3 Karnataka Subset

After filtering, we obtained **345 records** (115 years × 3 subdivisions), which were reshaped into **4,140 monthly observations** for time series analysis.

### 2.4 Data Quality

| Issue | Count | Treatment |
|-------|-------|-----------|
| Missing monthly values | 2 cells | Linear interpolation + forward/backward fill |
| Outlier years | 4 flagged | Retained (real extreme climate events) |
| Negative values | 0 | N/A |

---

## 3. METHODOLOGY

### 3.1 Technology Stack

| Component | Tool/Library | Purpose |
|-----------|-------------|---------|
| Language | Python 3.11 | Core programming |
| Data handling | Pandas, NumPy | Dataframes, numerical computation |
| Visualization | Matplotlib, Seaborn | Charts and plots |
| Statistical testing | Statsmodels | ADF, KPSS, Ljung-Box, ACF/PACF |
| Classical models | Statsmodels | SES, Holt-Winters, SARIMA |
| Prophet | Facebook Prophet | Automatic trend + seasonality |
| Deep learning | TensorFlow/Keras | LSTM neural network |
| Signal processing | SciPy | Spectral analysis (periodogram) |
| Dataset download | KaggleHub | Programmatic data access |

### 3.2 Project Pipeline

The project follows a structured 7-phase pipeline, each mapping directly to core topics in a Time Series & Forecasting course:

```
Phase 1: Data Foundation
    ↓
Phase 2: Descriptive & Exploratory Analysis
    ↓
Phase 3: Stationarity Testing & Decomposition
    ↓
Phase 4: Model Building (6 models)
    ↓
Phase 5: Model Evaluation & Residual Diagnostics
    ↓
Phase 6: 12-Month Ahead Forecasting
    ↓
Phase 7: Domain Applications
```

---

## 4. PHASE-WISE ANALYSIS & RESULTS

### Phase 1 — Data Foundation

**Course Topic Covered:** Time Series Data Collection, Structure, and Pre-processing

**What We Did:**
- Loaded the raw IMD CSV dataset (4,116 records, 36 subdivisions)
- Filtered to 3 Karnataka subdivisions: Coastal, North Interior, South Interior
- Handled 2 missing values using linear interpolation within each subdivision
- Detected outliers using the IQR (Interquartile Range) method:
  - Coastal KA: 1961 (5,554 mm — extreme flood year)
  - North Interior KA: 1916, 1975, 1997 (all >1,050 mm — above upper fence)
- Outliers were **flagged but not removed** — they represent real extreme climate events
- Reshaped data from wide format (1 row/year, 12 month columns) to long format (1 row/month) with a proper `DatetimeIndex` — essential for time series analysis
- Saved cleaned datasets for all subsequent phases

**Key Output:** `karnataka_monthly_ts.csv` — 4,140 rows of monthly rainfall with DatetimeIndex

---

### Phase 2 — Descriptive & Exploratory Analysis

**Course Topic Covered:** Visual and Statistical Exploration of Time Series

**What We Did:**
- Generated **8 types of visualizations** to understand the data:
  1. Seasonal box plots by month — reveals monsoon dominance
  2. Mean monthly rainfall profile with ±1σ bands
  3. Annual trends with linear regression and 10-year moving averages
  4. Heatmaps of monthly rainfall across 115 years
  5. Extreme year identification (anomalies > ±2σ from mean)
  6. Seasonal contribution donut charts
  7. Decade-wise bar chart comparison
  8. Inter-month correlation matrices

**Key Findings:**

| Metric | Coastal KA | N. Interior KA | S. Interior KA |
|--------|-----------|----------------|----------------|
| Mean Annual Rainfall | 3,406 mm | 718 mm | 1,040 mm |
| Monsoon Contribution (Jun-Sep) | **87.5%** | **69.9%** | **65.8%** |
| Coefficient of Variation | 140.0% | 107.6% | 97.9% |
| Skewness (monthly) | 1.464 | 1.064 | 0.895 |
| Wettest Year | 1961 (5,554 mm) | 1997 (1,096 mm) | 1961 (1,410 mm) |
| Driest Year | 1918 (2,511 mm) | 1920 (470 mm) | 1918 (733 mm) |

**Insight:** Coastal Karnataka's rainfall is overwhelmingly concentrated in the monsoon (87.5%), while interior regions show more spread across seasons. The 1918 drought and 1961 flood appear as extreme events across multiple subdivisions.

---

### Phase 3 — Stationarity & Decomposition

**Course Topic Covered:** Stationarity Testing, Time Series Decomposition, ACF/PACF Analysis

**What We Did:**

#### 3a. Stationarity Testing

Two formal tests were applied to each subdivision's monthly rainfall series:

**Augmented Dickey-Fuller (ADF) Test** — H₀: Series is non-stationary
- Tests for unit root in the time series
- If p-value < 0.05, we reject H₀ → series is stationary

**KPSS Test** — H₀: Series is stationary (complementary to ADF)
- If p-value > 0.05, we fail to reject H₀ → series is stationary

| Subdivision | ADF p-value | ADF Result | KPSS p-value | KPSS Result | Joint Verdict |
|------------|-------------|------------|-------------|-------------|---------------|
| Coastal KA | 0.000011 | ✅ Stationary | 0.100 | ✅ Stationary | **Both agree: STATIONARY** |
| N. Interior KA | 0.000000 | ✅ Stationary | 0.100 | ✅ Stationary | **Both agree: STATIONARY** |
| S. Interior KA | 0.000000 | ✅ Stationary | 0.100 | ✅ Stationary | **Both agree: STATIONARY** |

**Why this matters:** Stationarity is a prerequisite for ARIMA-family models. Since the original series is already stationary (strong seasonality doesn't violate stationarity in the mean sense), we can use d=0 or d=1 in our SARIMA model.

#### 3b. Decomposition

Two methods were applied:

1. **Classical Additive Decomposition:** Y(t) = Trend + Seasonal + Residual
   - Additive model chosen because rainfall variance doesn't scale with level
   - Reveals clear 12-month seasonal cycle

2. **STL (Seasonal-Trend decomposition using LOESS):**
   - More robust to outliers
   - Produces smoother trend component
   - Seasonal component confirms the dominant monsoon cycle

#### 3c. ACF & PACF Analysis

- **ACF (Auto-Correlation Function):** Shows strong positive spikes at lags 12, 24, and 36, confirming the **annual seasonality (s=12)**
- **PACF (Partial ACF):** Cuts off after a few lags — helpful for determining the AR order (p) in SARIMA
- After seasonal differencing (D=1, s=12), the ACF decays much faster, confirming seasonality removal

#### 3d. Spectral Analysis

Periodogram analysis reveals a dominant peak at **frequency = 1 cycle/year**, confirming the annual rainfall cycle. No significant sub-annual or multi-year cycles were detected.

---

### Phase 4 — Modelling Techniques (Heart of the Project)

**Course Topic Covered:** Time Series Forecasting Models

**Train-Test Split:** 80% training (1901–1992) / 20% testing (1993–2015) = 1,104 / 276 months

Six models were implemented, progressing from simple to advanced:

#### Model 1: Simple Moving Average (SMA-12)
- **Theory:** Forecast = average of last 12 observations
- **Strengths:** Simple, intuitive
- **Weakness:** Cannot capture seasonality — forecast is a flat line
- **Use case:** Baseline comparison only

#### Model 2: Simple Exponential Smoothing (SES)
- **Theory:** Forecast = weighted average giving exponentially decreasing weights to older observations. Parameter α controls the rate of decay.
- **Strengths:** Adapts to recent changes
- **Weakness:** No trend or seasonal component — single-level forecast
- **Result:** α = 1.0 for Coastal KA (meaning it essentially uses the last observation as forecast)

#### Model 3: Holt-Winters (Triple Exponential Smoothing)
- **Theory:** Extends SES with three components:
  - Level (α) — smoothed average
  - Trend (β) — smoothed trend
  - Seasonal (γ) — smoothed seasonal pattern with period s=12
- **Configuration:** Additive trend + additive seasonality, period=12
- **Result:** Captures the monsoon cycle extremely well. γ values are small (0.026–0.035), meaning the seasonal pattern is very stable across years.

#### Model 4: SARIMA — The Primary Model
- **Theory:** Seasonal ARIMA combines:
  - AR(p): Autoregressive component — past values predict future
  - I(d): Integration — differencing to achieve stationarity
  - MA(q): Moving average — past forecast errors predict future
  - Seasonal (P,D,Q)[s]: Same components applied at seasonal lag s=12
- **Grid Search:** Tested multiple (p,d,q)(P,D,Q)[12] combinations, selected by lowest AIC (Akaike Information Criterion)
- **Best orders found:**

| Subdivision | SARIMA Order | AIC |
|------------|-------------|-----|
| Coastal KA | **(0,1,2)(1,1,1)[12]** | 13,716.70 |
| N. Interior KA | **(0,1,2)(0,1,1)[12]** | 10,859.45 |
| S. Interior KA | **(0,1,2)(1,1,1)[12]** | 11,176.58 |

**Interpretation:** All three subdivisions share the same non-seasonal structure (0,1,2) — meaning no AR component is needed, first-order differencing, and 2 MA terms. The seasonal part includes seasonal differencing (D=1) and seasonal MA (Q=1), which is typical for monthly climate data.

#### Model 5: Facebook Prophet
- **Theory:** Decomposes time series into:
  - Trend: Piecewise linear with automatic changepoint detection
  - Seasonality: Fourier series representation of annual cycle
  - Holidays: Not applicable for rainfall
- **Configuration:** yearly_seasonality=True, changepoint_prior_scale=0.05 (conservative)
- **Strengths:** Fully automatic, handles missing data and outliers gracefully

#### Model 6: LSTM Neural Network
- **Theory:** Long Short-Term Memory networks learn non-linear temporal dependencies through gated memory cells. Unlike ARIMA, LSTMs can capture complex non-linear patterns.
- **Architecture:**
  - LSTM(64 units, return_sequences=True) → Dropout(0.2)
  - LSTM(32 units) → Dropout(0.2)
  - Dense(32, relu) → Dense(1)
- **Training:** 50 epochs, batch size=32, MSE loss, Adam optimizer, early stopping
- **Walk-forward prediction:** Uses actual values (not predictions) as input for next step — realistic evaluation

#### Model Performance Comparison (Test Set: 1993–2015)

**Coastal Karnataka (MAE in mm):**

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| SMA | 324.41 | 401.03 | -0.008 |
| SES | 295.21 | 496.69 | -0.546 |
| **Holt-Winters** | **82.23** | **135.75** | **0.885** |
| **SARIMA** | 81.61 | 136.68 | 0.883 |
| Prophet | 81.53 | 136.94 | 0.882 |
| LSTM | 80.84 | 143.91 | 0.870 |

**North Interior Karnataka (MAE in mm):**

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| SMA | 53.88 | 63.81 | -0.010 |
| SES | 51.56 | 72.09 | -0.289 |
| Holt-Winters | 23.78 | 35.71 | 0.684 |
| **SARIMA** | **22.39** | 35.39 | 0.689 |
| Prophet | 23.62 | 35.69 | 0.684 |
| **LSTM** | 23.08 | **34.62** | **0.703** |

**South Interior Karnataka (MAE in mm):**

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| SMA | 73.80 | 86.74 | -0.017 |
| SES | 78.23 | 108.76 | -0.599 |
| **Holt-Winters** | 27.75 | **39.52** | **0.789** |
| **SARIMA** | **27.64** | 39.65 | 0.787 |
| Prophet | 29.09 | 40.67 | 0.776 |
| LSTM | 29.92 | 42.83 | 0.752 |

**Key Observation:** The seasonal models (Holt-Winters, SARIMA, Prophet, LSTM) reduce MAE by **70–75%** compared to simple models (SMA, SES). Among the seasonal models, performance is remarkably similar, with SARIMA and Holt-Winters consistently performing best.

---

### Phase 5 — Model Evaluation

**Course Topic Covered:** Forecast Accuracy Metrics and Residual Diagnostics

#### 5a. Accuracy Metrics Used

1. **MAE (Mean Absolute Error):** Average absolute forecast error in mm — interpretable in original units
2. **RMSE (Root Mean Squared Error):** Penalizes large errors more heavily — important for extreme rainfall
3. **MAPE (Mean Absolute Percentage Error):** Percentage error — problematic for rainfall due to many zero/near-zero values in dry months
4. **R² (Coefficient of Determination):** Proportion of variance explained by the model

#### 5b. Ljung-Box Residual Test

The **Ljung-Box test** checks whether model residuals are uncorrelated (white noise). If residuals retain autocorrelation, the model has failed to capture some structure.

- **H₀:** Residuals are independently distributed (no autocorrelation)
- **Decision:** p > 0.05 → Adequate model (residuals are white noise)

| Model | Coastal KA | N. Interior KA | S. Interior KA |
|-------|-----------|----------------|----------------|
| SMA | ❌ p=0.000 | ❌ p=0.000 | ❌ p=0.000 |
| SES | ❌ p=0.000 | ❌ p=0.000 | ❌ p=0.000 |
| Holt-Winters | ✅ p=0.999 | ✅ p=0.105 | ✅ p=0.883 |
| SARIMA | ✅ p=0.997 | ✅ p=0.129 | ✅ p=0.912 |
| Prophet | ✅ p=0.996 | ✅ p=0.167 | ✅ p=0.751 |
| LSTM | ✅ p=0.252 | ✅ p=0.396 | ✅ p=0.745 |

**Conclusion:** SMA and SES fail the Ljung-Box test because they cannot capture seasonality — their residuals still contain the seasonal pattern. All four seasonal models pass, confirming they've adequately modelled the autocorrelation structure.

#### 5c. Residual Distribution

Residual histograms with normal distribution overlays and Shapiro-Wilk normality tests were generated to verify the Gaussian assumption required by SARIMA's confidence intervals.

#### 5d. Radar Charts

Multi-metric radar charts provide a visual comparison of all models across MAE, RMSE, MAPE, and R² — making it easy to identify the best overall model.

---

### Phase 6 — Forecasting & Interpretation

**Course Topic Covered:** Out-of-Sample Forecasting with Confidence Intervals

**What We Did:**
- Re-fitted the best SARIMA model on the **entire dataset** (1901–2015)
- Generated **12-month ahead forecasts** with 80% and 95% confidence intervals
- Also produced Holt-Winters and Prophet forecasts for comparison
- Compared forecasted monthly values against historical averages

#### 12-Month Forecast Summary

| Subdivision | Forecast Annual | Historical Avg | Deviation |
|------------|----------------|---------------|-----------|
| Coastal KA | 3,572 mm | 3,406 mm | **+4.9%** (slightly above normal) |
| N. Interior KA | 707 mm | 718 mm | **-1.6%** (near normal) |
| S. Interior KA | 1,101 mm | 1,040 mm | **+5.9%** (slightly above normal) |

#### Monsoon Forecast (Jun-Sep)

| Subdivision | Forecast Monsoon | Historical Avg | IMD Onset | Onset Prediction |
|------------|-----------------|---------------|-----------|-----------------|
| Coastal KA | 3,044 mm | 2,982 mm | June 1 | **Normal onset** |
| N. Interior KA | 501 mm | 502 mm | June 8 | **Normal onset** |
| S. Interior KA | 721 mm | 684 mm | June 5 | **Normal onset** |

All three subdivisions predict near-normal to slightly above-normal rainfall with normal monsoon onset.

---

### Phase 7 — Domain Applications

**Course Topic Covered:** Real-World Applications of Time Series Forecasting

#### 7a. Kharif Crop Planning

Based on forecasted monsoon rainfall (Jun-Sep), crop sowing advisories were generated:

| Crop | Region | Forecast Monsoon | Advisory |
|------|--------|-----------------|----------|
| Rice (Paddy) | Coastal KA | 3,044 mm | 🌊 EXCESS — Flood risk |
| Rice (Paddy) | S. Interior KA | 721 mm | ⚠️ INSUFFICIENT — Needs irrigation |
| Ragi (Finger Millet) | S. Interior KA | 721 mm | 👍 ADEQUATE |
| Jowar (Sorghum) | N. Interior KA | 501 mm | ✅ OPTIMAL |
| Maize | S. Interior KA | 721 mm | ✅ OPTIMAL |
| Groundnut | N. Interior KA | 501 mm | ✅ OPTIMAL |

#### 7b. Reservoir Inflow Estimation

Using simple runoff coefficients and catchment areas:

| Reservoir | Basin | Forecast Inflow | Capacity | Expected Fill |
|-----------|-------|-----------------|----------|---------------|
| KRS | Cauvery | 144.6 TMC | 49.5 TMC | 292% |
| Tungabhadra Dam | Tungabhadra | 210.9 TMC | 100.9 TMC | 209% |
| Linganamakki Dam | Sharavathi | 113.0 TMC | 151.8 TMC | 74% |

#### 7c. Drought Probability

Based on historical distribution and forecasts:

| Subdivision | Historical Drought Frequency | Forecast Verdict |
|------------|----------------------------|-----------------|
| Coastal KA | 15.7% moderate, 0% severe | ✅ No drought risk |
| N. Interior KA | 16.5% moderate, 0% severe | ✅ No drought risk |
| S. Interior KA | 17.4% moderate, 1.7% severe | ✅ No drought risk |

#### 7d. Climate Change Trend Detection

Linear regression on 115 years of annual rainfall:

| Subdivision | Trend | p-value | Significant? |
|------------|-------|---------|-------------|
| Coastal KA | **+3.84 mm/year** | 0.004 | ✅ Yes |
| S. Interior KA | **+1.08 mm/year** | 0.010 | ✅ Yes |
| N. Interior KA | +0.37 mm/year | 0.325 | ❌ No |

**Insight:** Coastal Karnataka shows a statistically significant increasing rainfall trend over 115 years (+3.84 mm/year = ~442 mm total increase since 1901). This aligns with climate change projections for the Western Ghats region.

---

## 5. ALIGNMENT WITH COURSE SYLLABUS

This project comprehensively covers every major topic in a Time Series & Forecasting course:

| Course Topic | Project Phase | How It's Covered |
|-------------|---------------|-----------------|
| TS data structure & datetime indexing | Phase 1 | Wide→Long reshape, DatetimeIndex, freq='MS' |
| Missing value handling & outlier detection | Phase 1 | Interpolation, IQR method |
| Visual exploration of TS | Phase 2 | 8 types of charts, seasonal analysis |
| Descriptive statistics | Phase 2 | Mean, median, CV, skewness, kurtosis |
| Stationarity testing | Phase 3 | ADF test, KPSS test, rolling statistics |
| Time series decomposition | Phase 3 | Classical additive, STL (robust) |
| ACF & PACF analysis | Phase 3 | Lag analysis, seasonal lag identification |
| Spectral analysis | Phase 3 | Periodogram, dominant frequency detection |
| Moving averages | Phase 4 | SMA-12 baseline model |
| Exponential smoothing (SES) | Phase 4 | Optimal α estimation |
| Holt-Winters (triple ES) | Phase 4 | Additive trend + seasonal, α/β/γ optimization |
| ARIMA / SARIMA modelling | Phase 4 | Grid search for (p,d,q)(P,D,Q)[12], AIC selection |
| Prophet | Phase 4 | Automatic changepoint + Fourier seasonality |
| LSTM neural networks | Phase 4 | Sequence-to-one, walk-forward prediction |
| MAE, RMSE, MAPE metrics | Phase 5 | Full comparison table across 6 models |
| Ljung-Box residual test | Phase 5 | Autocorrelation adequacy of residuals |
| Residual diagnostics | Phase 5 | ACF of residuals, normality check (Shapiro-Wilk) |
| Forecasting with confidence intervals | Phase 6 | 80% and 95% CI from SARIMA |
| Model comparison & selection | Phase 5-6 | Radar charts, multi-metric ranking |
| Domain applications | Phase 7 | Crop planning, reservoir, drought, climate change |

---

## 6. CONCLUSION

1. **Karnataka's rainfall is strongly seasonal** — the monsoon (Jun-Sep) contributes 66–88% of annual rainfall, making seasonal models essential.

2. **SARIMA and Holt-Winters are the best performers** — both achieve R² > 0.78 across all subdivisions, with SARIMA(0,1,2)(1,1,1)[12] emerging as the optimal specification.

3. **Simple models (SMA, SES) fail completely** — they cannot capture the seasonal structure and their residuals retain strong autocorrelation (Ljung-Box p ≈ 0).

4. **Advanced models (Prophet, LSTM) offer marginal improvement** over classical seasonal models for this dataset. The seasonal pattern in Karnataka rainfall is highly regular, which classical models can capture efficiently.

5. **Climate change signal detected** — Coastal Karnataka shows a statistically significant increase in rainfall (+3.84 mm/year, p=0.004), worth monitoring for long-term water resource planning.

6. **Forecasts suggest near-normal rainfall** — all three subdivisions predict rainfall within ±6% of historical average, with normal monsoon onset.

---

## 7. REFERENCES

1. India Meteorological Department (IMD) — Monthly Rainfall Data, 1901–2015
2. Box, G.E.P., Jenkins, G.M., & Reinsel, G.C. — *Time Series Analysis: Forecasting and Control* (Wiley)
3. Hyndman, R.J. & Athanasopoulos, G. — *Forecasting: Principles and Practice* (OTexts)
4. Taylor, S.J. & Letham, B. — *Forecasting at Scale* (Facebook Prophet, 2018)
5. Hochreiter, S. & Schmidhuber, J. — *Long Short-Term Memory* (Neural Computation, 1997)
6. Kaggle Dataset: Rajanand Ilangovan — "Rainfall in India" (kaggle.com/rajanand/rainfall-in-india)

---

## 8. TOOLS & LIBRARIES

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.11 | Programming language |
| Pandas | 2.x | Data manipulation |
| NumPy | 1.x | Numerical computing |
| Matplotlib | 3.x | Static plotting |
| Seaborn | 0.13+ | Statistical visualization |
| Statsmodels | 0.14.6 | SARIMA, ADF, KPSS, Ljung-Box |
| Prophet | 1.3.0 | Facebook Prophet model |
| TensorFlow | 2.x | LSTM neural network |
| Scikit-learn | 1.x | MinMaxScaler, metrics |
| SciPy | 1.x | Spectral analysis, statistical tests |
| KaggleHub | 1.0.0 | Dataset download |

---

*Report generated for Time Series & Forecasting Course Project*
*Dataset: IMD Rainfall in India (1901–2015)*
*Analysis performed: March 2026*
