"""
===========================================================================
PHASE 4 — MODELLING TECHNIQUES (HEART OF THE PROJECT)
===========================================================================
Course Topic: Time Series Forecasting Models

This script covers:
  CLASSICAL MODELS:
    1. Simple Moving Average (SMA)
    2. Simple Exponential Smoothing (SES)
    3. Holt-Winters (Triple Exponential Smoothing)
    4. SARIMA(p,d,q)(P,D,Q)[12] — the primary model

  ADVANCED MODELS:
    5. Facebook Prophet
    6. LSTM Neural Network

  For each model:
    - Fit on training data (80%)
    - Forecast on test data (20%)
    - Store predictions for Phase 5 evaluation
===========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
import pickle
import warnings
import os
import json

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ─── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOT_DIR = os.path.join(BASE_DIR, 'outputs', 'plots')
MODEL_DIR = os.path.join(BASE_DIR, 'outputs', 'models')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SUBDIVISIONS = ['COASTAL KARNATAKA', 'NORTH INTERIOR KARNATAKA', 'SOUTH INTERIOR KARNATAKA']
COLORS = {'COASTAL KARNATAKA': '#1565C0',
          'NORTH INTERIOR KARNATAKA': '#C62828',
          'SOUTH INTERIOR KARNATAKA': '#2E7D32'}

TRAIN_RATIO = 0.80  # 80-20 train-test split


def load_data():
    """Load time series data."""
    print("=" * 70)
    print("PHASE 4 — MODELLING TECHNIQUES")
    print("=" * 70)

    df_ts = pd.read_csv(os.path.join(DATA_DIR, 'karnataka_monthly_ts.csv'),
                        parse_dates=['DATE'], index_col='DATE')
    print(f"\n📂 Loaded: {df_ts.shape}")
    return df_ts


def get_series(df_ts, subdivision):
    """Get clean monthly series for a subdivision."""
    series = df_ts[df_ts['SUBDIVISION'] == subdivision]['RAINFALL_MM'].copy()
    series = series.asfreq('MS')
    series = series.interpolate(method='linear')
    # Replace any remaining NaN with 0 (rare dry months)
    series = series.fillna(0)
    # Ensure no negative values
    series = series.clip(lower=0)
    return series


def train_test_split_ts(series, train_ratio=TRAIN_RATIO):
    """Split time series into train and test sets."""
    n = len(series)
    split_idx = int(n * train_ratio)
    train = series[:split_idx]
    test = series[split_idx:]
    return train, test


def mape(actual, predicted):
    """Mean Absolute Percentage Error (handle zeros)."""
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 1: SIMPLE MOVING AVERAGE
# ═══════════════════════════════════════════════════════════════════════════
def model_moving_average(train, test, window=12):
    """Simple Moving Average forecast."""
    # For SMA, forecast = rolling average of last 'window' observations
    predictions = []
    history = list(train.values)

    for t in range(len(test)):
        ma = np.mean(history[-window:])
        predictions.append(ma)
        history.append(test.values[t])

    return pd.Series(predictions, index=test.index, name='SMA')


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 2: SIMPLE EXPONENTIAL SMOOTHING (SES)
# ═══════════════════════════════════════════════════════════════════════════
def model_ses(train, test):
    """Simple Exponential Smoothing."""
    model = SimpleExpSmoothing(train, initialization_method='estimated')
    fit = model.fit(optimized=True)

    predictions = fit.forecast(len(test))
    predictions.index = test.index
    predictions.name = 'SES'

    print(f"     SES α (smoothing level): {fit.params['smoothing_level']:.4f}")
    return predictions, fit


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 3: HOLT-WINTERS (TRIPLE EXPONENTIAL SMOOTHING)
# ═══════════════════════════════════════════════════════════════════════════
def model_holt_winters(train, test):
    """Holt-Winters with additive seasonality (period=12)."""
    model = ExponentialSmoothing(
        train,
        trend='add',
        seasonal='add',
        seasonal_periods=12,
        initialization_method='estimated'
    )
    fit = model.fit(optimized=True)

    predictions = fit.forecast(len(test))
    predictions.index = test.index
    predictions.name = 'Holt-Winters'

    print(f"     HW α={fit.params['smoothing_level']:.4f}, "
          f"β={fit.params['smoothing_trend']:.4f}, "
          f"γ={fit.params['smoothing_seasonal']:.4f}")
    return predictions, fit


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 4: SARIMA — THE PRIMARY MODEL
# ═══════════════════════════════════════════════════════════════════════════
def sarima_grid_search(train, seasonal_period=12):
    """
    Grid search for optimal SARIMA parameters.
    Tests common (p,d,q)(P,D,Q)[12] combinations.
    """
    print(f"     🔍 SARIMA grid search (this may take a few minutes)...")

    # Reduced search space for efficiency
    p_range = range(0, 3)
    d_range = [0, 1]
    q_range = range(0, 3)
    P_range = range(0, 2)
    D_range = [0, 1]
    Q_range = range(0, 2)

    best_aic = np.inf
    best_order = None
    best_seasonal = None
    results = []

    for p, d, q in itertools.product(p_range, d_range, q_range):
        for P, D, Q in itertools.product(P_range, D_range, Q_range):
            try:
                model = SARIMAX(train, order=(p, d, q),
                               seasonal_order=(P, D, Q, seasonal_period),
                               enforce_stationarity=False,
                               enforce_invertibility=False)
                fit = model.fit(disp=False, maxiter=200)
                aic = fit.aic

                results.append({
                    'order': (p, d, q),
                    'seasonal': (P, D, Q, seasonal_period),
                    'AIC': aic
                })

                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    best_seasonal = (P, D, Q, seasonal_period)
            except:
                continue

    print(f"     ✅ Best SARIMA: ({best_order})({best_seasonal[0]},{best_seasonal[1]},{best_seasonal[2]})[{seasonal_period}]")
    print(f"        AIC: {best_aic:.2f}")

    # Show top 5 models
    results_sorted = sorted(results, key=lambda x: x['AIC'])[:5]
    print(f"     📋 Top 5 models by AIC:")
    for i, r in enumerate(results_sorted):
        print(f"        {i+1}. SARIMA{r['order']}×{r['seasonal'][:3]}[12] — AIC={r['AIC']:.2f}")

    return best_order, best_seasonal


def model_sarima(train, test, order=None, seasonal_order=None, do_grid_search=True):
    """SARIMA model fitting and forecasting."""
    if do_grid_search and order is None:
        order, seasonal_order = sarima_grid_search(train)
    elif order is None:
        # Default fallback
        order = (1, 0, 1)
        seasonal_order = (1, 1, 1, 12)

    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=500)

    predictions = fit.forecast(len(test))
    predictions.index = test.index
    predictions.name = 'SARIMA'

    # Model summary
    print(f"\n     SARIMA Model Summary:")
    print(f"       Order: {order}")
    print(f"       Seasonal: {seasonal_order}")
    print(f"       AIC: {fit.aic:.2f}")
    print(f"       BIC: {fit.bic:.2f}")
    print(f"       Log Likelihood: {fit.llf:.2f}")

    return predictions, fit, order, seasonal_order


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 5: FACEBOOK PROPHET
# ═══════════════════════════════════════════════════════════════════════════
def model_prophet(train, test):
    """Facebook Prophet model."""
    from prophet import Prophet

    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_train = pd.DataFrame({
        'ds': train.index,
        'y': train.values
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10
    )
    model.fit(prophet_train)

    # Create future dataframe
    future = pd.DataFrame({'ds': test.index})
    forecast = model.predict(future)

    predictions = pd.Series(forecast['yhat'].values, index=test.index, name='Prophet')
    predictions = predictions.clip(lower=0)  # No negative rainfall

    # Also get confidence intervals
    lower = pd.Series(forecast['yhat_lower'].values, index=test.index).clip(lower=0)
    upper = pd.Series(forecast['yhat_upper'].values, index=test.index).clip(lower=0)

    return predictions, model, lower, upper


# ═══════════════════════════════════════════════════════════════════════════
# MODEL 6: LSTM NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════════════
def model_lstm(train, test, n_lag=12, n_epochs=50, n_units=64):
    """LSTM model for time series forecasting."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    # Suppress TF logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))

    # Create sequences
    def create_sequences(data, n_lag):
        X, y = [], []
        for i in range(n_lag, len(data)):
            X.append(data[i-n_lag:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_scaled, n_lag)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Build LSTM
    model = Sequential([
        LSTM(n_units, return_sequences=True, input_shape=(n_lag, 1)),
        Dropout(0.2),
        LSTM(n_units // 2, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    print(f"     Training LSTM ({n_units} units, {n_lag} lags, {n_epochs} epochs)...")
    history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=32,
                        verbose=0, callbacks=[early_stop])
    print(f"     Training complete (final loss: {history.history['loss'][-1]:.6f})")

    # Predict on test set (rolling forecast)
    predictions = []
    last_sequence = train_scaled[-n_lag:].flatten().tolist()

    for i in range(len(test)):
        seq = np.array(last_sequence[-n_lag:]).reshape(1, n_lag, 1)
        pred = model.predict(seq, verbose=0)[0, 0]
        predictions.append(pred)

        # Use actual test value for next step (walk-forward)
        actual_scaled = scaler.transform([[test.values[i]]])[0, 0]
        last_sequence.append(actual_scaled)

    # Inverse transform
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    predictions = np.clip(predictions, 0, None)  # No negative rainfall

    pred_series = pd.Series(predictions, index=test.index, name='LSTM')
    return pred_series, model, scaler


# ═══════════════════════════════════════════════════════════════════════════
# MAIN MODELLING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def run_all_models(df_ts, subdivision):
    """Run all 6 models for a given subdivision."""
    short = subdivision.replace(' KARNATAKA', '')
    print(f"\n{'━'*70}")
    print(f"  📍 {subdivision}")
    print(f"{'━'*70}")

    series = get_series(df_ts, subdivision)
    train, test = train_test_split_ts(series)

    print(f"  Train: {train.index[0].strftime('%Y-%m')} → {train.index[-1].strftime('%Y-%m')} ({len(train)} months)")
    print(f"  Test:  {test.index[0].strftime('%Y-%m')} → {test.index[-1].strftime('%Y-%m')} ({len(test)} months)")

    all_predictions = {}
    model_objects = {}

    # ─── Model 1: Moving Average ────────────────────────────────────────
    print(f"\n  📐 Model 1: Simple Moving Average (SMA-12)")
    pred_sma = model_moving_average(train, test, window=12)
    all_predictions['SMA'] = pred_sma
    print(f"     MAE: {mean_absolute_error(test, pred_sma):.2f} mm")

    # ─── Model 2: SES ────────────────────────────────────────────────────
    print(f"\n  📐 Model 2: Simple Exponential Smoothing (SES)")
    pred_ses, fit_ses = model_ses(train, test)
    all_predictions['SES'] = pred_ses
    model_objects['SES'] = fit_ses
    print(f"     MAE: {mean_absolute_error(test, pred_ses):.2f} mm")

    # ─── Model 3: Holt-Winters ──────────────────────────────────────────
    print(f"\n  📐 Model 3: Holt-Winters (Additive Trend + Seasonality)")
    pred_hw, fit_hw = model_holt_winters(train, test)
    all_predictions['Holt-Winters'] = pred_hw
    model_objects['Holt-Winters'] = fit_hw
    print(f"     MAE: {mean_absolute_error(test, pred_hw):.2f} mm")

    # ─── Model 4: SARIMA ────────────────────────────────────────────────
    print(f"\n  📐 Model 4: SARIMA (Grid Search)")
    pred_sarima, fit_sarima, order, seasonal = model_sarima(train, test, do_grid_search=True)
    all_predictions['SARIMA'] = pred_sarima
    model_objects['SARIMA'] = fit_sarima
    print(f"     MAE: {mean_absolute_error(test, pred_sarima):.2f} mm")

    # ─── Model 5: Prophet ────────────────────────────────────────────────
    print(f"\n  📐 Model 5: Facebook Prophet")
    pred_prophet, model_prophet_obj, lower_ci, upper_ci = model_prophet(train, test)
    all_predictions['Prophet'] = pred_prophet
    model_objects['Prophet'] = model_prophet_obj
    print(f"     MAE: {mean_absolute_error(test, pred_prophet):.2f} mm")

    # ─── Model 6: LSTM ──────────────────────────────────────────────────
    print(f"\n  📐 Model 6: LSTM Neural Network")
    pred_lstm, model_lstm_obj, scaler = model_lstm(train, test, n_lag=12, n_epochs=50)
    all_predictions['LSTM'] = pred_lstm
    model_objects['LSTM'] = model_lstm_obj
    print(f"     MAE: {mean_absolute_error(test, pred_lstm):.2f} mm")

    return train, test, all_predictions, model_objects


def plot_model_forecasts(train, test, all_predictions, subdivision):
    """Plot all model forecasts vs actual for comparison."""
    short = subdivision.replace(' KARNATAKA', '').replace(' ', '_')
    short_label = subdivision.replace(' KARNATAKA', '')

    model_colors = {
        'SMA': '#FF9800',
        'SES': '#9C27B0',
        'Holt-Winters': '#E91E63',
        'SARIMA': '#2196F3',
        'Prophet': '#4CAF50',
        'LSTM': '#FF5722'
    }

    # ─── Individual model plots ──────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(20, 16))
    fig.suptitle(f'Model Forecasts vs Actual — {subdivision}',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, (model_name, pred) in enumerate(all_predictions.items()):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        # Show last 60 months of training + all test
        train_show = train[-60:]
        ax.plot(train_show.index, train_show.values, linewidth=1,
                color='gray', alpha=0.5, label='Train (last 5 yrs)')
        ax.plot(test.index, test.values, linewidth=1.5,
                color='black', label='Actual')
        ax.plot(pred.index, pred.values, linewidth=1.5,
                color=model_colors[model_name], alpha=0.85, label=f'{model_name}')

        mae = mean_absolute_error(test, pred)
        rmse = np.sqrt(mean_squared_error(test, pred))
        ax.set_title(f'{model_name}\nMAE={mae:.1f}, RMSE={rmse:.1f}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylabel('Rainfall (mm)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, f'P4_01_model_forecasts_{short}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ Saved: P4_01_model_forecasts_{short}.png")

    # ─── Combined comparison plot ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.plot(test.index, test.values, linewidth=2, color='black',
            label='Actual', zorder=10)

    for model_name, pred in all_predictions.items():
        ax.plot(pred.index, pred.values, linewidth=1.5,
                color=model_colors[model_name], alpha=0.75,
                label=f'{model_name} (MAE={mean_absolute_error(test, pred):.1f})')

    ax.set_title(f'All Models — Forecast Comparison\n{subdivision}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rainfall (mm)', fontsize=12)
    ax.legend(fontsize=10, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'P4_02_combined_comparison_{short}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P4_02_combined_comparison_{short}.png")


def save_predictions(all_results):
    """Save all predictions for Phase 5 evaluation."""
    for sub, data in all_results.items():
        short = sub.replace(' KARNATAKA', '').replace(' ', '_')
        pred_df = pd.DataFrame({'ACTUAL': data['test']})
        for model_name, pred in data['predictions'].items():
            pred_df[model_name] = pred.values
        pred_path = os.path.join(DATA_DIR, f'predictions_{short}.csv')
        pred_df.to_csv(pred_path)
        print(f"   💾 Saved: predictions_{short}.csv")

    # Save SARIMA parameters
    params = {}
    for sub, data in all_results.items():
        if 'sarima_order' in data:
            params[sub] = {
                'order': list(data['sarima_order']),
                'seasonal_order': list(data['sarima_seasonal'])
            }
    with open(os.path.join(MODEL_DIR, 'sarima_params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    print(f"   💾 Saved: sarima_params.json")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    df_ts = load_data()

    all_results = {}

    for sub in SUBDIVISIONS:
        train, test, predictions, models = run_all_models(df_ts, sub)

        all_results[sub] = {
            'train': train,
            'test': test,
            'predictions': predictions,
            'models': models
        }

        # Store SARIMA params if available
        if 'SARIMA' in models:
            fit = models['SARIMA']
            all_results[sub]['sarima_order'] = fit.specification['order']
            all_results[sub]['sarima_seasonal'] = fit.specification['seasonal_order']

        plot_model_forecasts(train, test, predictions, sub)

    # Save predictions
    print(f"\n{'─'*70}")
    save_predictions(all_results)

    print(f"\n{'='*70}")
    print(f"✅ PHASE 4 COMPLETE — All 6 models trained and tested!")
    print(f"{'='*70}")
