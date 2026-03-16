"""
===========================================================================
PHASE 6 — FORECASTING & INTERPRETATION
===========================================================================
Course Topic: Out-of-sample Forecasting with Confidence Intervals

This script covers:
  1. 12-month ahead forecast using the best model (SARIMA)
  2. Confidence intervals (80% and 95%)
  3. Forecast visualization with historical context
  4. Seasonal forecast interpretation
  5. Monsoon onset prediction vs IMD average
  6. Forecast comparison across models
  7. Uncertainty quantification
===========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import json
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ─── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOT_DIR = os.path.join(BASE_DIR, 'outputs', 'plots')
MODEL_DIR = os.path.join(BASE_DIR, 'outputs', 'models')
REPORT_DIR = os.path.join(BASE_DIR, 'outputs', 'reports')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

SUBDIVISIONS = ['COASTAL KARNATAKA', 'NORTH INTERIOR KARNATAKA', 'SOUTH INTERIOR KARNATAKA']
COLORS = {'COASTAL KARNATAKA': '#1565C0',
          'NORTH INTERIOR KARNATAKA': '#C62828',
          'SOUTH INTERIOR KARNATAKA': '#2E7D32'}

MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

# IMD historical average monsoon onset for Karnataka coast: June 1
# Interior Karnataka: June 5-10
IMD_MONSOON_ONSET = {
    'COASTAL KARNATAKA': 'June 1',
    'NORTH INTERIOR KARNATAKA': 'June 8',
    'SOUTH INTERIOR KARNATAKA': 'June 5'
}

FORECAST_HORIZON = 12  # 12 months ahead


def load_data():
    """Load full time series data and SARIMA parameters."""
    print("=" * 70)
    print("PHASE 6 — FORECASTING & INTERPRETATION")
    print("=" * 70)

    df_ts = pd.read_csv(os.path.join(DATA_DIR, 'karnataka_monthly_ts.csv'),
                        parse_dates=['DATE'], index_col='DATE')

    # Load SARIMA params
    params_path = os.path.join(MODEL_DIR, 'sarima_params.json')
    sarima_params = {}
    if os.path.exists(params_path):
        with open(params_path) as f:
            sarima_params = json.load(f)
        print(f"📂 SARIMA parameters loaded")
    else:
        print(f"⚠️  SARIMA params not found, using defaults")

    return df_ts, sarima_params


def get_series(df_ts, subdivision):
    """Get clean series."""
    series = df_ts[df_ts['SUBDIVISION'] == subdivision]['RAINFALL_MM'].copy()
    series = series.asfreq('MS')
    series = series.interpolate(method='linear').fillna(0).clip(lower=0)
    return series


def forecast_sarima(series, sarima_params, subdivision, horizon=12):
    """Fit SARIMA on full data and forecast ahead with confidence intervals."""
    if subdivision in sarima_params:
        order = tuple(sarima_params[subdivision]['order'])
        seasonal = tuple(sarima_params[subdivision]['seasonal_order'])
    else:
        order = (1, 0, 1)
        seasonal = (1, 1, 1, 12)

    print(f"  SARIMA{order}×{seasonal[:3]}[{seasonal[3]}]")

    model = SARIMAX(series, order=order, seasonal_order=seasonal,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=500)

    # Forecast
    forecast_result = fit.get_forecast(steps=horizon)
    forecast_mean = forecast_result.predicted_mean.clip(lower=0)

    # Confidence intervals
    ci_80 = forecast_result.conf_int(alpha=0.20)
    ci_95 = forecast_result.conf_int(alpha=0.05)

    # Clip negatives
    ci_80 = ci_80.clip(lower=0)
    ci_95 = ci_95.clip(lower=0)

    return forecast_mean, ci_80, ci_95, fit


def forecast_holt_winters(series, horizon=12):
    """Holt-Winters forecast for comparison."""
    model = ExponentialSmoothing(series, trend='add', seasonal='add',
                                 seasonal_periods=12, initialization_method='estimated')
    fit = model.fit(optimized=True)
    forecast = fit.forecast(horizon).clip(lower=0)
    return forecast


def forecast_prophet(series, horizon=12):
    """Prophet forecast for comparison."""
    from prophet import Prophet

    prophet_df = pd.DataFrame({'ds': series.index, 'y': series.values})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, changepoint_prior_scale=0.05)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=horizon, freq='MS')
    forecast = model.predict(future)

    forecast_values = forecast.tail(horizon)
    pred = pd.Series(forecast_values['yhat'].values,
                     index=pd.date_range(start=series.index[-1] + pd.DateOffset(months=1),
                                         periods=horizon, freq='MS'),
                     name='Prophet').clip(lower=0)
    lower = forecast_values['yhat_lower'].values.clip(min=0)
    upper = forecast_values['yhat_upper'].values.clip(min=0)
    return pred, lower, upper


def plot_forecast(series, forecast_mean, ci_80, ci_95, subdivision,
                  hw_forecast=None, prophet_forecast=None):
    """Plot 12-month forecast with confidence intervals and historical context."""
    short = subdivision.replace(' KARNATAKA', '').replace(' ', '_')
    short_label = subdivision.replace(' KARNATAKA', '')

    fig, ax = plt.subplots(figsize=(18, 8))

    # Historical (last 10 years)
    history = series[-120:]
    ax.plot(history.index, history.values, linewidth=1, color='gray',
            alpha=0.5, label='Historical (last 10 years)')

    # Historical rolling mean
    rolling = history.rolling(12).mean()
    ax.plot(rolling.index, rolling.values, linewidth=2, color='black',
            alpha=0.6, label='12-month rolling mean')

    # SARIMA Forecast
    ax.plot(forecast_mean.index, forecast_mean.values, 'o-', linewidth=2.5,
            color=COLORS[subdivision], markersize=8, label='SARIMA Forecast',
            zorder=5)

    # 80% CI
    ax.fill_between(ci_80.index, ci_80.iloc[:, 0], ci_80.iloc[:, 1],
                    alpha=0.3, color=COLORS[subdivision], label='80% CI')
    # 95% CI
    ax.fill_between(ci_95.index, ci_95.iloc[:, 0], ci_95.iloc[:, 1],
                    alpha=0.15, color=COLORS[subdivision], label='95% CI')

    # Holt-Winters comparison
    if hw_forecast is not None:
        ax.plot(hw_forecast.index, hw_forecast.values, 's--', linewidth=1.5,
                color='#E91E63', markersize=5, alpha=0.7, label='Holt-Winters')

    # Prophet comparison
    if prophet_forecast is not None:
        ax.plot(prophet_forecast.index, prophet_forecast.values, '^--', linewidth=1.5,
                color='#4CAF50', markersize=5, alpha=0.7, label='Prophet')

    # Monsoon zone highlighting
    forecast_year = forecast_mean.index[0].year
    monsoon_start = pd.Timestamp(f'{forecast_year}-06-01')
    monsoon_end = pd.Timestamp(f'{forecast_year}-09-30')
    if monsoon_start >= forecast_mean.index[0] and monsoon_start <= forecast_mean.index[-1]:
        ax.axvspan(monsoon_start, monsoon_end, alpha=0.08, color='blue')
        ax.text(monsoon_start + pd.DateOffset(days=45),
                ax.get_ylim()[1] * 0.95, 'MONSOON\nSEASON',
                fontsize=11, ha='center', color='navy', fontweight='bold', alpha=0.5)

    ax.set_title(f'12-Month Rainfall Forecast — {short_label} KARNATAKA\n'
                 f'SARIMA Model with 80% and 95% Confidence Intervals',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rainfall (mm)', fontsize=12)
    ax.legend(fontsize=10, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'P6_01_forecast_{short}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P6_01_forecast_{short}.png")


def plot_seasonal_forecast_bar(forecasts):
    """Bar chart of forecasted monthly rainfall for all subdivisions."""
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(12)
    width = 0.25

    for i, (sub, forecast) in enumerate(forecasts.items()):
        short = sub.replace(' KARNATAKA', ' KA')
        ax.bar(x + i*width, forecast.values, width, label=short,
               color=COLORS[sub], alpha=0.8, edgecolor='white')

    ax.set_xticks(x + width)
    ax.set_xticklabels(MONTHS, fontsize=11)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Forecasted Rainfall (mm)', fontsize=12)
    ax.set_title(f'12-Month Rainfall Forecast — All Karnataka Subdivisions',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Highlight monsoon
    ax.axvspan(4.5, 8.5, alpha=0.06, color='blue')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'P6_02_seasonal_forecast_bar.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ Saved: P6_02_seasonal_forecast_bar.png")


def interpret_forecast(forecasts, ci_95_all, series_all):
    """Interpret forecasts with domain context."""
    print(f"\n{'═'*70}")
    print(f"  FORECAST INTERPRETATION")
    print(f"{'═'*70}")

    for sub in SUBDIVISIONS:
        forecast = forecasts[sub]
        ci = ci_95_all[sub]
        series = series_all[sub]
        short = sub.replace(' KARNATAKA', '')

        print(f"\n  📍 {short} KARNATAKA")
        print(f"     {'─'*55}")

        # Monthly forecast table
        print(f"     {'Month':<8} {'Forecast':>10} {'95% Lower':>12} {'95% Upper':>12} {'Hist. Avg':>12}")
        print(f"     {'─'*55}")

        for month_idx, (date, val) in enumerate(forecast.items()):
            month_num = date.month
            month_name = MONTHS[month_num - 1]
            lower = ci.iloc[month_idx, 0]
            upper = ci.iloc[month_idx, 1]

            # Historical average for this month
            hist_avg = series[series.index.month == month_num].mean()

            anomaly = val - hist_avg
            anomaly_pct = (anomaly / hist_avg * 100) if hist_avg > 0 else 0

            print(f"     {month_name:<8} {val:>10.1f} {lower:>12.1f} {upper:>12.1f} "
                  f"{hist_avg:>12.1f} ({anomaly_pct:+.1f}%)")

        # Monsoon summary
        monsoon_months = [d for d in forecast.index if d.month in [6, 7, 8, 9]]
        if monsoon_months:
            monsoon_total = forecast[monsoon_months].sum()
            hist_monsoon = series[series.index.month.isin([6, 7, 8, 9])].groupby(
                series[series.index.month.isin([6, 7, 8, 9])].index.year).sum().mean()

            print(f"\n     Monsoon (Jun-Sep) Forecast: {monsoon_total:.0f} mm")
            print(f"     Historical Average:         {hist_monsoon:.0f} mm")
            print(f"     Deviation:                  {monsoon_total - hist_monsoon:+.0f} mm "
                  f"({(monsoon_total - hist_monsoon)/hist_monsoon*100:+.1f}%)")

        # Monsoon onset analysis
        print(f"\n     IMD Historical Monsoon Onset: {IMD_MONSOON_ONSET[sub]}")
        june_forecast = forecast[forecast.index.month == 6]
        if len(june_forecast) > 0:
            june_val = june_forecast.values[0]
            hist_june = series[series.index.month == 6].mean()
            if june_val > hist_june * 0.5:
                print(f"     Predicted June rainfall ({june_val:.0f} mm) suggests NORMAL onset")
            elif june_val > hist_june * 0.3:
                print(f"     Predicted June rainfall ({june_val:.0f} mm) suggests DELAYED onset")
            else:
                print(f"     Predicted June rainfall ({june_val:.0f} mm) suggests WEAK/LATE onset")

    # Annual comparison
    print(f"\n  📊 Annual Forecast Summary:")
    print(f"     {'Subdivision':<30} {'Forecast':>10} {'Historical':>12} {'Deviation':>12}")
    print(f"     {'─'*65}")
    for sub in SUBDIVISIONS:
        f_total = forecasts[sub].sum()
        h_avg = series_all[sub].groupby(series_all[sub].index.year).sum().mean()
        dev = f_total - h_avg
        short = sub.replace(' KARNATAKA', ' KA')
        print(f"     {short:<30} {f_total:>10.0f} {h_avg:>12.0f} {dev:>+12.0f} "
              f"({dev/h_avg*100:+.1f}%)")


def save_forecast_data(forecasts, ci_95_all):
    """Save forecast data to CSV."""
    for sub in SUBDIVISIONS:
        short = sub.replace(' KARNATAKA', '').replace(' ', '_')
        forecast_df = pd.DataFrame({
            'FORECAST': forecasts[sub],
            'CI_95_LOWER': ci_95_all[sub].iloc[:, 0].values,
            'CI_95_UPPER': ci_95_all[sub].iloc[:, 1].values
        })
        path = os.path.join(DATA_DIR, f'forecast_{short}.csv')
        forecast_df.to_csv(path)
        print(f"   💾 Saved: forecast_{short}.csv")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    df_ts, sarima_params = load_data()

    forecasts = {}
    ci_95_all = {}
    series_all = {}

    for sub in SUBDIVISIONS:
        short = sub.replace(' KARNATAKA', '')
        print(f"\n{'━'*70}")
        print(f"  📍 {short} KARNATAKA — Forecasting")
        print(f"{'━'*70}")

        series = get_series(df_ts, sub)
        series_all[sub] = series

        # SARIMA forecast (primary)
        print(f"\n  🔮 SARIMA Forecast:")
        forecast_mean, ci_80, ci_95, fit = forecast_sarima(
            series, sarima_params, sub, FORECAST_HORIZON)
        forecasts[sub] = forecast_mean
        ci_95_all[sub] = ci_95

        # Holt-Winters forecast (comparison)
        print(f"  🔮 Holt-Winters Forecast")
        hw_forecast = forecast_holt_winters(series, FORECAST_HORIZON)

        # Prophet forecast (comparison)
        print(f"  🔮 Prophet Forecast")
        prophet_forecast, _, _ = forecast_prophet(series, FORECAST_HORIZON)

        # Plot
        plot_forecast(series, forecast_mean, ci_80, ci_95, sub,
                      hw_forecast=hw_forecast, prophet_forecast=prophet_forecast)

    # Seasonal bar chart
    plot_seasonal_forecast_bar(forecasts)

    # Interpretation
    interpret_forecast(forecasts, ci_95_all, series_all)

    # Save
    print(f"\n{'─'*70}")
    save_forecast_data(forecasts, ci_95_all)

    print(f"\n{'='*70}")
    print(f"✅ PHASE 6 COMPLETE — 12-month forecasts generated!")
    print(f"{'='*70}")
