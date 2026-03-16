"""
===========================================================================
PHASE 3 — STATIONARITY & DECOMPOSITION
===========================================================================
Course Topic: Stationarity Testing, Time Series Decomposition, ACF/PACF

This script covers:
  1. Augmented Dickey-Fuller (ADF) test for stationarity
  2. KPSS test for stationarity (complementary to ADF)
  3. Rolling mean & variance analysis (visual stationarity check)
  4. Classical (additive) decomposition — Trend + Seasonal + Residual
  5. STL decomposition (robust seasonal-trend decomposition)
  6. ACF (Auto-Correlation Function) plots
  7. PACF (Partial Auto-Correlation Function) plots
  8. Differencing to achieve stationarity
  9. Periodogram / spectral analysis
===========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import signal
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ─── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOT_DIR = os.path.join(BASE_DIR, 'outputs', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

SUBDIVISIONS = ['COASTAL KARNATAKA', 'NORTH INTERIOR KARNATAKA', 'SOUTH INTERIOR KARNATAKA']
COLORS = {'COASTAL KARNATAKA': '#1565C0',
          'NORTH INTERIOR KARNATAKA': '#C62828',
          'SOUTH INTERIOR KARNATAKA': '#2E7D32'}


def load_data():
    """Load time series data from Phase 1."""
    print("=" * 70)
    print("PHASE 3 — STATIONARITY & DECOMPOSITION")
    print("=" * 70)

    df_ts = pd.read_csv(os.path.join(DATA_DIR, 'karnataka_monthly_ts.csv'),
                        parse_dates=['DATE'], index_col='DATE')
    print(f"\n📂 Loaded time series data: {df_ts.shape}")
    return df_ts


def get_subdivision_series(df_ts, subdivision):
    """Extract a single subdivision's monthly rainfall as a clean Series."""
    series = df_ts[df_ts['SUBDIVISION'] == subdivision]['RAINFALL_MM'].copy()
    series = series.asfreq('MS')  # Monthly start frequency
    series = series.interpolate(method='linear')  # Fill any gaps
    return series


def adf_test(series, name=""):
    """Perform Augmented Dickey-Fuller test."""
    result = adfuller(series.dropna(), autolag='AIC')
    output = {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Lags Used': result[2],
        'Observations': result[3],
        'Critical Values': result[4],
        'Stationary': result[1] < 0.05
    }
    return output


def kpss_test(series, name=""):
    """Perform KPSS test for stationarity."""
    result = kpss(series.dropna(), regression='c', nlags='auto')
    output = {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Lags Used': result[2],
        'Critical Values': result[3],
        'Stationary': result[1] > 0.05  # KPSS: H0 = stationary
    }
    return output


def test_stationarity(df_ts):
    """Run ADF and KPSS tests on all subdivisions."""
    print(f"\n{'═'*70}")
    print(f"  STATIONARITY TESTING")
    print(f"{'═'*70}")

    results = {}
    for sub in SUBDIVISIONS:
        series = get_subdivision_series(df_ts, sub)
        adf = adf_test(series, sub)
        kpss_r = kpss_test(series, sub)

        results[sub] = {'ADF': adf, 'KPSS': kpss_r}

        short = sub.replace(' KARNATAKA', '')
        print(f"\n  📍 {short} KARNATAKA:")
        print(f"     {'─'*55}")

        # ADF Test
        print(f"     ADF Test (H₀: Non-stationary):")
        print(f"       Test Statistic: {adf['Test Statistic']:.4f}")
        print(f"       p-value:        {adf['p-value']:.6f}")
        for key, val in adf['Critical Values'].items():
            marker = " ←" if abs(adf['Test Statistic']) > abs(val) else ""
            print(f"       Critical ({key}): {val:.4f}{marker}")
        verdict = "✅ STATIONARY" if adf['Stationary'] else "❌ NON-STATIONARY"
        print(f"       Verdict:        {verdict}")

        # KPSS Test
        print(f"     KPSS Test (H₀: Stationary):")
        print(f"       Test Statistic: {kpss_r['Test Statistic']:.4f}")
        print(f"       p-value:        {kpss_r['p-value']:.4f}")
        for key, val in kpss_r['Critical Values'].items():
            print(f"       Critical ({key}): {val:.4f}")
        verdict = "✅ STATIONARY" if kpss_r['Stationary'] else "❌ NON-STATIONARY"
        print(f"       Verdict:        {verdict}")

        # Joint interpretation
        if adf['Stationary'] and kpss_r['Stationary']:
            interpretation = "Both tests agree: Series is STATIONARY"
        elif not adf['Stationary'] and not kpss_r['Stationary']:
            interpretation = "Both tests agree: Series is NON-STATIONARY"
        elif adf['Stationary'] and not kpss_r['Stationary']:
            interpretation = "Conflicting: Trend-stationary (needs differencing for trend)"
        else:
            interpretation = "Conflicting: Difference-stationary"
        print(f"     📋 Interpretation: {interpretation}")

    return results


def plot_rolling_stationarity(df_ts):
    """Visual stationarity check — rolling mean and standard deviation."""
    print(f"\n📊 Rolling Mean & Variance Analysis...")

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle('Visual Stationarity Check — Rolling Statistics',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, sub in enumerate(SUBDIVISIONS):
        series = get_subdivision_series(df_ts, sub)

        # Rolling mean
        rolling_mean = series.rolling(window=12).mean()
        rolling_std = series.rolling(window=12).std()

        # Plot rolling mean
        axes[idx, 0].plot(series.index, series.values, alpha=0.3,
                          color=COLORS[sub], label='Original')
        axes[idx, 0].plot(rolling_mean.index, rolling_mean.values,
                          linewidth=2, color='black', label='Rolling Mean (12m)')
        axes[idx, 0].fill_between(rolling_mean.index,
                                   (rolling_mean - rolling_std).values,
                                   (rolling_mean + rolling_std).values,
                                   alpha=0.15, color=COLORS[sub])
        short = sub.replace(' KARNATAKA', '')
        axes[idx, 0].set_title(f'{short} KA — Rolling Mean', fontsize=11, fontweight='bold')
        axes[idx, 0].set_ylabel('Rainfall (mm)')
        axes[idx, 0].legend(fontsize=8)

        # Rolling std
        axes[idx, 1].plot(rolling_std.index, rolling_std.values,
                          linewidth=1.5, color=COLORS[sub])
        axes[idx, 1].axhline(y=rolling_std.mean(), color='red', linestyle='--',
                             label=f'Mean σ = {rolling_std.mean():.1f}')
        axes[idx, 1].set_title(f'{short} KA — Rolling Std Dev', fontsize=11, fontweight='bold')
        axes[idx, 1].set_ylabel('Std Dev (mm)')
        axes[idx, 1].legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, 'P3_01_rolling_stationarity.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P3_01_rolling_stationarity.png")


def perform_decomposition(df_ts):
    """Classical additive decomposition and STL decomposition."""
    print(f"\n📊 Time Series Decomposition...")

    for sub in SUBDIVISIONS:
        series = get_subdivision_series(df_ts, sub)
        short = sub.replace(' KARNATAKA', '').replace(' ', '_')

        # ─── Classical Decomposition (Additive) ─────────────────────────
        decomp = seasonal_decompose(series, model='additive', period=12)

        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        title = f'Classical Additive Decomposition — {sub}'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        axes[0].plot(series.index, series.values, linewidth=0.5, color=COLORS[sub])
        axes[0].set_ylabel('Observed', fontsize=11)
        axes[0].set_title('Original Series', fontsize=11)

        axes[1].plot(decomp.trend.index, decomp.trend.values, linewidth=2, color='black')
        axes[1].set_ylabel('Trend', fontsize=11)
        axes[1].set_title('Trend Component', fontsize=11)

        axes[2].plot(decomp.seasonal.index, decomp.seasonal.values,
                     linewidth=1, color='green')
        axes[2].set_ylabel('Seasonal', fontsize=11)
        axes[2].set_title('Seasonal Component (Period = 12 months)', fontsize=11)

        axes[3].plot(decomp.resid.index, decomp.resid.values,
                     linewidth=0.5, color='gray', alpha=0.7)
        axes[3].set_ylabel('Residual', fontsize=11)
        axes[3].set_title('Residual Component', fontsize=11)
        axes[3].axhline(y=0, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(PLOT_DIR, f'P3_02_classical_decomp_{short}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: P3_02_classical_decomp_{short}.png")

        # ─── STL Decomposition (Robust) ─────────────────────────────────
        stl = STL(series, period=12, robust=True)
        stl_result = stl.fit()

        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        title = f'STL Decomposition (Robust) — {sub}'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        axes[0].plot(series.index, series.values, linewidth=0.5, color=COLORS[sub])
        axes[0].set_ylabel('Observed', fontsize=11)

        axes[1].plot(stl_result.trend.index, stl_result.trend.values,
                     linewidth=2, color='black')
        axes[1].set_ylabel('Trend', fontsize=11)

        axes[2].plot(stl_result.seasonal.index, stl_result.seasonal.values,
                     linewidth=1, color='green')
        axes[2].set_ylabel('Seasonal', fontsize=11)

        axes[3].plot(stl_result.resid.index, stl_result.resid.values,
                     linewidth=0.5, color='gray', alpha=0.7)
        axes[3].set_ylabel('Residual', fontsize=11)
        axes[3].axhline(y=0, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(PLOT_DIR, f'P3_03_stl_decomp_{short}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: P3_03_stl_decomp_{short}.png")


def plot_acf_pacf(df_ts):
    """ACF and PACF plots to identify autocorrelation structure."""
    print(f"\n📊 ACF & PACF Analysis...")

    for sub in SUBDIVISIONS:
        series = get_subdivision_series(df_ts, sub)
        short = sub.replace(' KARNATAKA', '').replace(' ', '_')

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Autocorrelation Analysis — {sub}',
                     fontsize=14, fontweight='bold', y=0.98)

        # ACF on original series
        plot_acf(series.dropna(), lags=48, ax=axes[0, 0],
                 title='ACF — Original Series', alpha=0.05)
        axes[0, 0].axvline(x=12, color='red', linestyle='--', alpha=0.5, label='Lag 12')
        axes[0, 0].axvline(x=24, color='red', linestyle='--', alpha=0.5, label='Lag 24')
        axes[0, 0].axvline(x=36, color='red', linestyle='--', alpha=0.5, label='Lag 36')
        axes[0, 0].legend(fontsize=8)

        # PACF on original series
        plot_pacf(series.dropna(), lags=48, ax=axes[0, 1],
                  title='PACF — Original Series', alpha=0.05, method='ywm')

        # Seasonal differencing (d=0, D=1, s=12)
        series_diff = series.diff(12).dropna()

        # ACF on seasonal differenced series
        plot_acf(series_diff.dropna(), lags=48, ax=axes[1, 0],
                 title='ACF — After Seasonal Differencing (D=1, s=12)', alpha=0.05)

        # PACF on seasonal differenced series
        plot_pacf(series_diff.dropna(), lags=48, ax=axes[1, 1],
                  title='PACF — After Seasonal Differencing (D=1, s=12)',
                  alpha=0.05, method='ywm')

        for ax_row in axes:
            for ax in ax_row:
                ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(PLOT_DIR, f'P3_04_acf_pacf_{short}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: P3_04_acf_pacf_{short}.png")

    # Print ACF interpretation
    print(f"\n   📋 ACF/PACF Interpretation:")
    print(f"     • Strong positive spikes at lags 12, 24, 36 → confirms annual seasonality")
    print(f"     • After seasonal differencing (D=1), ACF decays faster → seasonality removed")
    print(f"     • PACF cuts off after a few lags → helps determine AR order (p)")
    print(f"     • Suggested model: SARIMA(p,0,q)(P,1,Q)[12]")


def differencing_analysis(df_ts):
    """Test stationarity after different differencing strategies."""
    print(f"\n📊 Differencing Analysis to Achieve Stationarity...")

    results = {}
    for sub in SUBDIVISIONS:
        series = get_subdivision_series(df_ts, sub)
        short = sub.replace(' KARNATAKA', '')

        print(f"\n  📍 {short} KARNATAKA:")

        # Original
        adf_orig = adf_test(series)
        print(f"     Original:           ADF p={adf_orig['p-value']:.6f} "
              f"{'✅' if adf_orig['Stationary'] else '❌'}")

        # First difference (d=1)
        diff1 = series.diff().dropna()
        adf_d1 = adf_test(diff1)
        print(f"     1st Difference:     ADF p={adf_d1['p-value']:.6f} "
              f"{'✅' if adf_d1['Stationary'] else '❌'}")

        # Seasonal difference (D=1, s=12)
        diff_s = series.diff(12).dropna()
        adf_ds = adf_test(diff_s)
        print(f"     Seasonal Diff(12):  ADF p={adf_ds['p-value']:.6f} "
              f"{'✅' if adf_ds['Stationary'] else '❌'}")

        # Both (d=1, D=1, s=12)
        diff_both = series.diff(12).diff().dropna()
        adf_both = adf_test(diff_both)
        print(f"     Both (d=1,D=1):     ADF p={adf_both['p-value']:.6f} "
              f"{'✅' if adf_both['Stationary'] else '❌'}")

        results[sub] = {
            'original': adf_orig,
            'diff1': adf_d1,
            'seasonal_diff': adf_ds,
            'both': adf_both
        }

    return results


def spectral_analysis(df_ts):
    """Periodogram to identify dominant frequencies/cycles."""
    print(f"\n📊 Spectral Analysis (Periodogram)...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle('Spectral Analysis — Power Spectral Density',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, sub in enumerate(SUBDIVISIONS):
        series = get_subdivision_series(df_ts, sub)

        # Compute periodogram
        freqs, power = signal.periodogram(series.values, fs=12)  # fs=12 months/year

        # Plot
        axes[idx].semilogy(freqs, power, linewidth=1, color=COLORS[sub], alpha=0.8)

        # Mark dominant frequencies
        top_indices = np.argsort(power)[-5:][::-1]
        for ti in top_indices:
            if freqs[ti] > 0:
                period = 1 / freqs[ti]
                if period < 200:  # Only annotate meaningful periods
                    axes[idx].axvline(x=freqs[ti], color='red', linestyle='--', alpha=0.3)
                    axes[idx].annotate(f'{period:.1f} yr', (freqs[ti], power[ti]),
                                      fontsize=8, color='red', fontweight='bold')

        # Mark annual cycle
        axes[idx].axvline(x=1.0, color='orange', linestyle='--', alpha=0.7)
        axes[idx].text(1.02, axes[idx].get_ylim()[1] * 0.5, '1-year\ncycle',
                       fontsize=9, color='orange', fontweight='bold')

        short = sub.replace(' KARNATAKA', '')
        axes[idx].set_title(f'{short} KARNATAKA', fontsize=12,
                           fontweight='bold', color=COLORS[sub])
        axes[idx].set_xlabel('Frequency (cycles/year)', fontsize=10)
        axes[idx].set_ylabel('Power Spectral Density', fontsize=10)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, 'P3_05_spectral_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P3_05_spectral_analysis.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    df_ts = load_data()

    # Stationarity tests
    stationarity_results = test_stationarity(df_ts)

    # Visual stationarity check
    plot_rolling_stationarity(df_ts)

    # Decomposition
    perform_decomposition(df_ts)

    # ACF and PACF
    plot_acf_pacf(df_ts)

    # Differencing analysis
    diff_results = differencing_analysis(df_ts)

    # Spectral analysis
    spectral_analysis(df_ts)

    print(f"\n{'='*70}")
    print(f"✅ PHASE 3 COMPLETE — Stationarity & Decomposition analysis done!")
    print(f"{'='*70}")
