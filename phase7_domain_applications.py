"""
===========================================================================
PHASE 7 — DOMAIN APPLICATIONS & SOCIO-ECONOMIC ANALYSIS
===========================================================================
Course Topic: Real-world Applications of Time Series Forecasting

This script covers:
  1. Kharif crop planning (Rice, Ragi sowing decisions)
  2. Reservoir inflow estimation (Cauvery & Tungabhadra basins)
  3. Drought probability estimation
  4. Coastal vs Interior Karnataka comparison
  5. Climate change trend analysis
  6. Agricultural advisory generation
  7. Policy recommendations
===========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ─── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOT_DIR = os.path.join(BASE_DIR, 'outputs', 'plots')
REPORT_DIR = os.path.join(BASE_DIR, 'outputs', 'reports')
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

SUBDIVISIONS = ['COASTAL KARNATAKA', 'NORTH INTERIOR KARNATAKA', 'SOUTH INTERIOR KARNATAKA']
COLORS = {'COASTAL KARNATAKA': '#1565C0',
          'NORTH INTERIOR KARNATAKA': '#C62828',
          'SOUTH INTERIOR KARNATAKA': '#2E7D32'}

# Kharif crop rainfall requirements (mm during Jun-Sep)
CROP_REQUIREMENTS = {
    'Rice (Paddy)': {'min': 800, 'optimal': 1200, 'max': 1600,
                     'regions': ['COASTAL KARNATAKA', 'SOUTH INTERIOR KARNATAKA']},
    'Ragi (Finger Millet)': {'min': 350, 'optimal': 600, 'max': 900,
                              'regions': ['SOUTH INTERIOR KARNATAKA', 'NORTH INTERIOR KARNATAKA']},
    'Jowar (Sorghum)': {'min': 250, 'optimal': 450, 'max': 700,
                         'regions': ['NORTH INTERIOR KARNATAKA']},
    'Maize': {'min': 400, 'optimal': 650, 'max': 1000,
              'regions': ['NORTH INTERIOR KARNATAKA', 'SOUTH INTERIOR KARNATAKA']},
    'Groundnut': {'min': 300, 'optimal': 500, 'max': 800,
                  'regions': ['NORTH INTERIOR KARNATAKA']},
    'Cotton': {'min': 400, 'optimal': 700, 'max': 1100,
               'regions': ['NORTH INTERIOR KARNATAKA']}
}

# Reservoir data (approximate)
RESERVOIRS = {
    'KRS (Krishna Raja Sagara)': {
        'basin': 'Cauvery', 'region': 'SOUTH INTERIOR KARNATAKA',
        'capacity_tmc': 49.45, 'catchment_area_sq_km': 10619
    },
    'Tungabhadra Dam': {
        'basin': 'Tungabhadra', 'region': 'NORTH INTERIOR KARNATAKA',
        'capacity_tmc': 100.86, 'catchment_area_sq_km': 28180
    },
    'Linganamakki Dam': {
        'basin': 'Sharavathi', 'region': 'COASTAL KARNATAKA',
        'capacity_tmc': 151.75, 'catchment_area_sq_km': 1991
    }
}


def load_data():
    """Load all necessary data."""
    print("=" * 70)
    print("PHASE 7 — DOMAIN APPLICATIONS")
    print("=" * 70)

    df_ts = pd.read_csv(os.path.join(DATA_DIR, 'karnataka_monthly_ts.csv'),
                        parse_dates=['DATE'], index_col='DATE')
    df_wide = pd.read_csv(os.path.join(DATA_DIR, 'karnataka_wide.csv'))

    # Load forecasts
    forecasts = {}
    for sub in SUBDIVISIONS:
        short = sub.replace(' KARNATAKA', '').replace(' ', '_')
        fpath = os.path.join(DATA_DIR, f'forecast_{short}.csv')
        if os.path.exists(fpath):
            forecasts[sub] = pd.read_csv(fpath, index_col=0, parse_dates=True)

    print(f"📂 Data loaded. Forecasts available: {len(forecasts)}")
    return df_ts, df_wide, forecasts


def kharif_crop_analysis(df_wide, forecasts):
    """Analyze rainfall suitability for Kharif crops."""
    print(f"\n{'═'*70}")
    print(f"  KHARIF CROP PLANNING ANALYSIS")
    print(f"{'═'*70}")

    # Historical monsoon rainfall
    monsoon_data = {}
    for sub in SUBDIVISIONS:
        sub_data = df_wide[df_wide['SUBDIVISION'] == sub]
        monsoon_data[sub] = sub_data['Jun-Sep'].dropna()

    # Plot crop suitability
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.suptitle('Kharif Crop Rainfall Suitability — Karnataka Subdivisions',
                 fontsize=16, fontweight='bold', y=1.02)

    for idx, sub in enumerate(SUBDIVISIONS):
        ax = axes[idx]
        monsoon = monsoon_data[sub]

        # Histogram of historical monsoon rainfall
        ax.hist(monsoon.values, bins=25, density=True, alpha=0.4,
                color=COLORS[sub], edgecolor='white', label='Historical')

        # Normal fit
        mu, sigma = monsoon.mean(), monsoon.std()
        x_range = np.linspace(monsoon.min() - 100, monsoon.max() + 100, 200)
        ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), '--',
                linewidth=2, color='black', label=f'Normal fit\n(μ={mu:.0f}, σ={sigma:.0f})')

        # Crop requirement zones
        relevant_crops = {k: v for k, v in CROP_REQUIREMENTS.items()
                          if sub in v['regions']}

        y_max = ax.get_ylim()[1]
        for crop_name, req in relevant_crops.items():
            ax.axvline(x=req['min'], color='red', linestyle=':', alpha=0.4)
            ax.axvline(x=req['optimal'], color='green', linestyle=':', alpha=0.4)

        # Forecast monsoon total
        if sub in forecasts:
            fc = forecasts[sub]
            monsoon_fc_months = fc[fc.index.month.isin([6, 7, 8, 9])]
            if len(monsoon_fc_months) > 0:
                forecast_monsoon = monsoon_fc_months['FORECAST'].sum()
                ax.axvline(x=forecast_monsoon, color='purple', linewidth=3,
                           label=f'Forecast: {forecast_monsoon:.0f} mm')

        short = sub.replace(' KARNATAKA', '')
        ax.set_title(f'{short} KA', fontsize=12, fontweight='bold', color=COLORS[sub])
        ax.set_xlabel('Monsoon Rainfall (mm)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'P7_01_crop_suitability.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P7_01_crop_suitability.png")

    # Crop advisory table
    print(f"\n  📋 Crop Advisory Based on Forecasts:")
    print(f"     {'Crop':<25} {'Region':<25} {'Forecast':>10} {'Status'}")
    print(f"     {'─'*75}")

    for crop_name, req in CROP_REQUIREMENTS.items():
        for region in req['regions']:
            if region in forecasts:
                fc = forecasts[region]
                monsoon_fc_months = fc[fc.index.month.isin([6, 7, 8, 9])]
                if len(monsoon_fc_months) > 0:
                    forecast_total = monsoon_fc_months['FORECAST'].sum()
                    short_region = region.replace(' KARNATAKA', ' KA')

                    if forecast_total < req['min']:
                        status = "⚠️  INSUFFICIENT — Irrigation needed"
                    elif forecast_total > req['max']:
                        status = "🌊 EXCESS — Flood risk"
                    elif abs(forecast_total - req['optimal']) < (req['optimal'] * 0.15):
                        status = "✅ OPTIMAL"
                    else:
                        status = "👍 ADEQUATE"

                    print(f"     {crop_name:<25} {short_region:<25} {forecast_total:>8.0f}mm {status}")


def reservoir_inflow_estimation(df_wide, forecasts):
    """Estimate reservoir inflow based on rainfall forecasts."""
    print(f"\n{'═'*70}")
    print(f"  RESERVOIR INFLOW ESTIMATION")
    print(f"{'═'*70}")

    # Simple runoff coefficient assumption (0.3–0.5 for Karnataka terrain)
    RUNOFF_COEFFICIENTS = {
        'COASTAL KARNATAKA': 0.45,  # Higher (steep Western Ghats)
        'NORTH INTERIOR KARNATAKA': 0.30,  # Drier, flatter
        'SOUTH INTERIOR KARNATAKA': 0.35   # Moderate
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Reservoir Inflow Estimation from Rainfall Forecasts',
                 fontsize=16, fontweight='bold')

    for idx, (reservoir_name, info) in enumerate(RESERVOIRS.items()):
        sub = info['region']
        sub_data = df_wide[df_wide['SUBDIVISION'] == sub]
        short = sub.replace(' KARNATAKA', '')

        # Historical annual inflow estimate (rainfall × catchment × runoff coeff)
        runoff_coeff = RUNOFF_COEFFICIENTS[sub]
        catchment = info['catchment_area_sq_km']
        capacity = info['capacity_tmc']

        # Convert: rainfall(mm) × area(km²) = volume in million m³
        # Then convert to TMC (1 TMC = 28.317 million m³)
        hist_annual = sub_data['ANNUAL'].values
        est_inflow = (hist_annual / 1000) * catchment * runoff_coeff / 28.317  # TMC

        axes[idx].hist(est_inflow, bins=20, alpha=0.6, color=COLORS[sub],
                       edgecolor='white')
        axes[idx].axvline(x=capacity, color='red', linewidth=2.5, linestyle='--',
                          label=f'Dam Capacity: {capacity:.1f} TMC')
        axes[idx].axvline(x=np.mean(est_inflow), color='black', linewidth=2,
                          label=f'Avg Inflow: {np.mean(est_inflow):.1f} TMC')

        # Forecast inflow
        if sub in forecasts:
            fc = forecasts[sub]
            fc_annual = fc['FORECAST'].sum()
            fc_inflow = (fc_annual / 1000) * catchment * runoff_coeff / 28.317
            axes[idx].axvline(x=fc_inflow, color='purple', linewidth=2.5,
                              label=f'Forecast: {fc_inflow:.1f} TMC')

        axes[idx].set_title(f'{reservoir_name}\n({info["basin"]} Basin)',
                           fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Estimated Inflow (TMC)', fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)
        axes[idx].legend(fontsize=8)

        print(f"\n  🏗️  {reservoir_name} ({info['basin']} Basin)")
        print(f"     Region: {short} KARNATAKA")
        print(f"     Capacity: {capacity:.1f} TMC")
        print(f"     Historical avg inflow: {np.mean(est_inflow):.1f} TMC")
        if sub in forecasts:
            print(f"     Forecast inflow: {fc_inflow:.1f} TMC")
            fill_pct = (fc_inflow / capacity) * 100
            print(f"     Expected fill: {fill_pct:.0f}%")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'P7_02_reservoir_inflow.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ Saved: P7_02_reservoir_inflow.png")


def drought_probability(df_wide, forecasts):
    """Estimate drought probability from historical and forecast data."""
    print(f"\n{'═'*70}")
    print(f"  DROUGHT PROBABILITY ESTIMATION")
    print(f"{'═'*70}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Drought Probability Analysis — Karnataka Subdivisions',
                 fontsize=16, fontweight='bold')

    for idx, sub in enumerate(SUBDIVISIONS):
        sub_data = df_wide[df_wide['SUBDIVISION'] == sub]
        annual = sub_data['ANNUAL'].dropna()

        mean_rain = annual.mean()
        std_rain = annual.std()
        short = sub.replace(' KARNATAKA', '')

        # Define drought thresholds
        # Moderate drought: < mean - 1σ
        # Severe drought: < mean - 2σ
        moderate_threshold = mean_rain - std_rain
        severe_threshold = mean_rain - 2 * std_rain

        # Historical drought frequency
        moderate_count = (annual < moderate_threshold).sum()
        severe_count = (annual < severe_threshold).sum()
        total_years = len(annual)

        # Probability distribution
        x_range = np.linspace(annual.min() - 200, annual.max() + 200, 300)
        pdf = stats.norm.pdf(x_range, mean_rain, std_rain)

        axes[idx].plot(x_range, pdf, linewidth=2, color='black')

        # Fill drought zones
        axes[idx].fill_between(x_range, pdf, where=(x_range < moderate_threshold),
                               alpha=0.3, color='orange', label=f'Moderate (<{moderate_threshold:.0f}mm)')
        axes[idx].fill_between(x_range, pdf, where=(x_range < severe_threshold),
                               alpha=0.5, color='red', label=f'Severe (<{severe_threshold:.0f}mm)')
        axes[idx].fill_between(x_range, pdf, where=(x_range >= moderate_threshold),
                               alpha=0.2, color='green', label='Normal/Above')

        # Forecast
        if sub in forecasts:
            fc_annual = forecasts[sub]['FORECAST'].sum()
            axes[idx].axvline(x=fc_annual, color='purple', linewidth=2.5,
                              linestyle='--', label=f'Forecast: {fc_annual:.0f}mm')

        axes[idx].set_title(f'{short} KA', fontsize=12, fontweight='bold', color=COLORS[sub])
        axes[idx].set_xlabel('Annual Rainfall (mm)', fontsize=10)
        axes[idx].legend(fontsize=7, loc='upper right')

        print(f"\n  📍 {short} KARNATAKA:")
        print(f"     Historical mean: {mean_rain:.0f} mm (σ = {std_rain:.0f} mm)")
        print(f"     Moderate drought threshold (<1σ): {moderate_threshold:.0f} mm")
        print(f"     Severe drought threshold (<2σ):   {severe_threshold:.0f} mm")
        print(f"     Historical moderate droughts: {moderate_count}/{total_years} "
              f"({moderate_count/total_years*100:.1f}%)")
        print(f"     Historical severe droughts:   {severe_count}/{total_years} "
              f"({severe_count/total_years*100:.1f}%)")

        if sub in forecasts:
            fc_annual = forecasts[sub]['FORECAST'].sum()
            z_score = (fc_annual - mean_rain) / std_rain
            drought_prob = stats.norm.cdf(z_score)
            print(f"     Forecast annual: {fc_annual:.0f} mm (z = {z_score:+.2f})")
            if fc_annual < severe_threshold:
                print(f"     ⚠️  SEVERE DROUGHT RISK")
            elif fc_annual < moderate_threshold:
                print(f"     ⚠️  MODERATE DROUGHT RISK")
            else:
                print(f"     ✅ Normal/above-normal rainfall expected")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'P7_03_drought_probability.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ Saved: P7_03_drought_probability.png")


def coastal_vs_interior_comparison(df_ts, df_wide):
    """Compare Coastal Karnataka vs Interior Karnataka — dramatic contrast."""
    print(f"\n{'═'*70}")
    print(f"  COASTAL vs INTERIOR KARNATAKA — COMPARATIVE ANALYSIS")
    print(f"{'═'*70}")

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Coastal vs Interior Karnataka — Rainfall Contrast',
                 fontsize=16, fontweight='bold', y=0.98)

    coastal = df_wide[df_wide['SUBDIVISION'] == 'COASTAL KARNATAKA']
    north = df_wide[df_wide['SUBDIVISION'] == 'NORTH INTERIOR KARNATAKA']

    # 1. Annual trend comparison
    ax = axes[0, 0]
    ax.plot(coastal['YEAR'], coastal['ANNUAL'], linewidth=1.5,
            color=COLORS['COASTAL KARNATAKA'], alpha=0.7, label='Coastal KA')
    ax.plot(north['YEAR'], north['ANNUAL'], linewidth=1.5,
            color=COLORS['NORTH INTERIOR KARNATAKA'], alpha=0.7, label='North Interior KA')

    # Add trend lines
    z_c = np.polyfit(coastal['YEAR'].values, coastal['ANNUAL'].fillna(coastal['ANNUAL'].mean()).values, 1)
    z_n = np.polyfit(north['YEAR'].values, north['ANNUAL'].fillna(north['ANNUAL'].mean()).values, 1)
    ax.plot(coastal['YEAR'], np.polyval(z_c, coastal['YEAR']), '--',
            color=COLORS['COASTAL KARNATAKA'], linewidth=2,
            label=f'Trend: {z_c[0]:+.1f} mm/yr')
    ax.plot(north['YEAR'], np.polyval(z_n, north['YEAR']), '--',
            color=COLORS['NORTH INTERIOR KARNATAKA'], linewidth=2,
            label=f'Trend: {z_n[0]:+.1f} mm/yr')

    ax.set_title('Annual Rainfall Trends', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annual Rainfall (mm)')
    ax.legend(fontsize=9)

    # 2. Monthly profile overlap
    ax = axes[0, 1]
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
              'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    c_monthly = [coastal[m].mean() for m in months]
    n_monthly = [north[m].mean() for m in months]

    ax.fill_between(range(12), c_monthly, alpha=0.3, color=COLORS['COASTAL KARNATAKA'])
    ax.fill_between(range(12), n_monthly, alpha=0.3, color=COLORS['NORTH INTERIOR KARNATAKA'])
    ax.plot(range(12), c_monthly, 'o-', linewidth=2.5,
            color=COLORS['COASTAL KARNATAKA'], label='Coastal KA')
    ax.plot(range(12), n_monthly, 's-', linewidth=2.5,
            color=COLORS['NORTH INTERIOR KARNATAKA'], label='North Interior KA')

    ax.set_xticks(range(12))
    ax.set_xticklabels(months, fontsize=9)
    ax.set_title('Mean Monthly Rainfall Profile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rainfall (mm)')
    ax.legend(fontsize=10)

    # 3. Coefficient of Variation comparison
    ax = axes[1, 0]
    for sub in SUBDIVISIONS:
        sub_data = df_wide[df_wide['SUBDIVISION'] == sub]
        cvs = [(sub_data[m].std() / sub_data[m].mean() * 100) if sub_data[m].mean() > 0 else 0
               for m in months]
        short = sub.replace(' KARNATAKA', ' KA')
        ax.plot(range(12), cvs, 'o-', linewidth=2, label=short, color=COLORS[sub])

    ax.set_xticks(range(12))
    ax.set_xticklabels(months, fontsize=9)
    ax.set_title('Rainfall Variability (CV%) by Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (%)')
    ax.legend(fontsize=9)

    # 4. Ratio plot
    ax = axes[1, 1]
    ratio = coastal['ANNUAL'].values / np.maximum(north['ANNUAL'].values, 1)
    ax.plot(coastal['YEAR'], ratio, linewidth=1.5, color='purple', alpha=0.7)
    ax.axhline(y=ratio.mean(), color='red', linestyle='--',
               label=f'Mean ratio: {ratio.mean():.2f}')
    ax.set_title('Coastal/Interior Rainfall Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ratio (Coastal ÷ Interior)')
    ax.set_xlabel('Year')
    ax.legend(fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, 'P7_04_coastal_vs_interior.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P7_04_coastal_vs_interior.png")

    # Print comparison stats
    print(f"\n  📊 Contrast Summary:")
    print(f"     {'Metric':<35} {'Coastal KA':>12} {'N.Interior KA':>14} {'Ratio':>8}")
    print(f"     {'─'*70}")
    print(f"     {'Mean Annual Rainfall (mm)':<35} {coastal['ANNUAL'].mean():>12.0f} "
          f"{north['ANNUAL'].mean():>14.0f} {coastal['ANNUAL'].mean()/north['ANNUAL'].mean():>7.1f}x")
    print(f"     {'Peak Month Rainfall (mm)':<35} {max(c_monthly):>12.0f} "
          f"{max(n_monthly):>14.0f} {max(c_monthly)/max(n_monthly):>7.1f}x")
    print(f"     {'Monsoon % of Annual':<35} "
          f"{coastal['Jun-Sep'].mean()/coastal['ANNUAL'].mean()*100:>11.1f}% "
          f"{north['Jun-Sep'].mean()/north['ANNUAL'].mean()*100:>13.1f}%")
    print(f"     {'Post-Monsoon % (Oct-Dec)':<35} "
          f"{coastal['Oct-Dec'].mean()/coastal['ANNUAL'].mean()*100:>11.1f}% "
          f"{north['Oct-Dec'].mean()/north['ANNUAL'].mean()*100:>13.1f}%")


def climate_change_analysis(df_wide):
    """Long-term climate change trend analysis."""
    print(f"\n{'═'*70}")
    print(f"  CLIMATE CHANGE TREND ANALYSIS")
    print(f"{'═'*70}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Climate Change Indicators — Karnataka Rainfall (1901–2015)',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, sub in enumerate(SUBDIVISIONS):
        sub_data = df_wide[df_wide['SUBDIVISION'] == sub].sort_values('YEAR')
        short = sub.replace(' KARNATAKA', '')

        # 30-year moving average
        annual = sub_data.set_index('YEAR')['ANNUAL']
        ma30 = annual.rolling(window=30, center=True).mean()

        axes[0, 0].plot(annual.index, ma30.values, linewidth=2.5,
                        color=COLORS[sub], label=f'{short} KA')

    axes[0, 0].set_title('30-Year Moving Average of Annual Rainfall', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Rainfall (mm)')
    axes[0, 0].legend(fontsize=9)

    # Extreme events frequency (per decade)
    ax = axes[0, 1]
    for sub in SUBDIVISIONS:
        sub_data = df_wide[df_wide['SUBDIVISION'] == sub]
        annual = sub_data['ANNUAL']
        mean_val = annual.mean()
        std_val = annual.std()

        sub_data = sub_data.copy()
        sub_data['DECADE'] = (sub_data['YEAR'] // 10) * 10
        sub_data['IS_EXTREME'] = (sub_data['ANNUAL'] < mean_val - 1.5*std_val) | \
                                  (sub_data['ANNUAL'] > mean_val + 1.5*std_val)

        extreme_by_decade = sub_data.groupby('DECADE')['IS_EXTREME'].sum()
        short = sub.replace(' KARNATAKA', ' KA')
        ax.plot(extreme_by_decade.index, extreme_by_decade.values, 'o-',
                linewidth=2, label=short, color=COLORS[sub])

    ax.set_title('Extreme Rainfall Events per Decade', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Extreme Years')
    ax.legend(fontsize=9)

    # Monsoon contribution trend
    ax = axes[1, 0]
    for sub in SUBDIVISIONS:
        sub_data = df_wide[df_wide['SUBDIVISION'] == sub].sort_values('YEAR')
        monsoon_pct = (sub_data['Jun-Sep'] / sub_data['ANNUAL'] * 100)
        rolling = monsoon_pct.rolling(window=20, center=True).mean()
        short = sub.replace(' KARNATAKA', ' KA')
        ax.plot(sub_data['YEAR'].values, rolling.values, linewidth=2,
                color=COLORS[sub], label=short)

    ax.set_title('Monsoon Share of Annual Rainfall (20-yr avg)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Monsoon Share (%)')
    ax.legend(fontsize=9)

    # Variability trend (CV per decade)
    ax = axes[1, 1]
    for sub in SUBDIVISIONS:
        sub_data = df_wide[df_wide['SUBDIVISION'] == sub].copy()
        sub_data['DECADE'] = (sub_data['YEAR'] // 10) * 10
        cv_by_decade = sub_data.groupby('DECADE')['ANNUAL'].apply(
            lambda x: x.std() / x.mean() * 100 if x.mean() > 0 else 0)
        short = sub.replace(' KARNATAKA', ' KA')
        ax.plot(cv_by_decade.index, cv_by_decade.values, 'o-',
                linewidth=2, label=short, color=COLORS[sub])

    ax.set_title('Rainfall Variability (CV%) per Decade', fontsize=12, fontweight='bold')
    ax.set_ylabel('Coefficient of Variation (%)')
    ax.set_xlabel('Decade')
    ax.legend(fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, 'P7_05_climate_change.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P7_05_climate_change.png")

    # Trend statistics
    print(f"\n  📈 Linear Trend Analysis (annual change):")
    for sub in SUBDIVISIONS:
        sub_data = df_wide[df_wide['SUBDIVISION'] == sub].sort_values('YEAR')
        slope, intercept, r_val, p_val, std_err = stats.linregress(
            sub_data['YEAR'].values, sub_data['ANNUAL'].fillna(sub_data['ANNUAL'].mean()).values)
        short = sub.replace(' KARNATAKA', '')
        significance = "✅ Significant" if p_val < 0.05 else "❌ Not significant"
        print(f"     {short} KA: {slope:+.2f} mm/year (p={p_val:.4f}) {significance}")


def generate_final_report(df_wide, forecasts):
    """Generate comprehensive final report."""
    report_path = os.path.join(REPORT_DIR, 'final_analysis_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("FINAL ANALYSIS REPORT\n")
        f.write("Monthly Rainfall Trend & Forecasting in Karnataka\n")
        f.write("Data: IMD Rainfall Dataset (1901-2015)\n")
        f.write("=" * 70 + "\n\n")

        f.write("PROJECT SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write("This project analyzes 115 years of monthly rainfall data for\n")
        f.write("three Karnataka subdivisions using time series techniques.\n\n")

        f.write("SUBDIVISIONS ANALYZED:\n")
        f.write("  1. Coastal Karnataka (Western Ghats coast)\n")
        f.write("  2. North Interior Karnataka (Deccan plateau)\n")
        f.write("  3. South Interior Karnataka (transition zone)\n\n")

        f.write("MODELS IMPLEMENTED:\n")
        f.write("  1. Simple Moving Average (SMA)\n")
        f.write("  2. Simple Exponential Smoothing (SES)\n")
        f.write("  3. Holt-Winters Triple Exponential Smoothing\n")
        f.write("  4. SARIMA (Seasonal ARIMA) — Primary model\n")
        f.write("  5. Facebook Prophet\n")
        f.write("  6. LSTM Neural Network\n\n")

        f.write("KEY FINDINGS:\n")
        f.write("-" * 40 + "\n")
        for sub in SUBDIVISIONS:
            sub_data = df_wide[df_wide['SUBDIVISION'] == sub]
            short = sub.replace(' KARNATAKA', '')
            f.write(f"\n{short} KARNATAKA:\n")
            f.write(f"  Mean annual rainfall: {sub_data['ANNUAL'].mean():.0f} mm\n")
            f.write(f"  Monsoon contribution: {sub_data['Jun-Sep'].mean()/sub_data['ANNUAL'].mean()*100:.1f}%\n")
            f.write(f"  Driest year: {int(sub_data.loc[sub_data['ANNUAL'].idxmin(), 'YEAR'])} "
                    f"({sub_data['ANNUAL'].min():.0f} mm)\n")
            f.write(f"  Wettest year: {int(sub_data.loc[sub_data['ANNUAL'].idxmax(), 'YEAR'])} "
                    f"({sub_data['ANNUAL'].max():.0f} mm)\n")

        f.write("\n\nDOMAIN APPLICATIONS:\n")
        f.write("-" * 40 + "\n")
        f.write("  1. Kharif crop planning — sowing decision support\n")
        f.write("  2. Reservoir inflow estimation — water management\n")
        f.write("  3. Drought probability assessment — disaster preparedness\n")
        f.write("  4. Coastal vs Interior contrast — regional planning\n")
        f.write("  5. Climate change trend detection — policy support\n")

        f.write("\n\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")

    print(f"\n   📝 Final report saved: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    df_ts, df_wide, forecasts = load_data()

    # Kharif crop analysis
    kharif_crop_analysis(df_wide, forecasts)

    # Reservoir inflow
    reservoir_inflow_estimation(df_wide, forecasts)

    # Drought probability
    drought_probability(df_wide, forecasts)

    # Coastal vs Interior comparison
    coastal_vs_interior_comparison(df_ts, df_wide)

    # Climate change analysis
    climate_change_analysis(df_wide)

    # Final report
    generate_final_report(df_wide, forecasts)

    print(f"\n{'='*70}")
    print(f"✅ PHASE 7 COMPLETE — Domain applications analysis done!")
    print(f"{'='*70}")
