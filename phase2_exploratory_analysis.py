"""
===========================================================================
PHASE 2 — DESCRIPTIVE & EXPLORATORY ANALYSIS
===========================================================================
Course Topic: Visual and Statistical Exploration of Time Series

This script covers:
  1. Raw monthly rainfall series plots with trend overlays
  2. Seasonal pattern analysis — monsoon dominance (Jun–Sep)
  3. Box plots by month revealing seasonality
  4. Annual totals with trend lines
  5. Extreme year identification (droughts & floods)
  6. Heatmap of monthly rainfall across years
  7. Comparative analysis: Coastal vs Interior Karnataka
  8. Seasonal contribution pie charts
  9. Decade-wise analysis
===========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ─── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
PLOT_DIR = os.path.join(BASE_DIR, 'outputs', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

COLORS = {
    'COASTAL KARNATAKA': '#1565C0',
    'NORTH INTERIOR KARNATAKA': '#C62828',
    'SOUTH INTERIOR KARNATAKA': '#2E7D32'
}

COLORS_LIGHT = {
    'COASTAL KARNATAKA': '#90CAF9',
    'NORTH INTERIOR KARNATAKA': '#EF9A9A',
    'SOUTH INTERIOR KARNATAKA': '#A5D6A7'
}


def load_data():
    """Load cleaned datasets from Phase 1."""
    print("=" * 70)
    print("PHASE 2 — DESCRIPTIVE & EXPLORATORY ANALYSIS")
    print("=" * 70)

    df_ts = pd.read_csv(os.path.join(DATA_DIR, 'karnataka_monthly_ts.csv'),
                        parse_dates=['DATE'], index_col='DATE')
    df_wide = pd.read_csv(os.path.join(DATA_DIR, 'karnataka_wide.csv'))

    print(f"\n📂 Loaded time series: {df_ts.shape}")
    print(f"📂 Loaded wide format: {df_wide.shape}")
    return df_ts, df_wide


def plot_seasonal_boxplots(df_ts):
    """Box plots by month — reveals seasonality pattern."""
    print(f"\n📊 1. Monthly Rainfall Box Plots (Seasonality)...")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
    fig.suptitle('Monthly Rainfall Distribution — Seasonality Analysis',
                 fontsize=16, fontweight='bold')

    for idx, sub in enumerate(COLORS.keys()):
        sub_data = df_ts[df_ts['SUBDIVISION'] == sub].copy()
        month_data = [sub_data[sub_data['MONTH'] == m]['RAINFALL_MM'].values for m in range(1, 13)]

        bp = axes[idx].boxplot(month_data, labels=MONTHS, patch_artist=True,
                               medianprops=dict(color='black', linewidth=2),
                               flierprops=dict(marker='o', markersize=3, alpha=0.5))

        # Color monsoon months differently
        for j, patch in enumerate(bp['boxes']):
            if j >= 5 and j <= 8:  # Jun-Sep (indices 5-8)
                patch.set_facecolor(COLORS[sub])
                patch.set_alpha(0.7)
            else:
                patch.set_facecolor(COLORS_LIGHT[sub])
                patch.set_alpha(0.5)

        short_name = sub.replace(' KARNATAKA', '').replace(' ', '\n')
        axes[idx].set_title(f'{short_name}\nKARNATAKA', fontsize=12, fontweight='bold',
                           color=COLORS[sub])
        axes[idx].set_xlabel('Month', fontsize=10)
        axes[idx].set_ylabel('Rainfall (mm)', fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)

        # Highlight monsoon zone
        axes[idx].axvspan(5.5, 9.5, alpha=0.08, color='blue', label='Monsoon (Jun-Sep)')
        axes[idx].legend(fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'P2_01_seasonal_boxplots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P2_01_seasonal_boxplots.png")


def plot_mean_monthly_profile(df_ts):
    """Average climate profile — mean monthly rainfall."""
    print(f"📊 2. Mean Monthly Rainfall Profile...")

    fig, ax = plt.subplots(figsize=(12, 6))

    for sub in COLORS.keys():
        sub_data = df_ts[df_ts['SUBDIVISION'] == sub]
        monthly_mean = sub_data.groupby('MONTH')['RAINFALL_MM'].mean()
        monthly_std = sub_data.groupby('MONTH')['RAINFALL_MM'].std()

        ax.plot(range(1, 13), monthly_mean.values, 'o-', linewidth=2.5,
                markersize=8, label=sub, color=COLORS[sub])
        ax.fill_between(range(1, 13),
                        (monthly_mean - monthly_std).values,
                        (monthly_mean + monthly_std).values,
                        alpha=0.15, color=COLORS[sub])

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTHS, fontsize=10)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Mean Rainfall (mm)', fontsize=12)
    ax.set_title('Mean Monthly Rainfall Profile — Karnataka Subdivisions\n(Shaded area = ±1 Standard Deviation)',
                 fontsize=14, fontweight='bold')
    ax.axvspan(5.5, 9.5, alpha=0.08, color='blue')
    ax.text(7.5, ax.get_ylim()[1] * 0.95, 'MONSOON', fontsize=12,
            ha='center', color='navy', fontweight='bold', alpha=0.6)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'P2_02_mean_monthly_profile.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P2_02_mean_monthly_profile.png")


def plot_annual_trends(df_wide):
    """Annual rainfall trends with linear regression line."""
    print(f"📊 3. Annual Rainfall Trends with Trend Lines...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle('Annual Rainfall Trends — Karnataka Subdivisions (1901–2015)\nwith Linear Trend Lines',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, sub in enumerate(COLORS.keys()):
        sub_data = df_wide[df_wide['SUBDIVISION'] == sub].sort_values('YEAR')
        years = sub_data['YEAR'].values
        annual = sub_data['ANNUAL'].values

        # Bar plot
        mean_rainfall = np.nanmean(annual)
        bar_colors = [COLORS[sub] if a >= mean_rainfall else COLORS_LIGHT[sub]
                      for a in annual]
        axes[idx].bar(years, annual, color=bar_colors, alpha=0.7, width=0.8)

        # Linear trend line
        valid_mask = ~np.isnan(annual)
        z = np.polyfit(years[valid_mask], annual[valid_mask], 1)
        p = np.poly1d(z)
        axes[idx].plot(years, p(years), '--', linewidth=2.5, color='black',
                       label=f'Trend: {z[0]:+.2f} mm/year')

        # Mean line
        axes[idx].axhline(y=mean_rainfall, color='orange', linestyle=':',
                          linewidth=2, alpha=0.8, label=f'Mean: {mean_rainfall:.0f} mm')

        # Moving average
        ma = pd.Series(annual).rolling(window=10, center=True).mean()
        axes[idx].plot(years, ma.values, linewidth=2.5, color='darkred',
                       alpha=0.8, label='10-year moving avg')

        short_name = sub.replace(' KARNATAKA', '')
        axes[idx].set_title(f'{short_name} KARNATAKA', fontsize=13,
                           fontweight='bold', color=COLORS[sub])
        axes[idx].set_ylabel('Annual Rainfall (mm)', fontsize=11)
        axes[idx].legend(fontsize=9, loc='upper right')

    axes[2].set_xlabel('Year', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(PLOT_DIR, 'P2_03_annual_trends.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P2_03_annual_trends.png")


def plot_rainfall_heatmap(df_wide):
    """Heatmap of monthly rainfall across years — visual fingerprint."""
    print(f"📊 4. Monthly Rainfall Heatmaps...")

    fig, axes = plt.subplots(3, 1, figsize=(18, 14))
    fig.suptitle('Monthly Rainfall Heatmap — Karnataka (1901–2015)',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, sub in enumerate(COLORS.keys()):
        sub_data = df_wide[df_wide['SUBDIVISION'] == sub].sort_values('YEAR')
        heatmap_data = sub_data[MONTHS].values
        years = sub_data['YEAR'].values

        im = axes[idx].imshow(heatmap_data.T, aspect='auto', cmap='YlGnBu',
                              interpolation='nearest')

        # Set tick labels
        year_ticks = np.arange(0, len(years), 10)
        axes[idx].set_xticks(year_ticks)
        axes[idx].set_xticklabels(years[year_ticks], fontsize=8, rotation=45)
        axes[idx].set_yticks(range(12))
        axes[idx].set_yticklabels(MONTHS, fontsize=9)

        short_name = sub.replace(' KARNATAKA', '')
        axes[idx].set_title(f'{short_name} KARNATAKA', fontsize=12,
                           fontweight='bold', color=COLORS[sub])

        cbar = plt.colorbar(im, ax=axes[idx], pad=0.02)
        cbar.set_label('Rainfall (mm)', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, 'P2_04_rainfall_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P2_04_rainfall_heatmap.png")


def identify_extreme_years(df_wide):
    """Identify and visualize extreme rainfall years (droughts & floods)."""
    print(f"\n📊 5. Extreme Year Identification...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Extreme Rainfall Years — Drought & Flood Events',
                 fontsize=16, fontweight='bold')

    for idx, sub in enumerate(COLORS.keys()):
        sub_data = df_wide[df_wide['SUBDIVISION'] == sub].sort_values('YEAR')
        annual = sub_data['ANNUAL'].values
        years = sub_data['YEAR'].values
        mean_val = np.nanmean(annual)
        std_val = np.nanstd(annual)

        # Anomaly = departure from mean
        anomaly = annual - mean_val

        # Color by anomaly
        pos_colors = [COLORS[sub] if a > 0 else COLORS_LIGHT[sub] for a in anomaly]
        axes[idx].bar(years, anomaly, color=pos_colors, alpha=0.7)

        # Mark extreme events (>2σ)
        extreme_mask = np.abs(anomaly) > 2 * std_val
        for yr, anom, is_ext in zip(years, anomaly, extreme_mask):
            if is_ext:
                label = f'{int(yr)}'
                axes[idx].annotate(label, (yr, anom), fontsize=7,
                                  ha='center', va='bottom' if anom > 0 else 'top',
                                  fontweight='bold', color='red')

        axes[idx].axhline(y=0, color='black', linewidth=1)
        axes[idx].axhline(y=2*std_val, color='red', linestyle='--', alpha=0.5,
                          label=f'±2σ ({2*std_val:.0f} mm)')
        axes[idx].axhline(y=-2*std_val, color='red', linestyle='--', alpha=0.5)

        short_name = sub.replace(' KARNATAKA', '')
        axes[idx].set_title(f'{short_name} KA', fontsize=12,
                           fontweight='bold', color=COLORS[sub])
        axes[idx].set_xlabel('Year', fontsize=10)
        axes[idx].set_ylabel('Rainfall Anomaly (mm)', fontsize=10)
        axes[idx].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'P2_05_extreme_years.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P2_05_extreme_years.png")

    # Print extreme events
    print(f"\n   🌧️  Notable Extreme Events:")
    for sub in COLORS.keys():
        sub_data = df_wide[df_wide['SUBDIVISION'] == sub]
        mean_val = sub_data['ANNUAL'].mean()
        std_val = sub_data['ANNUAL'].std()
        droughts = sub_data[sub_data['ANNUAL'] < mean_val - 2*std_val][['YEAR', 'ANNUAL']]
        floods = sub_data[sub_data['ANNUAL'] > mean_val + 2*std_val][['YEAR', 'ANNUAL']]
        short = sub.replace(' KARNATAKA', '')
        print(f"\n   {short} KARNATAKA:")
        if len(droughts) > 0:
            print(f"     Severe Droughts: {', '.join([f'{int(r.YEAR)}({r.ANNUAL:.0f}mm)' for _, r in droughts.iterrows()])}")
        if len(floods) > 0:
            print(f"     Extreme Floods:  {', '.join([f'{int(r.YEAR)}({r.ANNUAL:.0f}mm)' for _, r in floods.iterrows()])}")


def plot_seasonal_contribution(df_wide):
    """Pie charts showing seasonal contribution to annual rainfall."""
    print(f"\n📊 6. Seasonal Contribution Analysis...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Seasonal Rainfall Contribution — Karnataka Subdivisions',
                 fontsize=16, fontweight='bold')

    season_labels = ['Winter\n(Jan-Feb)', 'Pre-Monsoon\n(Mar-May)',
                     'Monsoon\n(Jun-Sep)', 'Post-Monsoon\n(Oct-Dec)']
    season_colors = ['#90CAF9', '#FFF176', '#1565C0', '#FF8A65']

    for idx, sub in enumerate(COLORS.keys()):
        sub_data = df_wide[df_wide['SUBDIVISION'] == sub]
        seasonal_means = [
            sub_data['Jan-Feb'].mean(),
            sub_data['Mar-May'].mean(),
            sub_data['Jun-Sep'].mean(),
            sub_data['Oct-Dec'].mean()
        ]

        wedges, texts, autotexts = axes[idx].pie(
            seasonal_means, labels=season_labels, autopct='%1.1f%%',
            colors=season_colors, startangle=90,
            textprops={'fontsize': 9},
            pctdistance=0.75,
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2))

        for autotext in autotexts:
            autotext.set_fontweight('bold')

        short_name = sub.replace(' KARNATAKA', '')
        axes[idx].set_title(f'{short_name} KA', fontsize=12,
                           fontweight='bold', color=COLORS[sub], pad=15)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'P2_06_seasonal_contribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P2_06_seasonal_contribution.png")


def plot_decade_analysis(df_wide):
    """Decade-wise rainfall comparison."""
    print(f"📊 7. Decade-wise Analysis...")

    fig, ax = plt.subplots(figsize=(14, 7))

    df_wide['DECADE'] = (df_wide['YEAR'] // 10) * 10
    decades = sorted(df_wide['DECADE'].unique())

    x = np.arange(len(decades))
    width = 0.25

    for i, sub in enumerate(COLORS.keys()):
        sub_data = df_wide[df_wide['SUBDIVISION'] == sub]
        decade_means = sub_data.groupby('DECADE')['ANNUAL'].mean()
        decade_means = decade_means.reindex(decades, fill_value=0)

        short = sub.replace(' KARNATAKA', ' KA')
        ax.bar(x + i*width, decade_means.values, width, label=short,
               color=COLORS[sub], alpha=0.8, edgecolor='white')

    ax.set_xticks(x + width)
    ax.set_xticklabels([f'{d}s' for d in decades], fontsize=10, rotation=45)
    ax.set_xlabel('Decade', fontsize=12)
    ax.set_ylabel('Mean Annual Rainfall (mm)', fontsize=12)
    ax.set_title('Decade-wise Mean Annual Rainfall — Karnataka Subdivisions',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'P2_07_decade_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P2_07_decade_analysis.png")


def plot_correlation_matrix(df_wide):
    """Correlation between monthly rainfall values."""
    print(f"📊 8. Inter-month Correlation Analysis...")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Inter-Month Rainfall Correlation — Karnataka Subdivisions',
                 fontsize=16, fontweight='bold')

    for idx, sub in enumerate(COLORS.keys()):
        sub_data = df_wide[df_wide['SUBDIVISION'] == sub][MONTHS]
        corr = sub_data.corr()

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, ax=axes[idx],
                    square=True, linewidths=0.5,
                    annot_kws={'size': 7})

        short_name = sub.replace(' KARNATAKA', '')
        axes[idx].set_title(f'{short_name} KA', fontsize=12,
                           fontweight='bold', color=COLORS[sub])

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'P2_08_month_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P2_08_month_correlation.png")


def print_descriptive_statistics(df_ts, df_wide):
    """Print comprehensive descriptive statistics."""
    print(f"\n{'═'*70}")
    print(f"  DESCRIPTIVE STATISTICS SUMMARY")
    print(f"{'═'*70}")

    for sub in COLORS.keys():
        sub_ts = df_ts[df_ts['SUBDIVISION'] == sub]['RAINFALL_MM']
        sub_wide = df_wide[df_wide['SUBDIVISION'] == sub]

        print(f"\n  📍 {sub}")
        print(f"     {'─'*50}")
        print(f"     Monthly rainfall (mm):")
        print(f"       Mean:     {sub_ts.mean():.2f}")
        print(f"       Median:   {sub_ts.median():.2f}")
        print(f"       Std Dev:  {sub_ts.std():.2f}")
        print(f"       Skewness: {sub_ts.skew():.3f}")
        print(f"       Kurtosis: {sub_ts.kurtosis():.3f}")
        print(f"       CV (%):   {(sub_ts.std()/sub_ts.mean()*100):.1f}")

        annual = sub_wide['ANNUAL']
        print(f"     Annual rainfall (mm):")
        print(f"       Mean:     {annual.mean():.2f}")
        print(f"       Std Dev:  {annual.std():.2f}")
        print(f"       Min:      {annual.min():.2f} (Year: {int(sub_wide.loc[annual.idxmin(), 'YEAR'])})")
        print(f"       Max:      {annual.max():.2f} (Year: {int(sub_wide.loc[annual.idxmax(), 'YEAR'])})")

        # Monsoon contribution
        monsoon_pct = (sub_wide['Jun-Sep'].mean() / sub_wide['ANNUAL'].mean()) * 100
        print(f"     Monsoon contribution (Jun-Sep): {monsoon_pct:.1f}% of annual rainfall")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    df_ts, df_wide = load_data()

    plot_seasonal_boxplots(df_ts)
    plot_mean_monthly_profile(df_ts)
    plot_annual_trends(df_wide)
    plot_rainfall_heatmap(df_wide)
    identify_extreme_years(df_wide)
    plot_seasonal_contribution(df_wide)
    plot_decade_analysis(df_wide)
    plot_correlation_matrix(df_wide)
    print_descriptive_statistics(df_ts, df_wide)

    print(f"\n{'='*70}")
    print(f"✅ PHASE 2 COMPLETE — Exploratory analysis done!")
    print(f"{'='*70}")
