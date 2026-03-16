"""
===========================================================================
PHASE 1 — DATA FOUNDATION
===========================================================================
Course Topic: Time Series Data Collection, Structure & Pre-processing

This script covers:
  1. Loading the IMD monthly rainfall dataset (1901–2015)
  2. Filtering Karnataka subdivisions (Coastal, North Interior, South Interior)
  3. Handling missing values using forward-fill + interpolation
  4. Outlier detection using IQR method
  5. Reshaping wide-format monthly data into a long-format time series
  6. Creating a proper DatetimeIndex for time series analysis
  7. Saving the cleaned, analysis-ready dataset
===========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# ─── Configuration ──────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'rainfall_india.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')

KARNATAKA_SUBDIVISIONS = [
    'COASTAL KARNATAKA',
    'NORTH INTERIOR KARNATAKA',
    'SOUTH INTERIOR KARNATAKA'
]

MONTH_COLS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
              'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

MONTH_MAP = {m: i+1 for i, m in enumerate(MONTH_COLS)}

os.makedirs(PLOT_DIR, exist_ok=True)


def load_and_explore(filepath):
    """Load raw dataset and print summary statistics."""
    print("=" * 70)
    print("PHASE 1 — DATA FOUNDATION")
    print("=" * 70)

    df = pd.read_csv(filepath)

    print(f"\n📂 Dataset loaded: {filepath}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Year range: {df['YEAR'].min()} – {df['YEAR'].max()}")
    print(f"   Subdivisions: {df['SUBDIVISION'].nunique()}")

    print(f"\n📊 Column overview:")
    print(f"   {'Column':<20} {'Dtype':<12} {'Missing':<10} {'Sample'}")
    print(f"   {'─'*60}")
    for col in df.columns:
        missing = df[col].isnull().sum()
        sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else 'N/A'
        print(f"   {col:<20} {str(df[col].dtype):<12} {missing:<10} {sample}")

    return df


def filter_karnataka(df):
    """Filter only Karnataka subdivisions."""
    df_karnataka = df[df['SUBDIVISION'].isin(KARNATAKA_SUBDIVISIONS)].copy()
    df_karnataka = df_karnataka.reset_index(drop=True)

    print(f"\n🗺️  Filtered Karnataka data:")
    for sub in KARNATAKA_SUBDIVISIONS:
        count = len(df_karnataka[df_karnataka['SUBDIVISION'] == sub])
        yr_min = df_karnataka[df_karnataka['SUBDIVISION'] == sub]['YEAR'].min()
        yr_max = df_karnataka[df_karnataka['SUBDIVISION'] == sub]['YEAR'].max()
        print(f"   • {sub}: {count} years ({yr_min}–{yr_max})")

    print(f"   Total records: {len(df_karnataka)}")
    return df_karnataka


def handle_missing_values(df):
    """Handle missing values using interpolation and forward-fill."""
    print(f"\n🔧 Missing Value Treatment:")

    # Check missing values per column
    missing = df[MONTH_COLS + ['ANNUAL']].isnull().sum()
    total_missing = missing.sum()
    print(f"   Total missing cells (monthly + annual): {total_missing}")

    if total_missing > 0:
        print(f"   Missing by column:")
        for col in MONTH_COLS + ['ANNUAL']:
            m = df[col].isnull().sum()
            if m > 0:
                print(f"     {col}: {m} missing values")

    # Strategy: For each subdivision, interpolate monthly values linearly
    for sub in KARNATAKA_SUBDIVISIONS:
        mask = df['SUBDIVISION'] == sub
        for col in MONTH_COLS:
            df.loc[mask, col] = df.loc[mask, col].interpolate(method='linear')
            df.loc[mask, col] = df.loc[mask, col].fillna(method='ffill')
            df.loc[mask, col] = df.loc[mask, col].fillna(method='bfill')

    # Recompute ANNUAL from monthly totals
    df['ANNUAL'] = df[MONTH_COLS].sum(axis=1)
    # Recompute seasonal columns
    df['Jan-Feb'] = df[['JAN', 'FEB']].sum(axis=1)
    df['Mar-May'] = df[['MAR', 'APR', 'MAY']].sum(axis=1)
    df['Jun-Sep'] = df[['JUN', 'JUL', 'AUG', 'SEP']].sum(axis=1)
    df['Oct-Dec'] = df[['OCT', 'NOV', 'DEC']].sum(axis=1)

    remaining = df[MONTH_COLS + ['ANNUAL']].isnull().sum().sum()
    print(f"   ✅ After treatment: {remaining} missing values remain")

    return df


def detect_outliers(df):
    """Detect outliers using the IQR method and flag extreme years."""
    print(f"\n🔍 Outlier Detection (IQR Method):")

    outlier_records = []
    for sub in KARNATAKA_SUBDIVISIONS:
        sub_data = df[df['SUBDIVISION'] == sub]['ANNUAL']
        Q1 = sub_data.quantile(0.25)
        Q3 = sub_data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df['SUBDIVISION'] == sub) &
                      ((df['ANNUAL'] < lower) | (df['ANNUAL'] > upper))]

        print(f"\n   {sub}:")
        print(f"     Q1={Q1:.1f}, Q3={Q3:.1f}, IQR={IQR:.1f}")
        print(f"     Bounds: [{lower:.1f}, {upper:.1f}]")
        print(f"     Outlier years ({len(outliers)}):", end=" ")
        if len(outliers) > 0:
            for _, row in outliers.iterrows():
                print(f"{int(row['YEAR'])}({row['ANNUAL']:.0f}mm)", end="  ")
                outlier_records.append({
                    'SUBDIVISION': sub,
                    'YEAR': int(row['YEAR']),
                    'ANNUAL': row['ANNUAL'],
                    'Type': 'Drought' if row['ANNUAL'] < lower else 'Flood'
                })
            print()
        else:
            print("None detected")

    # Flag outliers in dataframe but do NOT remove them (they are real climate events)
    df['IS_OUTLIER'] = False
    for rec in outlier_records:
        mask = (df['SUBDIVISION'] == rec['SUBDIVISION']) & (df['YEAR'] == rec['YEAR'])
        df.loc[mask, 'IS_OUTLIER'] = True

    print(f"\n   ⚠️  Note: Outliers are FLAGGED, not removed (they represent real")
    print(f"      extreme climate events like droughts and floods)")

    return df, outlier_records


def reshape_to_timeseries(df):
    """
    Reshape wide-format (one row per year, 12 month columns)
    into long-format (one row per month) with proper DatetimeIndex.
    """
    print(f"\n📐 Reshaping to Time Series Format:")

    records = []
    for _, row in df.iterrows():
        for month_col in MONTH_COLS:
            month_num = MONTH_MAP[month_col]
            records.append({
                'SUBDIVISION': row['SUBDIVISION'],
                'DATE': pd.Timestamp(year=int(row['YEAR']), month=month_num, day=1),
                'YEAR': int(row['YEAR']),
                'MONTH': month_num,
                'MONTH_NAME': month_col,
                'RAINFALL_MM': row[month_col],
                'IS_OUTLIER_YEAR': row['IS_OUTLIER']
            })

    df_ts = pd.DataFrame(records)
    df_ts = df_ts.sort_values(['SUBDIVISION', 'DATE']).reset_index(drop=True)
    df_ts.set_index('DATE', inplace=True)

    print(f"   Original shape: {df.shape} (wide format)")
    print(f"   Reshaped to: {df_ts.shape} (long/time series format)")
    print(f"   Date range: {df_ts.index.min().strftime('%B %Y')} → {df_ts.index.max().strftime('%B %Y')}")
    print(f"   Frequency: Monthly")

    # Summary stats per subdivision
    print(f"\n   📈 Summary Statistics (Rainfall in mm):")
    for sub in KARNATAKA_SUBDIVISIONS:
        sub_data = df_ts[df_ts['SUBDIVISION'] == sub]['RAINFALL_MM']
        print(f"\n   {sub}:")
        print(f"     Mean: {sub_data.mean():.1f} mm | Median: {sub_data.median():.1f} mm")
        print(f"     Std:  {sub_data.std():.1f} mm | CV: {(sub_data.std()/sub_data.mean()*100):.1f}%")
        print(f"     Min:  {sub_data.min():.1f} mm | Max: {sub_data.max():.1f} mm")

    return df_ts


def plot_data_overview(df_ts, df_wide):
    """Generate initial data overview plots."""
    print(f"\n📊 Generating data overview plots...")

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle('Monthly Rainfall Time Series — Karnataka Subdivisions (1901–2015)',
                 fontsize=16, fontweight='bold', y=0.98)

    colors = {'COASTAL KARNATAKA': '#1e88e5',
              'NORTH INTERIOR KARNATAKA': '#e53935',
              'SOUTH INTERIOR KARNATAKA': '#43a047'}

    for idx, sub in enumerate(KARNATAKA_SUBDIVISIONS):
        sub_data = df_ts[df_ts['SUBDIVISION'] == sub]['RAINFALL_MM']
        axes[idx].plot(sub_data.index, sub_data.values,
                       linewidth=0.5, alpha=0.7, color=colors[sub])
        # Add 12-month rolling mean
        rolling = sub_data.rolling(window=12, center=True).mean()
        axes[idx].plot(rolling.index, rolling.values,
                       linewidth=2, color='black', alpha=0.8, label='12-month rolling mean')
        axes[idx].set_title(sub, fontsize=13, fontweight='bold', color=colors[sub])
        axes[idx].set_ylabel('Rainfall (mm)', fontsize=11)
        axes[idx].legend(loc='upper right')
        axes[idx].grid(True, alpha=0.3)

    axes[2].set_xlabel('Year', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(PLOT_DIR, 'P1_01_raw_timeseries.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P1_01_raw_timeseries.png")

    # ─── Missing Values Heatmap ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    missing_matrix = df_wide.pivot_table(index='SUBDIVISION', columns='YEAR',
                                          values='ANNUAL', aggfunc='count')
    sns.heatmap(missing_matrix.isnull().astype(int), cmap='RdYlGn_r',
                cbar_kws={'label': 'Missing (1=Yes)'}, ax=ax,
                xticklabels=10, yticklabels=True)
    ax.set_title('Data Completeness Heatmap — Karnataka Subdivisions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'P1_02_missing_values_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P1_02_missing_values_heatmap.png")

    # ─── Annual Rainfall Comparison ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 6))
    for sub in KARNATAKA_SUBDIVISIONS:
        sub_data = df_wide[df_wide['SUBDIVISION'] == sub]
        ax.plot(sub_data['YEAR'], sub_data['ANNUAL'],
                linewidth=1.5, label=sub, color=colors[sub], alpha=0.85)
    ax.set_title('Annual Rainfall — Karnataka Subdivisions (1901–2015)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Annual Rainfall (mm)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'P1_03_annual_rainfall_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P1_03_annual_rainfall_comparison.png")

    # ─── Outlier Box Plots ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, sub in enumerate(KARNATAKA_SUBDIVISIONS):
        sub_data = df_wide[df_wide['SUBDIVISION'] == sub]['ANNUAL']
        bp = axes[idx].boxplot(sub_data.dropna(), patch_artist=True,
                               boxprops=dict(facecolor=colors[sub], alpha=0.6),
                               medianprops=dict(color='black', linewidth=2))
        axes[idx].set_title(sub.replace('KARNATAKA', 'KA'), fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('Annual Rainfall (mm)')
    fig.suptitle('Outlier Detection — Annual Rainfall Box Plots', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'P1_04_outlier_boxplots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P1_04_outlier_boxplots.png")


def save_cleaned_data(df_ts, df_wide):
    """Save cleaned datasets for subsequent phases."""
    ts_path = os.path.join(os.path.dirname(__file__), 'data', 'karnataka_monthly_ts.csv')
    wide_path = os.path.join(os.path.dirname(__file__), 'data', 'karnataka_wide.csv')

    df_ts.to_csv(ts_path)
    df_wide.to_csv(wide_path, index=False)

    print(f"\n💾 Cleaned Data Saved:")
    print(f"   • Time series (long format): {ts_path}")
    print(f"   • Wide format: {wide_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # Step 1: Load and explore raw data
    df_raw = load_and_explore(DATA_PATH)

    # Step 2: Filter Karnataka
    df_karnataka = filter_karnataka(df_raw)

    # Step 3: Handle missing values
    df_karnataka = handle_missing_values(df_karnataka)

    # Step 4: Detect outliers
    df_karnataka, outlier_records = detect_outliers(df_karnataka)

    # Step 5: Reshape to time series format
    df_ts = reshape_to_timeseries(df_karnataka)

    # Step 6: Generate overview plots
    plot_data_overview(df_ts, df_karnataka)

    # Step 7: Save cleaned data
    save_cleaned_data(df_ts, df_karnataka)

    print(f"\n{'='*70}")
    print(f"✅ PHASE 1 COMPLETE — Data foundation ready for analysis!")
    print(f"{'='*70}")
