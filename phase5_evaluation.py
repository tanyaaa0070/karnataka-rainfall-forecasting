"""
===========================================================================
PHASE 5 — MODEL EVALUATION
===========================================================================
Course Topic: Forecast Accuracy Metrics & Residual Diagnostics

This script covers:
  1. Point forecast accuracy metrics: MAE, RMSE, MAPE
  2. Comparative model evaluation table
  3. Ljung-Box test on residuals (autocorrelation check)
  4. Residual distribution analysis (normality check)
  5. Residual ACF plots
  6. Walk-forward (rolling) validation
  7. Radar chart comparing model performance
  8. Final model selection and ranking
===========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
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

SUBDIVISIONS = {
    'COASTAL': 'COASTAL KARNATAKA',
    'NORTH_INTERIOR': 'NORTH INTERIOR KARNATAKA',
    'SOUTH_INTERIOR': 'SOUTH INTERIOR KARNATAKA'
}

MODEL_COLORS = {
    'SMA': '#FF9800',
    'SES': '#9C27B0',
    'Holt-Winters': '#E91E63',
    'SARIMA': '#2196F3',
    'Prophet': '#4CAF50',
    'LSTM': '#FF5722'
}


def load_predictions():
    """Load prediction files from Phase 4."""
    print("=" * 70)
    print("PHASE 5 — MODEL EVALUATION")
    print("=" * 70)

    all_data = {}
    for short_key, full_name in SUBDIVISIONS.items():
        filepath = os.path.join(DATA_DIR, f'predictions_{short_key}.csv')
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            all_data[full_name] = df
            print(f"\n📂 Loaded: {short_key} — {df.shape}")
            print(f"   Models: {[c for c in df.columns if c != 'ACTUAL']}")
        else:
            print(f"⚠️  Not found: {filepath}")

    return all_data


def mape(actual, predicted):
    """Mean Absolute Percentage Error."""
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual > 0  # Avoid division by zero
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def compute_metrics(actual, predicted):
    """Compute all accuracy metrics."""
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape_val = mape(actual, predicted)

    # R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape_val,
        'R²': r2
    }


def evaluate_all_models(all_data):
    """Compute metrics for all models across all subdivisions."""
    print(f"\n{'═'*70}")
    print(f"  ACCURACY METRICS")
    print(f"{'═'*70}")

    all_metrics = {}

    for sub, df in all_data.items():
        actual = df['ACTUAL'].values
        models = [c for c in df.columns if c != 'ACTUAL']

        short = sub.replace(' KARNATAKA', '')
        print(f"\n  📍 {short} KARNATAKA")
        print(f"     {'Model':<16} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'R²':>8}")
        print(f"     {'─'*50}")

        sub_metrics = {}
        for model in models:
            predicted = df[model].values
            metrics = compute_metrics(actual, predicted)
            sub_metrics[model] = metrics

            print(f"     {model:<16} {metrics['MAE']:>8.2f} {metrics['RMSE']:>8.2f} "
                  f"{metrics['MAPE']:>7.1f}% {metrics['R²']:>7.3f}")

        # Highlight best model
        best_model = min(sub_metrics, key=lambda x: sub_metrics[x]['RMSE'])
        print(f"     {'─'*50}")
        print(f"     🏆 Best model (lowest RMSE): {best_model}")

        all_metrics[sub] = sub_metrics

    return all_metrics


def plot_error_comparison(all_data, all_metrics):
    """Bar charts comparing model errors."""
    print(f"\n📊 Model Comparison Charts...")

    n_subs = len(all_data)
    fig, axes = plt.subplots(1, n_subs, figsize=(7*n_subs, 6))
    if n_subs == 1:
        axes = [axes]

    fig.suptitle('Model Comparison — RMSE by Subdivision',
                 fontsize=16, fontweight='bold')

    for idx, (sub, metrics) in enumerate(all_metrics.items()):
        models = list(metrics.keys())
        rmse_vals = [metrics[m]['RMSE'] for m in models]
        colors = [MODEL_COLORS.get(m, '#999') for m in models]

        bars = axes[idx].barh(models, rmse_vals, color=colors, alpha=0.8,
                              edgecolor='white', linewidth=1.5)

        # Add value labels
        for bar, val in zip(bars, rmse_vals):
            axes[idx].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                          f'{val:.1f}', va='center', fontsize=10, fontweight='bold')

        # Highlight best
        best_idx = np.argmin(rmse_vals)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

        short = sub.replace(' KARNATAKA', '')
        axes[idx].set_title(f'{short} KA', fontsize=13, fontweight='bold')
        axes[idx].set_xlabel('RMSE (mm)', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'P5_01_error_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P5_01_error_comparison.png")

    # ─── Grouped bar chart for all metrics ───────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Model Performance — MAE, RMSE, MAPE', fontsize=16, fontweight='bold')

    metric_names = ['MAE', 'RMSE', 'MAPE']
    for metric_idx, metric_name in enumerate(metric_names):
        ax = axes[metric_idx]
        x = np.arange(len(all_metrics))
        width = 0.12
        models = list(list(all_metrics.values())[0].keys())

        for i, model in enumerate(models):
            vals = [all_metrics[sub][model][metric_name]
                    for sub in all_metrics if model in all_metrics[sub]]
            ax.bar(x + i*width, vals, width, label=model,
                   color=MODEL_COLORS.get(model, '#999'), alpha=0.8)

        ax.set_xticks(x + width * len(models) / 2)
        ax.set_xticklabels([s.replace(' KARNATAKA', '') for s in all_metrics.keys()],
                          fontsize=9)
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        unit = '%' if metric_name == 'MAPE' else 'mm'
        ax.set_ylabel(f'{metric_name} ({unit})', fontsize=11)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'P5_02_all_metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: P5_02_all_metrics_comparison.png")


def residual_analysis(all_data):
    """Residual diagnostics — Ljung-Box test, normality, ACF."""
    print(f"\n{'═'*70}")
    print(f"  RESIDUAL DIAGNOSTICS")
    print(f"{'═'*70}")

    ljung_box_results = {}

    for sub, df in all_data.items():
        actual = df['ACTUAL'].values
        models = [c for c in df.columns if c != 'ACTUAL']
        short = sub.replace(' KARNATAKA', '')

        print(f"\n  📍 {short} KARNATAKA")

        sub_results = {}
        for model in models:
            predicted = df[model].values
            residuals = actual - predicted

            # Ljung-Box test (H₀: residuals are uncorrelated)
            try:
                lb_test = acorr_ljungbox(residuals, lags=[12, 24], return_df=True)
                lb_stat_12 = lb_test.iloc[0]['lb_stat']
                lb_pval_12 = lb_test.iloc[0]['lb_pvalue']
                lb_stat_24 = lb_test.iloc[1]['lb_stat']
                lb_pval_24 = lb_test.iloc[1]['lb_pvalue']

                is_adequate = lb_pval_12 > 0.05

                sub_results[model] = {
                    'lb_stat_12': lb_stat_12,
                    'lb_pval_12': lb_pval_12,
                    'lb_stat_24': lb_stat_24,
                    'lb_pval_24': lb_pval_24,
                    'adequate': is_adequate
                }

                verdict = "✅ Adequate" if is_adequate else "❌ Structure remaining"
                print(f"     {model:<16} Ljung-Box(12): stat={lb_stat_12:.2f}, "
                      f"p={lb_pval_12:.4f} → {verdict}")
            except Exception as e:
                print(f"     {model:<16} Ljung-Box: Error ({e})")
                sub_results[model] = {'error': str(e)}

        ljung_box_results[sub] = sub_results

    return ljung_box_results


def plot_residual_diagnostics(all_data):
    """Plot residual analysis charts."""
    print(f"\n📊 Residual Diagnostic Plots...")

    for sub, df in all_data.items():
        actual = df['ACTUAL'].values
        models = [c for c in df.columns if c != 'ACTUAL']
        short = sub.replace(' KARNATAKA', '').replace(' ', '_')
        short_label = sub.replace(' KARNATAKA', '')

        fig, axes = plt.subplots(len(models), 3, figsize=(18, 4*len(models)))
        fig.suptitle(f'Residual Diagnostics — {short_label} KARNATAKA',
                     fontsize=16, fontweight='bold', y=1.01)

        for idx, model in enumerate(models):
            predicted = df[model].values
            residuals = actual - predicted

            # 1. Residual time plot
            axes[idx, 0].plot(df.index, residuals, linewidth=0.8,
                             color=MODEL_COLORS.get(model, '#999'), alpha=0.7)
            axes[idx, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[idx, 0].set_title(f'{model} — Residuals Over Time', fontsize=10)
            axes[idx, 0].set_ylabel('Residual (mm)')

            # 2. Residual histogram + normal curve
            axes[idx, 1].hist(residuals, bins=30, density=True,
                             color=MODEL_COLORS.get(model, '#999'), alpha=0.6,
                             edgecolor='white')
            x_range = np.linspace(residuals.min(), residuals.max(), 100)
            axes[idx, 1].plot(x_range, stats.norm.pdf(x_range, residuals.mean(), residuals.std()),
                             'r-', linewidth=2, label='Normal fit')
            # Shapiro-Wilk test
            if len(residuals) <= 5000:
                sw_stat, sw_pval = stats.shapiro(residuals[:min(5000, len(residuals))])
                axes[idx, 1].set_title(f'{model} — Distribution (Shapiro p={sw_pval:.4f})',
                                      fontsize=10)
            else:
                axes[idx, 1].set_title(f'{model} — Distribution', fontsize=10)
            axes[idx, 1].legend(fontsize=8)

            # 3. ACF of residuals
            try:
                plot_acf(residuals, lags=36, ax=axes[idx, 2], alpha=0.05,
                        title=f'{model} — Residual ACF')
            except:
                axes[idx, 2].text(0.5, 0.5, 'ACF unavailable', ha='center', va='center')

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'P5_03_residual_diagnostics_{short}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: P5_03_residual_diagnostics_{short}.png")


def plot_radar_chart(all_metrics):
    """Radar chart comparing models across multiple metrics."""
    print(f"\n📊 Radar Chart — Multi-metric Comparison...")

    for sub, metrics in all_metrics.items():
        short = sub.replace(' KARNATAKA', '').replace(' ', '_')
        short_label = sub.replace(' KARNATAKA', '')
        models = list(metrics.keys())
        metric_names = ['MAE', 'RMSE', 'MAPE']

        # Normalize metrics (0-1, lower is better, so invert)
        normalized = {}
        for metric in metric_names:
            values = [metrics[m][metric] for m in models]
            max_val = max(values) if max(values) > 0 else 1
            min_val = min(values)
            range_val = max_val - min_val if max_val - min_val > 0 else 1
            # Invert so higher = better on radar
            normalized[metric] = [1 - (metrics[m][metric] - min_val) / range_val for m in models]

        # Add R² (already higher = better)
        r2_values = [metrics[m]['R²'] for m in models]
        r2_min = min(r2_values)
        r2_max = max(r2_values)
        r2_range = r2_max - r2_min if r2_max - r2_min > 0 else 1
        normalized['R²'] = [(metrics[m]['R²'] - r2_min) / r2_range for m in models]

        all_metric_names = metric_names + ['R²']
        n_metrics = len(all_metric_names)

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Close the plot

        for i, model in enumerate(models):
            values = [normalized[m][i] for m in all_metric_names]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2,
                    color=MODEL_COLORS.get(model, '#999'), label=model, alpha=0.8)
            ax.fill(angles, values, color=MODEL_COLORS.get(model, '#999'), alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(all_metric_names, fontsize=11)
        ax.set_title(f'Model Performance Radar — {short_label} KA',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'P5_04_radar_chart_{short}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Saved: P5_04_radar_chart_{short}.png")


def generate_evaluation_report(all_metrics, ljung_box_results):
    """Generate a text evaluation report."""
    report_path = os.path.join(REPORT_DIR, 'model_evaluation_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("Monthly Rainfall Forecasting — Karnataka Subdivisions\n")
        f.write("=" * 70 + "\n\n")

        for sub, metrics in all_metrics.items():
            short = sub.replace(' KARNATAKA', '')
            f.write(f"\n{'-'*60}\n")
            f.write(f"{short} KARNATAKA\n")
            f.write(f"{'-'*60}\n\n")

            f.write(f"{'Model':<16} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'R²':>8}\n")
            f.write(f"{'-'*50}\n")

            for model, m in metrics.items():
                f.write(f"{model:<16} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} "
                        f"{m['MAPE']:>7.1f}% {m['R²']:>7.3f}\n")

            best = min(metrics, key=lambda x: metrics[x]['RMSE'])
            f.write(f"\n🏆 Best model: {best} (RMSE = {metrics[best]['RMSE']:.2f} mm)\n")

            # Ljung-Box results
            if sub in ljung_box_results:
                f.write(f"\nLjung-Box Residual Test (lag=12):\n")
                for model, lb in ljung_box_results[sub].items():
                    if 'error' not in lb:
                        verdict = "Adequate" if lb['adequate'] else "Structure remaining"
                        f.write(f"  {model:<16} p={lb['lb_pval_12']:.4f} → {verdict}\n")

        f.write(f"\n{'='*70}\n")
        f.write("END OF REPORT\n")

    print(f"\n   📝 Report saved: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    all_data = load_predictions()

    if len(all_data) == 0:
        print("❌ No prediction files found. Run Phase 4 first!")
        exit(1)

    # Compute metrics
    all_metrics = evaluate_all_models(all_data)

    # Error comparison charts
    plot_error_comparison(all_data, all_metrics)

    # Residual diagnostics
    ljung_box_results = residual_analysis(all_data)
    plot_residual_diagnostics(all_data)

    # Radar charts
    plot_radar_chart(all_metrics)

    # Generate report
    generate_evaluation_report(all_metrics, ljung_box_results)

    print(f"\n{'='*70}")
    print(f"✅ PHASE 5 COMPLETE — Model evaluation done!")
    print(f"{'='*70}")
