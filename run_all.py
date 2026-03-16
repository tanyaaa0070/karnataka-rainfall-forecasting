"""
===========================================================================
MASTER RUNNER — Monthly Rainfall Trend & Forecasting in Karnataka
===========================================================================
Run all 7 phases sequentially. Each phase depends on the previous one.

Usage:
    python run_all.py          → Run all phases
    python run_all.py 1        → Run only Phase 1
    python run_all.py 1 3      → Run Phases 1 through 3
    python run_all.py 4 5      → Run Phases 4 and 5
===========================================================================
"""

import sys
import time
import os
import subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PHASES = {
    1: ('phase1_data_foundation', 'Data Foundation'),
    2: ('phase2_exploratory_analysis', 'Descriptive & Exploratory Analysis'),
    3: ('phase3_stationarity_decomposition', 'Stationarity & Decomposition'),
    4: ('phase4_modelling', 'Modelling Techniques'),
    5: ('phase5_evaluation', 'Model Evaluation'),
    6: ('phase6_forecasting', 'Forecasting & Interpretation'),
    7: ('phase7_domain_applications', 'Domain Applications'),
}


def run_phase(phase_num):
    """Run a single phase by executing it as a subprocess."""
    if phase_num not in PHASES:
        print(f"❌ Invalid phase number: {phase_num}")
        return False

    module_name, description = PHASES[phase_num]
    print(f"\n{'█'*70}")
    print(f"█  PHASE {phase_num} — {description.upper()}")
    print(f"{'█'*70}\n")

    start = time.time()
    try:
        script_path = os.path.join(os.getcwd(), f"{module_name}.py")
        result = subprocess.run([sys.executable, script_path], check=True)
        elapsed = time.time() - start
        print(f"\n⏱️  Phase {phase_num} completed in {elapsed:.1f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"\n❌ Phase {phase_num} failed after {elapsed:.1f}s: {e}")
        return False
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n❌ Phase {phase_num} failed after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = sys.argv[1:]

    if len(args) == 0:
        phases_to_run = list(PHASES.keys())
    elif len(args) == 1:
        phases_to_run = [int(args[0])]
    elif len(args) == 2:
        start_phase = int(args[0])
        end_phase = int(args[1])
        phases_to_run = list(range(start_phase, end_phase + 1))
    else:
        phases_to_run = [int(a) for a in args]

    print("═" * 70)
    print("  MONTHLY RAINFALL TREND & FORECASTING IN KARNATAKA")
    print("  Time Series & Forecasting Course Project")
    print("  Dataset: IMD Rainfall Data (1901–2015)")
    print("═" * 70)
    print(f"\n  Phases to run: {phases_to_run}")

    total_start = time.time()
    results = {}

    for phase in phases_to_run:
        success = run_phase(phase)
        results[phase] = success
        if not success:
            print(f"\n⚠️  Phase {phase} failed. Stopping execution.")
            break

    total_elapsed = time.time() - total_start

    print(f"\n{'═'*70}")
    print(f"  EXECUTION SUMMARY")
    print(f"{'═'*70}")
    for phase, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"   Phase {phase} ({PHASES[phase][1]}): {status}")
    print(f"\n   Total time: {total_elapsed:.1f} seconds")
    print(f"{'═'*70}")


if __name__ == '__main__':
    main()
