"""Regenerate LOCO plots from existing saved CSV predictions without retraining.
"""

from pathlib import Path
import pandas as pd

from Gait.experiments.leave_one_cohort_out import (
    plot_uncertainty_correct_vs_incorrect,
    plot_uncertainty_by_cohort,
    plot_reliability_diagram,
    plot_roc_uncertainty_error_detection,
)

def main():
    csv_path = Path("Gait/outputs/loco/loco_subject_predictions.csv")
    plots_dir = Path("Gait/outputs/loco/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading saved subject predictions from: {csv_path}")
    sub_df = pd.read_csv(csv_path)

    p1 = plots_dir / "uncertainty_correct_vs_incorrect.png"
    p2 = plots_dir / "uncertainty_by_cohort.png"
    p3 = plots_dir / "reliability_curve.png"
    p4 = plots_dir / "roc_uncertainty_error_detection.png"

    print("Regenerating plots...")
    plot_uncertainty_correct_vs_incorrect(sub_df, p1)
    plot_uncertainty_by_cohort(sub_df, p2)
    plot_reliability_diagram(sub_df, p3)
    plot_roc_uncertainty_error_detection(sub_df, p4)

    print("\nVerifying plot generation:")
    all_exist = True
    for p in [p1, p2, p3, p4]:
        exists = p.exists() and p.stat().st_size > 0
        size_kb = p.stat().st_size / 1024 if exists else 0
        print(f"  [{'EXISTS' if exists else 'MISSING'}] {p.name} ({size_kb:.1f} KB)")
        if not exists:
            all_exist = False

    assert all_exist, "One or more plots failed to generate!"
    print("\nSUCCESS: All four LOCO plots regenerated and verified!")

if __name__ == "__main__":
    main()
