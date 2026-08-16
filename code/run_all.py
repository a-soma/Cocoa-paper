"""
run_all.py

Runs every numbered script in sequence and prints a short summary of the
key coefficient reported by each one, as an end to end smoke test of the
replication package.

Run with: python run_all.py
"""

import importlib
import sys

SCRIPTS = [
    "01_data_processing",
    "02_table1_descriptive_stats",
    "03_table2_baseline_regressions",
    "04_table3_asymmetry_h3",
    "05_table4_iv_leaveout",
    "06_table5_dml_control_selection",
    "07_causal_forest",
    "08_local_projections",
    "09_table6_institutional_heterogeneity",
    "10_synthetic_control_civ",
]


def main():
    for name in SCRIPTS:
        print(f"\n\n{'#' * 70}\n# Running {name}.py\n{'#' * 70}")
        module = importlib.import_module(name)
        try:
            module.main()
        except Exception as exc:  # keep going so one failure doesn't hide the rest
            print(f"[run_all] {name}.py raised {type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
