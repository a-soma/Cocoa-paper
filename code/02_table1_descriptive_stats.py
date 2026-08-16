"""
02_table1_descriptive_stats.py

Reproduces Table 1 (descriptive statistics) exactly.
"""

import pandas as pd
from _common import DATA_PATH, print_header


VARIABLES = {
    "cpi_inflation_logdiff": "Inflation (log difference of the CPI, percentage points)",
    "reer_growth": "Change in the official real effective exchange rate",
    "rer_bilateral_growth": "Change in the bilateral real exchange rate",
    "shock_it_baseline": "Exposure weighted cocoa shock",
    "exposure_i_baseline_first5": "Cocoa exposure (export share)",
    "gdp_growth_weo": "Real GDP growth (%)",
    "money_growth_m2_wdi": "M2 money supply growth (%)",
    "trade_openness_wdi": "Trade openness (exports plus imports over GDP)",
}


def main():
    df = pd.read_csv(DATA_PATH)
    print_header("Table 1. Descriptive statistics")
    rows = []
    for col, label in VARIABLES.items():
        s = df[col].dropna()
        rows.append({
            "Variable": label,
            "N": len(s),
            "Mean": round(s.mean(), 2),
            "Std. dev.": round(s.std(), 2),
            "Min": round(s.min(), 2),
            "Max": round(s.max(), 2),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
