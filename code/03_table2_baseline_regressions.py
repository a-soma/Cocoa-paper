"""
03_table2_baseline_regressions.py

Reproduces Table 2 (baseline results for H1, H2a, H2b), Section 5.

H1: dependent variable is cpi_inflation_logdiff (the log difference of the
CPI, expressed in percentage points; see Section 7.1 for why this, and not
the World Bank arithmetic percentage change series, is the reference
inflation measure used throughout).
H2a: dependent variable is reer_growth (log difference of the official
real effective exchange rate index).
H2b: dependent variable is rer_bilateral_growth (log difference of the
constructed bilateral real exchange rate).
"""

import pandas as pd
from _common import load_panel, fit_panel_ols, print_header

CONTROLS = ["gdp_growth_weo", "money_growth_m2_wdi", "trade_openness_wdi"]


def run_spec(df, dep_var, controls=CONTROLS):
    sub = df.dropna(subset=[dep_var, "shock_it_baseline"] + controls)
    y = sub[dep_var]
    X = sub[["shock_it_baseline"] + controls]
    res = fit_panel_ols(y, X)
    return res, sub


def main():
    df = load_panel()

    print_header("Table 2. Baseline results, country and year fixed effects")

    res_h1, sub_h1 = run_spec(df, "cpi_inflation_logdiff", CONTROLS + ["ecuador_crisis_dummy"])
    res_h2a, sub_h2a = run_spec(df, "reer_growth")
    res_h2b, sub_h2b = run_spec(df, "rer_bilateral_growth")

    for name, res, sub in [("H1 (inflation)", res_h1, sub_h1),
                            ("H2a (official REER)", res_h2a, sub_h2a),
                            ("H2b (bilateral RER)", res_h2b, sub_h2b)]:
        b = res.params["shock_it_baseline"]
        se = res.std_errors["shock_it_baseline"]
        p = res.pvalues["shock_it_baseline"]
        n = res.nobs
        n_entities = sub.index.get_level_values(0).nunique()
        print(f"\n{name}: coefficient = {b:.3f}, std. error = {se:.3f}, "
              f"p = {p:.3f}, N = {n}, countries = {n_entities}, "
              f"within R2 = {res.rsquared_within:.3f}")

    print("\nFull H1 specification:")
    print(res_h1)

    return res_h1, res_h2a, res_h2b


if __name__ == "__main__":
    main()
