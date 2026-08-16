"""
04_table3_asymmetry_h3.py

Reproduces Table 3 (H3, asymmetric response to positive and negative cocoa
shocks, Section 5), by splitting the exposure weighted cocoa shock into a
positive component (positive_shock_x_exposure) and a negative component
(negative_shock_x_exposure), estimated jointly in the same two way fixed
effects specification as Table 2's H1 column, and testing the null
hypothesis that the two components have equal coefficients (a Wald test on
the difference).
"""

import pandas as pd
from _common import load_panel, fit_panel_ols, wald_equality_test, print_header

CONTROLS = ["gdp_growth_weo", "money_growth_m2_wdi", "trade_openness_wdi", "ecuador_crisis_dummy"]
DEP_VAR = "cpi_inflation_logdiff"


def main():
    df = load_panel()
    print_header("Table 3. Asymmetric response to positive and negative cocoa shocks, H3, inflation")

    regressors = ["positive_shock_x_exposure", "negative_shock_x_exposure"] + CONTROLS
    sub = df.dropna(subset=[DEP_VAR] + regressors)
    y = sub[DEP_VAR]
    X = sub[regressors]
    res = fit_panel_ols(y, X)

    b_pos = res.params["positive_shock_x_exposure"]
    se_pos = res.std_errors["positive_shock_x_exposure"]
    p_pos = res.pvalues["positive_shock_x_exposure"]
    b_neg = res.params["negative_shock_x_exposure"]
    se_neg = res.std_errors["negative_shock_x_exposure"]
    p_neg = res.pvalues["negative_shock_x_exposure"]

    print(f"\nPositive shock component: {b_pos:.3f} (se {se_pos:.3f}, p {p_pos:.3f})")
    print(f"Negative shock component: {b_neg:.3f} (se {se_neg:.3f}, p {p_neg:.3f})")

    w = wald_equality_test(res, "positive_shock_x_exposure", "negative_shock_x_exposure")
    print(f"Difference, positive minus negative: {w['diff']:.3f} "
          f"(se {w['se']:.3f}, p {w['p']:.3f})")
    print(f"\nN = {res.nobs}, countries = {sub.index.get_level_values(0).nunique()}")

    return res, w


if __name__ == "__main__":
    main()
