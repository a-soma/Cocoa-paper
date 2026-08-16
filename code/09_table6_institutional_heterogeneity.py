"""
08_table6_institutional_heterogeneity.py

Reproduces Table 6 (institutional heterogeneity and the Ivorian political
crisis control, Section 7.5).

fixed_price_regime is a time varying indicator, equal to 1 when a country
has an administered producer price mechanism (Ghana throughout; Cote
d'Ivoire in 1991 to 1998 under the Caisse de Stabilisation and in 2012 to
2024 under the Conseil du Cafe Cacao) and 0 otherwise (liberalized market).
shock_x_fixedprice is shock_it_baseline interacted with fixed_price_regime.
shock_x_civ_crisis is shock_it_baseline interacted with civ_crisis_dummy
(the Ivorian political crisis years defined in 01_data_processing.py).

Column 1 omits the crisis control; column 2 adds shock_x_civ_crisis to test
whether the institutional differential survives an explicit control for
Cote d'Ivoire's political crisis years.

Reproduction fidelity: column 1 (without the crisis control) reproduces
Table 6 exactly, including the p value of 0.006 on the administered regime
differential. Column 2 (with the crisis control) reproduces the paper's
qualitative conclusion, the differential loses significance once the
crisis control is added, but not the exact point estimates; this is the
same class of small specification difference documented for the other
extension tables in README.md.
"""

import pandas as pd
from _common import load_panel, fit_panel_ols, print_header

CONTROLS = ["gdp_growth_weo", "money_growth_m2_wdi", "trade_openness_wdi", "ecuador_crisis_dummy"]
DEP_VAR = "cpi_inflation_logdiff"


def run(df, include_crisis_control):
    regressors = ["shock_it_baseline", "shock_x_fixedprice"] + CONTROLS
    if include_crisis_control:
        regressors = regressors + ["shock_x_civ_crisis"]
    sub = df.dropna(subset=[DEP_VAR] + regressors)
    y = sub[DEP_VAR]
    X = sub[regressors]
    res = fit_panel_ols(y, X)
    return res, len(sub)


def main():
    df = load_panel()
    print_header("Table 6. Institutional heterogeneity and control for the Ivorian political crisis, H1")

    for label, with_crisis in [("Without crisis control", False), ("With crisis control", True)]:
        res, n = run(df, with_crisis)
        print(f"\n{label} (N={n}):")
        print(f"  Shock, liberalized regime:              {res.params['shock_it_baseline']:.2f} "
              f"[p={res.pvalues['shock_it_baseline']:.3f}]")
        print(f"  Differential coefficient, administered: {res.params['shock_x_fixedprice']:.2f} "
              f"[p={res.pvalues['shock_x_fixedprice']:.3f}]")
        if with_crisis:
            print(f"  Shock, Ivorian political crisis:        {res.params['shock_x_civ_crisis']:.2f} "
                  f"[p={res.pvalues['shock_x_civ_crisis']:.3f}]")


if __name__ == "__main__":
    main()
