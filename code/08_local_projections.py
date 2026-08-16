"""
07_local_projections.py

Reproduces Figure 4 (local projections, Section 7.4): for each horizon h
from 0 to 6, estimates

    y_{i,t+h} = alpha_i + lambda_t + beta_h * shock_it_baseline_t
                + controls_it + epsilon_it

by two way fixed effects PanelOLS with Driscoll and Kraay standard errors,
for both real exchange rate outcomes (official REER and bilateral RER),
following Jorda (2005) and Jorda and Taylor (2016).

Reproduction fidelity: the shape of the impulse response (coefficients
oscillating around zero with wide confidence bands, none individually
significant after a Bonferroni correction, as described in Section 7.4) is
what should reproduce; exact coefficients at each horizon depend on the
precise standard error bandwidth choice, as elsewhere in these extension
scripts.
"""

import numpy as np
import pandas as pd
from _common import load_panel, fit_panel_ols, print_header

CONTROLS = ["gdp_growth_weo", "money_growth_m2_wdi", "trade_openness_wdi"]
HORIZONS = range(0, 7)


def build_leads(df, dep_var, max_h):
    df = df.sort_values(["country", "year"]).copy()
    for h in range(0, max_h + 1):
        df[f"{dep_var}_lead{h}"] = df.groupby("country")[dep_var].shift(-h)
    return df


def run_lp(df, dep_var):
    df = build_leads(df.reset_index(), dep_var, max(HORIZONS)).set_index(["country", "year"])
    rows = []
    for h in HORIZONS:
        target = f"{dep_var}_lead{h}"
        sub = df.dropna(subset=[target, "shock_it_baseline"] + CONTROLS)
        y = sub[target]
        X = sub[["shock_it_baseline"] + CONTROLS]
        res = fit_panel_ols(y, X)
        rows.append({
            "horizon": h,
            "coef": res.params["shock_it_baseline"],
            "se": res.std_errors["shock_it_baseline"],
            "ci_lower": res.params["shock_it_baseline"] - 1.96 * res.std_errors["shock_it_baseline"],
            "ci_upper": res.params["shock_it_baseline"] + 1.96 * res.std_errors["shock_it_baseline"],
            "n": res.nobs,
        })
    return pd.DataFrame(rows)


def main():
    df = load_panel()
    print_header("Local projections, dynamic response to the cocoa shock (Figure 4)")

    for name, dep in [("Official REER (log difference)", "reer_growth"),
                       ("Bilateral RER (log difference)", "rer_bilateral_growth")]:
        print(f"\n{name}:")
        out = run_lp(df, dep)
        print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
