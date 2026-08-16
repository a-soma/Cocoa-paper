"""
06_causal_forest.py

Reproduces the causal forest estimates cited in Section 7.3 (Wager and
Athey, 2018; Athey et al., 2019), a nonparametric complement to the linear
baseline that does not impose a functional form on treatment effect
heterogeneity.

Uses econml's CausalForestDML: the two way fixed effects are removed by
double demeaning (as in the other extension scripts), then a causal forest
estimates a constant (average) treatment effect of shock_it_baseline on
each outcome, controlling for the three baseline controls.

Reproduction fidelity: forest based estimators are inherently randomized
(bootstrap resampling of trees) and, per Section 7.3, the sample here is
small (150 to 206 observations) for this type of method, which the paper
itself flags as limiting this check to an illustrative role. Point
estimates from this script are not expected to match the paper's cited
values (14.20 for H1, -0.72 for H2a, -0.51 for H2b) exactly; the
qualitative conclusion, small and statistically insignificant average
effects consistent with the linear results, is what should reproduce.
"""

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from _common import DATA_PATH, print_header

BASELINE_CONTROLS = ["gdp_growth_weo", "money_growth_m2_wdi", "trade_openness_wdi"]
RANDOM_STATE = 0


def two_way_demean(frame, cols):
    out = frame.copy()
    for c in cols:
        overall = out[c].mean()
        cm = out.groupby("country")[c].transform("mean")
        ym = out.groupby("year")[c].transform("mean")
        out[c] = out[c] - cm - ym + overall
    return out


def run_one(df, dep_var):
    sub = df.dropna(subset=[dep_var, "shock_it_baseline"] + BASELINE_CONTROLS).copy()
    sub = two_way_demean(sub, [dep_var, "shock_it_baseline"] + BASELINE_CONTROLS)

    X = sub[BASELINE_CONTROLS].values
    T = sub["shock_it_baseline"].values
    Y = sub[dep_var].values

    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        model_t=RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        n_estimators=1000,
        random_state=RANDOM_STATE,
        cv=5,
    )
    est.fit(Y, T, X=X)
    ate = est.ate(X)
    ate_inf = est.ate_inference(X)
    return float(np.asarray(ate).ravel()[0]), float(ate_inf.pvalue()), len(sub)


def main():
    df = pd.read_csv(DATA_PATH)
    print_header("Causal forest average treatment effects (Section 7.3)")

    specs = {
        "H1 (inflation)": "cpi_inflation_logdiff",
        "H2a (official REER)": "reer_growth",
        "H2b (bilateral RER)": "rer_bilateral_growth",
    }
    for name, dep in specs.items():
        ate, p, n = run_one(df, dep)
        print(f"{name}: ATE = {ate:.2f}, p = {p:.3f}, N = {n}")


if __name__ == "__main__":
    main()
