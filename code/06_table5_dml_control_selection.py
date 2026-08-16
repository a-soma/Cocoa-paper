"""
05_table5_dml_control_selection.py

Reproduces Table 5 (double machine learning, control selection, Section
7.2) using the partially linear double machine learning estimator of
Chernozhukov et al. (2018), with either Lasso or random forest selecting
among eleven candidate controls: current account balance, fiscal balance,
public debt, government spending, national savings, GDP per capita, the
oil price, and nominal GDP, in addition to the three baseline controls
(GDP growth, M2 growth, trade openness).

The dependent variable for H1 in this table is cpi_inflation_wdi (the
World Bank arithmetic percentage change series), matching the "arithmetic
measure" coefficient of 6.45 discussed in Section 7.1, not the log
difference measure used as the headline H1 result in Table 2.

Method: the panel is first double demeaned by country and year (removing
the two way fixed effects), then a partially linear model is estimated by
five fold cross fitting: for each fold, a regressor (Lasso with
cross validated penalty, or random forest) is trained on the other four
folds to predict the double demeaned outcome and the double demeaned
treatment (shock_it_baseline) from the double demeaned candidate controls;
the residuals are then combined and regressed on each other to recover the
treatment effect, following Chernozhukov et al. (2018).

Reproduction fidelity: cross fitting with Lasso or a random forest depends
on the random partition of folds and, for the random forest, on tree level
randomness. A fixed random_state is set below for this script's own
internal reproducibility, but this does not guarantee the exact point
estimates in Table 5, which were produced by an earlier, separately seeded
run. The sign, rough magnitude, and non significance of every coefficient
in this script matches Table 5's qualitative conclusion in every
specification tested.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from scipy import stats
from _common import DATA_PATH, print_header

CANDIDATES = [
    "current_account_gdp_weo", "fiscal_balance_gdp_weo", "gov_debt_gdp_weo",
    "gov_expenditure_gdp_weo", "national_savings_gdp_weo", "gdp_percapita_usd_wdi",
    "oil_price_usd_bbl", "gdp_current_usd_weo",
    "gdp_growth_weo", "money_growth_m2_wdi", "trade_openness_wdi",
]
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


def dml_plm(y, d, X, learner, n_folds=5, random_state=RANDOM_STATE):
    """Partially linear DML with K fold cross fitting (Chernozhukov et al., 2018)."""
    n = len(y)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    y_resid = np.zeros(n)
    d_resid = np.zeros(n)
    y_arr, d_arr, X_arr = np.asarray(y), np.asarray(d), np.asarray(X)
    for train_idx, test_idx in kf.split(X_arr):
        m_y = learner().fit(X_arr[train_idx], y_arr[train_idx])
        m_d = learner().fit(X_arr[train_idx], d_arr[train_idx])
        y_resid[test_idx] = y_arr[test_idx] - m_y.predict(X_arr[test_idx])
        d_resid[test_idx] = d_arr[test_idx] - m_d.predict(X_arr[test_idx])

    theta = np.sum(d_resid * y_resid) / np.sum(d_resid ** 2)
    psi = (y_resid - theta * d_resid) * d_resid
    j0 = np.mean(d_resid ** 2)
    var = np.mean(psi ** 2) / (j0 ** 2) / n
    se = np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(theta / se)))
    return theta, se, p


def run_one(df, dep_var, controls):
    sub = df.dropna(subset=[dep_var, "shock_it_baseline"] + controls).copy()
    sub = two_way_demean(sub, [dep_var, "shock_it_baseline"] + controls)

    lasso = lambda: LassoCV(cv=5, random_state=RANDOM_STATE, max_iter=20000)
    rf = lambda: RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)

    b_base, se_base, p_base = dml_plm(sub[dep_var], sub["shock_it_baseline"], sub[BASELINE_CONTROLS], lasso)
    b_lasso, se_lasso, p_lasso = dml_plm(sub[dep_var], sub["shock_it_baseline"], sub[CANDIDATES], lasso)
    b_rf, se_rf, p_rf = dml_plm(sub[dep_var], sub["shock_it_baseline"], sub[CANDIDATES], rf)

    return {
        "baseline": (b_base, p_base),
        "lasso": (b_lasso, p_lasso),
        "random_forest": (b_rf, p_rf),
        "n": len(sub),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    print_header("Table 5. Double machine learning, control selection")

    specs = {
        "H1 (inflation, arithmetic measure)": "cpi_inflation_wdi",
        "H2a (official REER)": "reer_growth",
        "H2b (bilateral RER)": "rer_bilateral_growth",
    }
    results = {}
    for name, dep in specs.items():
        r = run_one(df, dep, CANDIDATES)
        results[name] = r
        print(f"\n{name} (N={r['n']}):")
        for label in ("baseline", "lasso", "random_forest"):
            b, p = r[label]
            print(f"  {label:15s}: coefficient = {b:8.3f}, p = {p:.3f}")
    return results


if __name__ == "__main__":
    main()
