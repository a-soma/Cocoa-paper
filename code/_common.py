"""Shared helpers used by the numbered table and figure scripts."""

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

DATA_PATH = "data/cacao_panel_master.csv"


def load_panel():
    df = pd.read_csv(DATA_PATH)
    return df.set_index(["country", "year"])


def fit_panel_ols(y, X, bandwidth=3, debiased=True):
    """Two way fixed effects PanelOLS with Driscoll and Kraay (Bartlett
    kernel) standard errors, matching the covariance estimator used
    throughout the paper (Driscoll and Kraay, 1998)."""
    mod = PanelOLS(y, X, entity_effects=True, time_effects=True)
    return mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=bandwidth, debiased=debiased)


def wald_equality_test(res, param_a, param_b):
    """Wald test for equality of two coefficients from a fitted PanelOLS
    result, used for the H3 asymmetry test (Table 3)."""
    from scipy import stats

    b_a = res.params[param_a]
    b_b = res.params[param_b]
    cov = res.cov
    var_a = cov.loc[param_a, param_a]
    var_b = cov.loc[param_b, param_b]
    cov_ab = cov.loc[param_a, param_b]
    diff = b_a - b_b
    se_diff = np.sqrt(var_a + var_b - 2 * cov_ab)
    z = diff / se_diff
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"diff": diff, "se": se_diff, "z": z, "p": p}


def print_header(title):
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))
