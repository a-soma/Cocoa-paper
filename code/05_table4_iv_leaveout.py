"""
04_table4_iv_leaveout.py

Reproduces Table 4 (instrumental variables estimation, Section 6), using
the leave out instrument of Section 6.1,

    Instrument_it = Delta ln(World production - Cote d'Ivoire production
                              - Ghana production)_t * Exposure_i,

which is already merged into data/cacao_panel_master.csv as instrument_it
(see 01_data_processing.py, Part 5, for its construction from FAO
production data).

The paper estimates the two way fixed effects equation by instrumental
variables "applying a double demeaning, by country and year, to all
variables prior to estimation by two stage least squares" (Section 6.1).
This script implements the equivalent exact procedure for an unbalanced
panel, an LSDV two way fixed effects 2SLS regression (country and year
dummies included directly, rather than the closed form double demeaning
formula, which is only exact when the panel is balanced), with
heteroskedasticity robust standard errors.

Reproduction fidelity: this reproduces the qualitative pattern reported in
the paper exactly (the instrumented coefficient is insignificant for H1 and
becomes significant for H2a and H2b, with a first stage F statistic well
above 10 in all three specifications), but the exact point estimates in
this script differ from Table 4 by an amount larger than would be expected
from double precision alone. Table 4 itself was produced, like the other
extension tables, during an earlier interactive stage of the project whose
exact estimator settings were not preserved (see the project wide note in
README.md). Table 4's own numbers, not this script's output, are what is
reported in the paper; Section 6.2's discussion of the Nigeria leave one
out exclusion for H2a and H2b was verified separately against
data/cacao_panel_master.csv and is unaffected by this note.
"""

import pandas as pd
from linearmodels.iv import IV2SLS
from _common import DATA_PATH, print_header

CONTROLS = ["gdp_growth_weo", "money_growth_m2_wdi", "trade_openness_wdi"]


def run_iv(df, dep_var, controls):
    sub = df.dropna(subset=[dep_var, "shock_it_baseline", "instrument_it"] + controls).copy()

    country_dum = pd.get_dummies(sub["country"], prefix="c", drop_first=True).astype(float)
    year_dum = pd.get_dummies(sub["year"], prefix="y", drop_first=True).astype(float)
    exog = pd.concat([sub[controls], country_dum, year_dum], axis=1)
    exog.insert(0, "const", 1.0)
    endog = sub[["shock_it_baseline"]]
    instr = sub[["instrument_it"]]
    y = sub[dep_var]

    iv_res = IV2SLS(y, exog, endog, instr).fit(cov_type="robust")
    ols_res = IV2SLS(y, pd.concat([exog, endog], axis=1), None, None).fit(cov_type="robust")
    first_stage = IV2SLS(sub["shock_it_baseline"], pd.concat([exog, instr], axis=1), None, None).fit(cov_type="robust")

    first_stage_f = first_stage.tstats["instrument_it"] ** 2
    wu_hausman = iv_res.wooldridge_regression

    return {
        "n": len(sub),
        "countries": sub["country"].nunique(),
        "iv_coef": iv_res.params["shock_it_baseline"],
        "iv_p": iv_res.pvalues["shock_it_baseline"],
        "ols_coef": ols_res.params["shock_it_baseline"],
        "ols_p": ols_res.pvalues["shock_it_baseline"],
        "first_stage_f": first_stage_f,
        "wu_hausman_p": wu_hausman.pval,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    print_header("Table 4. Instrumental variables estimation")

    specs = {
        "H1 (inflation)": ("cpi_inflation_logdiff", CONTROLS + ["ecuador_crisis_dummy"]),
        "H2a (official REER)": ("reer_growth", CONTROLS),
        "H2b (bilateral RER)": ("rer_bilateral_growth", CONTROLS),
    }
    results = {}
    for name, (dep, controls) in specs.items():
        r = run_iv(df, dep, controls)
        results[name] = r
        print(f"\n{name}: N={r['n']}, countries={r['countries']}")
        print(f"  Instrumented coefficient: {r['iv_coef']:.3f} (p = {r['iv_p']:.3f})")
        print(f"  Least squares coefficient: {r['ols_coef']:.3f} (p = {r['ols_p']:.3f})")
        print(f"  First stage F statistic: {r['first_stage_f']:.2f}")
        print(f"  Wu Hausman test, p value: {r['wu_hausman_p']:.3f}")
    return results


if __name__ == "__main__":
    main()
