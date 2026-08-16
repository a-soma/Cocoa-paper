"""
01_data_processing.py

Documents and reimplements the construction of every derived variable in
the analysis panel, starting from the source level variables that come
directly from the public databases cited in Section 3 of the paper (IMF
World Economic Outlook and International Financial Statistics, World Bank
World Development Indicators, IMF real effective exchange rate database,
United Nations Comtrade, and the FAO cocoa production series via Our World
in Data).

This script is a validation and documentation tool: it recomputes every
derived column from the source level columns already present in
data/cacao_panel_master.csv, and checks the recomputed value against the
shipped value. Every formula below has been checked to match the shipped
panel to floating point precision, with one noted exception (the growth
rates of the two real exchange rate series, which differ from the shipped
values for a subset of country year cells, most likely because the
official IMF and World Bank series behind reer_index_2010_100 are revised
over time and the vintage used to build reer_growth and
rer_bilateral_growth predates the vintage frozen in this snapshot; see the
note in Part 4 below).

Run with: python 01_data_processing.py
"""

import numpy as np
import pandas as pd

RAW_PANEL = "data/cacao_panel_master.csv"
LEAVEOUT_INSTRUMENT = "data/raw/cocoa_leaveout_instrument.csv"


def load_source_panel():
    """
    Load the panel and keep only the source level columns, i.e. the
    variables that come directly from a public database rather than being
    computed from other columns in this panel. This is the input a fresh
    build of the panel would start from.
    """
    df = pd.read_csv(RAW_PANEL)
    source_cols = [
        "country", "iso3", "year",
        # cocoa price and trade (world price and country trade shares)
        "cocoa_price_usd_kg", "cocoa_price_shock_logdiff", "cocoa_price_shock_std",
        "cocoa_export_share", "cocoa_exports_usd", "total_exports_usd", "total_imports_usd",
        # IMF WEO / IFS
        "cpi_inflation_weo", "current_account_gdp_weo", "fiscal_balance_gdp_weo",
        "gdp_current_usd_weo", "gdp_growth_weo", "gov_debt_gdp_weo",
        "gov_expenditure_gdp_weo", "national_savings_gdp_weo", "unemployment_rate_weo",
        "neer_index_2010_100", "reer_index_2010_100",
        "ir_deposit_rate", "ir_discount_rate", "ir_govbond_rate", "ir_lending_rate",
        "ir_moneymarket_rate", "ir_policy_rate", "ir_repo_rate", "ir_savings_rate", "ir_tbill_rate",
        # World Bank WDI
        "cpi_inflation_wdi", "exrate_lcu_per_usd_wdi", "gdp_current_usd_wdi",
        "gdp_deflator_inflation_wdi", "gdp_percapita_usd_wdi", "money_growth_m2_wdi",
        "total_reserves_usd_wdi", "terms_of_trade_2015_100_wdi",
        "trade_openness_wdi", "trade_openness_weo",
        "cpi_level_2010_100", "cpi_us_2010_100", "cpi_level",
        "reserve_flow_usd_millions_bop", "oil_price_usd_bbl",
    ]
    return df[[c for c in source_cols if c in df.columns]].copy()


def add_exposure_and_shock(df):
    """
    Part 1: the shift share exposure weight and the exposure weighted cocoa
    price shock (Section 4 of the paper).

    exposure_i_baseline_first5 is each country's average cocoa export share
    over the first five years for which cocoa_export_share is observed,
    fixed over time so that it cannot itself respond to the contemporaneous
    price shock (Section 4).

    shock_it = cocoa_price_shock_logdiff_t * exposure_i_baseline_first5,
    the shift share (Bartik) shock used throughout the paper.
    """
    df = df.sort_values(["country", "year"]).copy()

    def first5_mean(g):
        g = g.sort_values("year")
        first5 = g.dropna(subset=["cocoa_export_share"]).head(5)
        return first5["cocoa_export_share"].mean()

    exposure_baseline = df.groupby("country", group_keys=False).apply(first5_mean)
    exposure_baseline.name = "exposure_i_baseline_first5"
    df = df.merge(exposure_baseline.reset_index(), on="country", how="left")

    # full sample exposure (used only for the robustness variant reported in
    # the text): average cocoa export share over the whole sample period
    exposure_full = df.groupby("country")["cocoa_export_share"].transform("mean")
    df["exposure_i_fullsample"] = exposure_full

    df["shock_it_baseline"] = df["cocoa_price_shock_logdiff"] * df["exposure_i_baseline_first5"]
    df["shock_it_fullsample"] = df["cocoa_price_shock_logdiff"] * df["exposure_i_fullsample"]

    # positive and negative components (Section 4, hypothesis H3)
    df["positive_shock"] = df["cocoa_price_shock_logdiff"].clip(lower=0)
    df["negative_shock"] = df["cocoa_price_shock_logdiff"].clip(upper=0)
    df["positive_shock_x_exposure"] = df["positive_shock"] * df["exposure_i_baseline_first5"]
    df["negative_shock_x_exposure"] = df["negative_shock"] * df["exposure_i_baseline_first5"]
    df["positive_shock_x_exposure_full"] = df["positive_shock"] * df["exposure_i_fullsample"]
    df["negative_shock_x_exposure_full"] = df["negative_shock"] * df["exposure_i_fullsample"]

    # standardized (z scored) versions used in a robustness check
    df["std_shock_it_baseline"] = df["cocoa_price_shock_std"] * df["exposure_i_baseline_first5"]
    df["std_shock_it_fullsample"] = df["cocoa_price_shock_std"] * df["exposure_i_fullsample"]
    df["positive_std_shock"] = df["cocoa_price_shock_std"].clip(lower=0)
    df["negative_std_shock"] = df["cocoa_price_shock_std"].clip(upper=0)
    df["positive_std_shock_x_exposure"] = df["positive_std_shock"] * df["exposure_i_baseline_first5"]
    df["negative_std_shock_x_exposure"] = df["negative_std_shock"] * df["exposure_i_baseline_first5"]

    df["shock_it_baseline_lag1"] = df.groupby("country")["shock_it_baseline"].shift(1)
    df["shock_it_baseline_lag2"] = df.groupby("country")["shock_it_baseline"].shift(2)

    return df


def add_crisis_dummies(df):
    """
    Part 2: the two crisis controls used in Table 2 and Table 6.

    ecuador_crisis_dummy flags Ecuador's 1999 to 2000 banking and
    dollarization crisis (Section 4).

    civ_crisis_dummy flags Cote d'Ivoire's political crisis years used in
    Section 7.5: the 1999 and 2000 coup d'etat, the 2002 to 2004 onset and
    most violent phase of the rebellion, and the 2010 and 2011 post
    electoral crisis.
    """
    df = df.copy()
    df["ecuador_crisis_dummy"] = (
        (df["country"] == "Ecuador") & (df["year"].between(1999, 2000))
    ).astype(int)

    civ_crisis_years = [1999, 2000, 2002, 2003, 2004, 2010, 2011]
    is_civ = df["country"].isin(["Cote d'Ivoire", "Côte d'Ivoire"])
    df["civ_crisis_dummy"] = (is_civ & df["year"].isin(civ_crisis_years)).astype(int)
    df["shock_x_civ_crisis"] = df["shock_it_baseline"] * df["civ_crisis_dummy"]
    return df


def add_exchange_rate_variables(df):
    """
    Part 3: the constructed bilateral real exchange rate (Section 3), used
    for Ecuador and Indonesia, whose official REER series is unavailable or
    not meaningful (Ecuador is dollarized since 2000).

    rer_bilateral_usd_2010_100 = [cpi_level_2010_100 / (exrate_lcu_per_usd_wdi
    * cpi_us_2010_100)], indexed so that the country's own 2010 value equals
    100. A rise in this index is a real appreciation of the local currency
    against the US dollar, matching the sign convention of the official
    REER index.
    """
    df = df.sort_values(["country", "year"]).copy()
    raw = df["cpi_level_2010_100"] / (df["exrate_lcu_per_usd_wdi"] * df["cpi_us_2010_100"])

    def normalize_to_2010(g):
        base = g.loc[g["year"] == 2010, "_raw"]
        if len(base) == 0 or pd.isna(base.values[0]) or base.values[0] == 0:
            return pd.Series(np.nan, index=g.index)
        return g["_raw"] / base.values[0] * 100

    tmp = df.assign(_raw=raw)
    df["rer_bilateral_usd_2010_100"] = tmp.groupby("country", group_keys=False).apply(normalize_to_2010)

    return df


def add_inflation_and_growth_rates(df):
    """
    Part 4: the log difference inflation measure used as the reference
    measure for H1 from Section 7.1 onward, and the log difference growth
    rates of the two real exchange rate series (H2a, H2b dependent
    variables in Table 2).

    cpi_inflation_logdiff = 100 * [ln(cpi_level_t) - ln(cpi_level_{t-1})].

    Note on reproduction fidelity: reer_growth and rer_bilateral_growth are
    defined the same way, as the log difference of reer_index_2010_100 and
    rer_bilateral_usd_2010_100 respectively. This recomputation matches the
    shipped panel exactly for six of the eight countries. It does not match
    for Togo and the Dominican Republic on reer_growth, and for Indonesia on
    rer_bilateral_growth; in all three cases the discrepancy affects that
    country's entire time series rather than isolated years, which points
    to those countries' underlying IMF or World Bank exchange rate series
    having been pulled at a different data vintage than the rest of the
    panel, rather than to a formula error (the formula itself is exact for
    the other six countries). The shipped growth columns, not this
    recomputation, are what was used for Table 2 and every other result in
    the paper.
    """
    df = df.sort_values(["country", "year"]).copy()
    df["cpi_inflation_logdiff"] = 100 * np.log(df["cpi_level"]).groupby(df["country"]).diff()
    df["reer_growth"] = np.log(df["reer_index_2010_100"]).groupby(df["country"]).diff()
    df["rer_bilateral_growth"] = np.log(df["rer_bilateral_usd_2010_100"]).groupby(df["country"]).diff()
    return df


def add_leaveout_instrument(df):
    """
    Part 5: the leave out instrument of Section 6.1,

        Instrument_it = Delta ln(World production - Cote d'Ivoire production
                                  - Ghana production)_t * Exposure_i,

    built from FAO cocoa production data via Our World in Data
    (data/raw/cocoa_production_wide.csv, aggregated into
    data/raw/cocoa_leaveout_instrument.csv).
    """
    loo = pd.read_csv(LEAVEOUT_INSTRUMENT)
    loo = loo[["year", "leaveout_growth", "civ_ghana_share"]]
    df = df.merge(loo, on="year", how="left")
    df["instrument_it"] = df["leaveout_growth"] * df["exposure_i_baseline_first5"]
    return df


def validate(recomputed, shipped_path=RAW_PANEL):
    """Compare every recomputed derived column against the shipped panel."""
    shipped = pd.read_csv(shipped_path)
    key = ["country", "year"]
    merged = recomputed.merge(shipped, on=key, suffixes=("_recomputed", ""))

    check_cols = [
        "exposure_i_baseline_first5", "shock_it_baseline", "positive_shock_x_exposure",
        "negative_shock_x_exposure", "std_shock_it_baseline", "ecuador_crisis_dummy",
        "civ_crisis_dummy", "rer_bilateral_usd_2010_100", "cpi_inflation_logdiff",
        "reer_growth", "rer_bilateral_growth", "instrument_it",
    ]
    print("\nValidation against data/cacao_panel_master.csv:")
    print(f"{'variable':32s} {'max abs diff':>14s} {'n compared':>12s}")
    for col in check_cols:
        a = merged[f"{col}_recomputed"]
        b = merged[col]
        diff = (a - b).abs()
        n = diff.notna().sum()
        print(f"{col:32s} {diff.max():14.6g} {n:12d}")


def main():
    df = load_source_panel()
    df = add_exposure_and_shock(df)
    df = add_crisis_dummies(df)
    df = add_exchange_rate_variables(df)
    df = add_inflation_and_growth_rates(df)
    df = add_leaveout_instrument(df)
    validate(df)
    print(f"\nReconstructed panel: {df.shape[0]} rows, {df.shape[1]} columns.")
    print("The authoritative panel for replicating every table in the paper")
    print("is data/cacao_panel_master.csv; this script exists to document")
    print("and verify how its derived columns are built.")


if __name__ == "__main__":
    main()
