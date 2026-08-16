"""
09_synthetic_control_civ.py

Reproduces Figure 6 (synthetic control, Cote d'Ivoire, cocoa price shock of
2023 and 2024, Section 7.6), following Abadie and Gardeazabal (2003) and
Abadie et al. (2010, 2021).

Donor pool: Nigeria, Indonesia, the Dominican Republic, and Ecuador (the
four lowest cocoa exposure countries in the sample). Weights on the donor
countries are chosen, subject to being non negative and summing to one, to
minimize the squared gap between Cote d'Ivoire's observed inflation
trajectory and the weighted average of the donors over the pre shock
period, 2010 to 2022. The fitted weights are then used to build a
counterfactual for 2023 and 2024 and compare it to Cote d'Ivoire's observed
inflation in those two years.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from _common import DATA_PATH, print_header

DONORS = ["Nigeria", "Indonesia", "Dominican Republic", "Ecuador"]
TARGET = "Côte d'Ivoire"
PRE_PERIOD = range(2010, 2023)
POST_PERIOD = [2023, 2024]
OUTCOME = "cpi_inflation_wdi"


def main():
    df = pd.read_csv(DATA_PATH)

    pivot = df.pivot_table(index="year", columns="country", values=OUTCOME)
    pre = pivot.loc[list(PRE_PERIOD)]
    donor_pre = pre[DONORS].values
    target_pre = pre[TARGET].values

    def loss(w):
        pred = donor_pre @ w
        return np.sum((target_pre - pred) ** 2)

    n = len(DONORS)
    w0 = np.repeat(1 / n, n)
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n
    res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons)
    weights = res.x

    print_header("Synthetic control, Cote d'Ivoire, cocoa price shock of 2023 and 2024 (Figure 6)")
    print("\nDonor weights:")
    for donor, w in sorted(zip(DONORS, weights), key=lambda t: -t[1]):
        if w > 1e-6:
            print(f"  {donor}: {w * 100:.1f} percent")

    rmspe = np.sqrt(loss(weights) / len(PRE_PERIOD))
    print(f"\nPre period (2010 to 2022) root mean squared prediction error: {rmspe:.3f}")

    post = pivot.loc[POST_PERIOD]
    counterfactual = post[DONORS].values @ weights
    observed = post[TARGET].values
    gap = observed - counterfactual
    print("\nPost period comparison:")
    for year, o, c, g in zip(POST_PERIOD, observed, counterfactual, gap):
        print(f"  {year}: observed = {o:.2f}, synthetic = {c:.2f}, gap = {g:.2f} percentage points")


if __name__ == "__main__":
    main()
