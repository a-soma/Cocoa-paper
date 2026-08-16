# Replication code: Cocoa Price Shocks and Macroeconomic Adjustment in Cocoa Dependent Economies

This folder contains the data processing and analysis code for the paper
"Cocoa Price Shocks and Macroeconomic Adjustment in Cocoa Dependent
Economies: Evidence from Inflation and Real Exchange Rate Dynamics"
(Soma, Traore, Koassi, Kontiliguissonko, Traore, and Ouedraogo).

## Contents

- `data/cacao_panel_master.csv`: the analysis ready panel (8 countries, 1991
  to 2024, 272 country year observations, 77 variables). This is the exact
  file used to produce every table and figure in the paper.
- `01_data_processing.py`: documents and re implements the construction of
  the panel from the public sources cited in Section 3 of the paper (IMF
  World Economic Outlook and International Financial Statistics, World Bank
  World Development Indicators, IMF real effective exchange rate database,
  United Nations Comtrade, and the FAO cocoa production series via Our World
  in Data). See the note below on reproduction fidelity.
- `02_table1_descriptive_stats.py`: Table 1.
- `03_table2_baseline_regressions.py`: Table 2 (H1, H2a, H2b baseline).
- `04_table3_asymmetry_h3.py`: Table 3 (H3, positive versus negative shock).
- `05_table4_iv_leaveout.py`: Table 4 (leave out instrument, Section 6).
- `06_table5_dml_control_selection.py`: Table 5 (double machine learning).
- `07_causal_forest.py`: causal forest estimates cited in Section 7.3.
- `08_local_projections.py`: Figure 4, local projections at horizons 0 to 6.
- `09_table6_institutional_heterogeneity.py`: Table 6 and the Ivorian
  political crisis control (Section 7.5).
- `10_synthetic_control_civ.py`: Figure 6, the Cote d'Ivoire 2023 to 2024
  case study (Section 7.6).
- `run_all.py`: runs every script in sequence and prints a summary of the
  key coefficients, for a quick end to end check.
- `analysis_notebook.ipynb`: a single notebook that walks through the same
  pipeline with narrative cells, organized to mirror the structure of the
  paper. Use this for interactive exploration; use the numbered scripts for
  a clean, scriptable run.

## Setup

```
pip install -r requirements.txt
```

Python 3.10 or later is recommended. All scripts assume they are run from
this folder, so that the relative path `data/cacao_panel_master.csv`
resolves correctly.

## A note on reproduction fidelity

Two different things are shipped here, and they should not be confused.

`data/cacao_panel_master.csv` is the exact, final panel used to produce the
paper. Running the numbered scripts against this file reproduces, exactly,
Table 1, Table 2 (H1, H2a, H2b), Table 3 (the H3 asymmetry test, including
the p value of 0.076 on the equality test cited in the text), the "without
crisis control" column of Table 6, and Figure 6 (the synthetic control
donor weights of 59 percent Ecuador and 41 percent Dominican Republic, and
the 1.1 to 1.2 percentage point post period gap, both cited in Section
7.6). Figure 4 (local projections) also reproduces closely, including the
significant four year horizon coefficient for the bilateral real exchange
rate discussed in Section 7.4.

Three parts of the pipeline are close but not exact, and are flagged with a
comment at the top of the relevant script:

- Table 4 (the leave out instrumental variables estimation) reproduces the
  qualitative pattern (an insignificant H1 coefficient, significant H2a and
  H2b coefficients, first stage F statistics comfortably above 10) but not
  every point estimate to the last decimal.
- Table 5 (double machine learning) and the causal forest estimates of
  Section 7.3 depend on cross fitting fold assignment and, for the random
  forest and causal forest models, on tree level randomness; a fixed random
  seed is set for this script's own internal reproducibility, but this does
  not guarantee matching the paper's original, separately seeded run.
  Every specification tested reproduces the paper's conclusion, small,
  statistically insignificant coefficients, close in sign and rough
  magnitude to the linear baseline.
- The "with crisis control" column of Table 6 reproduces the qualitative
  conclusion (the institutional differential loses significance once the
  Ivorian political crisis years are controlled for) but not the exact
  point estimates.

`01_data_processing.py` re implements, from source level columns already
present in `cacao_panel_master.csv`, every derived variable used in the
paper: the exposure weights, the shift share shock and its positive and
negative components, the standardized shock, the crisis dummies, the
constructed bilateral real exchange rate, the log difference inflation
measure, and the leave out instrument. Running it prints a validation
report comparing every recomputed column against the shipped panel; as of
this writing, every derived column matches to floating point precision
except the bilateral real exchange rate level for Indonesia and the growth
rate of the official REER for Togo and the Dominican Republic, which the
script's own docstring explains in more detail (most likely a data vintage
difference in the underlying IMF or World Bank series for those three
countries, not a formula error, since the same formula is exact for the
other seven). The original bulk downloads from IMF, World Bank, and UN
Comtrade are not included in this repository (in places they run to several
hundred megabytes, and one exceeds a gigabyte for a single vintage); a
fresh download taken on a different date can also differ from the vintage
used here by routine data revisions. `data/cacao_panel_master.csv` is the
authoritative file for replicating the paper.

## Data sources

- IMF, World Economic Outlook database and International Financial
  Statistics: https://www.imf.org/en/Data
- IMF, real effective exchange rate database:
  https://data.imf.org
- World Bank, World Development Indicators:
  https://databank.worldbank.org/source/world-development-indicators
- United Nations, Comtrade database: https://comtradeplus.un.org
- FAO cocoa production series via Our World in Data:
  https://ourworldindata.org/cocoa-production

## Citation

If you use this code or data, please cite the paper. See the corresponding
author's contact details in the paper for questions about the replication
package.
