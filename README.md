# Bank of Korea Monetary Policy Competition 2026

Data and analysis from our team's (통신보안) entry to the 23rd BOK Monetary Policy Competition.
The first round was a written report on the May 2026 rate decision. In the Seoul regional round we
presented on the July 2026 decision and recommended a 25bp hike (2.50% to 2.75%). We placed third
in the Seoul regional round.

My part was estimating and diagnosing the SVAR, and presenting the real economy section.

## Macro indicator workbook

`data/macro_indicator_workbook.xlsx` is where we kept all the data for the presentation: 110
indicators, monthly from 2025-01 to 2026-07 (quarterly for things released quarterly).

| sheet | what's in it |
|---|---|
| 02 policy rate & money | BOK base rate, Fed funds rate, KR-US rate gap, call rate, M1 / M2 / Lf |
| 03 prices | CPI, core, PPI, expected inflation (experts, households, BEI), CPI by item |
| 04 growth | GDP, industrial production, retail sales, capex, construction, CCSI, BSI |
| 05 labor | employment, unemployment, youth, by industry, wages, IT bonuses |
| 06 trade | exports, imports, current account, semiconductor exports, SCFI |
| 07 financial markets | KTB 3y / 10y, spreads, corporate bonds, KOSPI, foreign flows, FX |
| 08 credit | household credit, mortgages, corporate loans, delinquency rates |
| 09 housing & stability | house prices, jeonse, FSI, FVI, real estate PF, FX reserves |
| 10 global | US / euro area / Japan / China growth and policy rates, oil, CRB, DXY |

Sheet 01 is a dashboard that pulls from the other sheets, and sheet 11 lists the sources. Each
indicator has a note on how often it's released and with what lag, which we used to figure out what
would actually be published before presentation day. Preliminary vs. revised figures are marked in
the notes too.

The full list of indicators is in `data/indicator_catalog.json`.

## SVAR: oil prices and expected inflation

`svar-oil-shock/` has the data collection setup for the SVAR: monthly Dubai crude price, KRW/USD,
CPI and 1-year-ahead expected inflation from ECOS, 2010-01 to 2026-02. The README there lists the
ECOS series codes.

## Code

`scripts/` has the Python code. Every script expects to be run from the repo root layout
(`data/raw`, `data/processed`, `reports/`), and the ECOS scripts need an API key in the
`ECOS_API_KEY` environment variable.

- ECOS catalog: `cache_ecos_tables.py`, `cache_ecos_catalog.py`, `cache_ecos_curated_items.py`, `cache_ecos_items_for_matches.py`
- workbook: `build_data_workbook_xlsx.py`, `build_unresolved_with_source_notes.py`
- SVAR: `eda_raw_series.py`, `eda_final_series.py`, `adf_stl_cpi.py`, `estimate_var_diagnostics.py`, `compare_stl_var_residual_lags.py`, `estimate_irf_cholesky.py`
- scenario simulation: `custom_oil_path_preupdate_simulation.py`
- report table: `build_inflation_expectation_gap_table.py`

A few pieces (data processing, STL helper, the other scenario runs) are still being added.

## Sources

BOK ECOS, BOK monetary policy decisions and minutes, BOK Economic Outlook and Financial Stability
Report, KOSIS, Ministry of Employment and Labor, Korea Real Estate Board, KOFIA, KRX, FSS,
FRB / ECB / BOJ, KNOC Petronet.
