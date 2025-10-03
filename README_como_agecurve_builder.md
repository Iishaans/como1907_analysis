
# Como Age-Curve Dataset Builder

This package creates a **wide, BI-ready CSV** for Como 1907 that includes:
- Per-player minutes, age, position labels
- xG+xA, progressive passes, final-third touches (FBref)
- Contracts (Transfermarkt)
- Wages (Manually scraped Capology data - web scraper temporarily unavailable)

## Outputs
- `data/como_agecurve_wide.csv` — main training table
- `data/intermediate/*.csv` — raw pulls per season/source

## How to run (local machine)
1. Ensure Python 3.9+ and install deps:
   ```bash
   pip install pandas beautifulsoup4 lxml requests
   ```
2. Run the script:
   ```bash
   python como_agecurve_builder.py
   ```
3. Open the CSV in Excel, Numbers, or load into your BI tool.

## Notes
- FBref sometimes nests tables in HTML **comments**; this script extracts both visible and commented tables.
- Column names vary slightly season-to-season; the script normalizes the common fields (e.g., progressive passes).
- Transfermarkt is scraped with simple `read_html` heuristics; if layout changes, update selectors.
- Capology wage data is manually scraped (`Como_Wage_Breakdown_2425_2526_Cleaned.csv`) due to web scraper issues.
- Financials are **estimates**; treat as directional, not official.

## Feature Highlights
- `Minutes_2425`, `Minutes_2526`
- `xG_plus_xAG_2425`, `xG_plus_xAG_2526`
- `PrgPasses_25`, `PrgPasses_per90_25` (and `..._26`)
- `FinalThirdTouches_25`, `FinalThirdTouches_per90_25` (and `..._26`)
- `Latest_Pos`, `Latest_Pos4`, `Age_latest`

## Repro Tips
- If you want more KPIs (pressures, carries, receptions), add from FBref's `Defensive Actions` / `Carrying` tables.
- To extend beyond Como, change the `FBREF_TEAM_ID` and sources accordingly.
