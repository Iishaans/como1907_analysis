
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Como 1907: Age-Curve Wide Dataset Builder
----------------------------------------
Scrapes FBref (player stats by season), Transfermarkt (contracts), Capology (wages),
and produces a single wide CSV with per-player features for 2024-25 and 2025-26.

Outputs:
  - data/como_agecurve_wide.csv
  - data/intermediate/* (per-table raw pulls)

Requires: Python 3.9+, requests, pandas, beautifulsoup4, lxml
Usage:
  python como_agecurve_builder.py
"""

import re
import os
import time
import json
import math
import unicodedata
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment

# -------- Configuration --------
FBREF_TEAM_ID = "28c9c3cd"   # Como 1907 team id on FBref
SEASONS = ["2024-2025", "2025-2026"]  # last season + current
FBREF_BASE = "https://fbref.com"
FBREF_TEAM_SEASON = f"{FBREF_BASE}/en/squads/{FBREF_TEAM_ID}/{{season}}/Como-Stats"

TM_DETAILED_SQUAD = "https://www.transfermarkt.com/como-1907/kader/verein/1047/saison_id/{year}"
TM_HEADERS = {"User-Agent": "Mozilla/5.0"}  # TM blocks default agents

CAPO_SALARIES = {
    "2024-2025": "https://www.capology.com/club/como/salaries/2024-2025/",
    "2025-2026": "https://www.capology.com/club/como/salaries/"
}

OUT_DIR = "data"
INT_DIR = os.path.join(OUT_DIR, "intermediate")
os.makedirs(INT_DIR, exist_ok=True)

# -------- Helpers --------
def normalize_name(name: str) -> str:
    if pd.isna(name):
        return name
    name = unicodedata.normalize("NFKD", str(name))
    name = re.sub(r"\s+", " ", name).strip()
    return name

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns and tidy up names."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in tup if str(x) != "nan"]).strip()
                      for tup in df.columns.values]
    else:
        df.columns = [str(c) for c in df.columns]
    # Remove duplicate whitespace and 'Unnamed' noise
    df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
    df.columns = [re.sub(r"^Unnamed.*?_?", "", c).strip("_ ").strip() for c in df.columns]
    return df

def normalize_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common variants to canonical names: Player, Pos, Age, Min, 90s."""
    cols = {c.lower(): c for c in df.columns}

    def find_col(*needles):
        for lc, orig in cols.items():
            if any(n in lc for n in needles):
                return orig
        return None

    mapping = {}
    player = find_col("player")
    if player and player != "Player":
        mapping[player] = "Player"

    pos = find_col("pos", "position")
    if pos and pos != "Pos":
        mapping[pos] = "Pos"

    age = find_col("age")
    if age and age != "Age":
        mapping[age] = "Age"

    # Minutes sometimes appear as 'Min', 'Minutes', or under 'Playing Time_Min'
    mins = find_col(" minutes", " min", "playing time_min", "_min")
    if mins and mins != "Min":
        mapping[mins] = "Min"

    nineties = find_col("90s")
    if nineties and nineties != "90s":
        mapping[nineties] = "90s"

    if mapping:
        df.rename(columns=mapping, inplace=True)
    return df

def read_fbref_tables(url: str) -> Dict[str, pd.DataFrame]:
    """Parse all visible and HTML-commented tables on an FBref team season page."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # Basic sanity check to detect bot walls or wrong page
    title = (soup.title.get_text(strip=True) if soup.title else "") or ""
    if not any(k in title for k in ["Como", "FBref"]):
        print(f"[WARN] Unexpected page title while fetching {url}: {title}")

    tables: Dict[str, pd.DataFrame] = {}

    # Regular tables
    for table in soup.find_all("table"):
        caption = table.find("caption")
        key = caption.get_text(strip=True) if caption else table.get("id") or "table"
        try:
            df = pd.read_html(str(table))[0]
            df = flatten_columns(df)
            tables[key] = df
        except Exception:
            pass

    # Commented tables (FBref often wraps advanced tables in comments)
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for c in comments:
        csoup = BeautifulSoup(c, "lxml")
        for table in csoup.find_all("table"):
            caption = table.find("caption")
            key = caption.get_text(strip=True) if caption else table.get("id") or "table"
            try:
                df = pd.read_html(str(table))[0]
                df = flatten_columns(df)
                tables[key] = df
            except Exception:
                pass

    return tables

def pick_first_matching(keys: List[str], tables: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    for k in keys:
        for tname, df in tables.items():
            if k.lower() in str(tname).lower():
                return df.copy()
    return None

def clean_player_df(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten columns, normalize core column names, and standardize position buckets."""
    df = flatten_columns(df.copy())
    df = normalize_core_columns(df)

    if "Player" not in df.columns:
        # Not a player-level table (or FBref changed layout) — return empty to avoid crash
        return pd.DataFrame()

    # Drop aggregate/empty rows
    df = df[df["Player"].notna()]
    df = df.loc[~df["Player"].astype(str).str.contains("Squad Total|Opponent|Opponents", case=False, na=False)]

    # Normalize names/positions
    df["Player"] = df["Player"].map(normalize_name)
    if "Pos" in df.columns:
        df["PrimaryPos"] = df["Pos"].astype(str).str.split(",").str[0].str.strip()
        df["PrimaryPos4"] = df["PrimaryPos"].str.upper().map({
            "GK": "GK",
            "CB": "DF","RB": "DF","LB": "DF","DF": "DF","FB": "DF","WB": "DF",
            "DM": "MF","CM": "MF","AM": "MF","MF": "MF",
            "ST": "FW","FW": "FW","W": "FW","RW": "FW","LW": "FW"
        }).fillna(df["PrimaryPos"])
    return df

def prep_cols(df: pd.DataFrame, prefix: str, keep: List[str]) -> pd.DataFrame:
    cols = [c for c in keep if c in df.columns]
    df = df[cols].copy()
    ren = {c: f"{prefix}_{c}" for c in cols if c != "Player"}
    return df.rename(columns=ren)

# -------- Scrapers --------
def scrape_fbref_for_season(season: str) -> pd.DataFrame:
    url = FBREF_TEAM_SEASON.format(season=season)
    print(f"Scraping FBref: {url}")
    tables = read_fbref_tables(url)

    # Candidate tables
    standard = pick_first_matching(['Standard Stats', 'Standard'], tables)
    shooting = pick_first_matching(['Shooting'], tables)
    passing = pick_first_matching(['Passing'], tables)
    pass_types = pick_first_matching(['Pass Types'], tables)
    possession = pick_first_matching(['Possession', 'Poss'], tables)
    playing = pick_first_matching(['Playing Time'], tables)
    misc = pick_first_matching(['Misc', 'Miscellaneous'], tables)

    parts: List[pd.DataFrame] = []

    if standard is not None:
        df = clean_player_df(standard)
        if not df.empty:
            keep = ['Player', 'Pos', 'Age', 'Min', 'Gls', 'Ast', 'G+A', 'xG', 'npxG', 'xAG', 'npxG+xAG', 'G+A-PK', '90s']
            parts.append(prep_cols(df, f"{season}_std", keep))

    if shooting is not None:
        df = clean_player_df(shooting)
        if not df.empty:
            keep = ['Player', 'Sh', 'SoT', 'G/Sh', 'G/SoT', 'Dist']
            parts.append(prep_cols(df, f"{season}_shoot", keep))

    if passing is not None:
        df = clean_player_df(passing)
        if not df.empty:
            # Normalize progressive passes column name variants
            for cand in ['PrgP', 'Prog', 'ProgPasses', 'Prg Passes', 'Progressive passes']:
                if cand in df.columns:
                    df.rename(columns={cand: 'Prog'}, inplace=True)
                    break
            keep = ['Player', 'Cmp', 'Att', 'Cmp%', 'Prog']
            parts.append(prep_cols(df, f"{season}_pass", keep))

    if pass_types is not None:
        df = clean_player_df(pass_types)
        if not df.empty:
            keep = ['Player', 'KP']  # key passes
            parts.append(prep_cols(df, f"{season}_passtypes", keep))

    if possession is not None:
        df = clean_player_df(possession)
        if not df.empty:
            # Normalize final-third touches label variants
            for cand in ['Att 3rd', 'Att3rd', 'Final 3rd', 'Attacking third']:
                if cand in df.columns:
                    df.rename(columns={cand: 'Att 3rd'}, inplace=True)
                    break
            keep = ['Player', 'Touches', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Att Pen']
            # Some seasons may name 'Att Pen' differently
            if 'Att Pen' not in df.columns:
                for c in df.columns:
                    if 'pen' in c.lower() and ('att' in c.lower() or 'touches' in c.lower()):
                        df.rename(columns={c: 'Att Pen'}, inplace=True)
                        break
            parts.append(prep_cols(df, f"{season}_poss", keep))

    if playing is not None:
        df = clean_player_df(playing)
        if not df.empty:
            keep = ['Player', 'Starts', 'Min', 'Mn/Start', 'Compl', '90s']
            parts.append(prep_cols(df, f"{season}_play", keep))

    if misc is not None:
        df = clean_player_df(misc)
        if not df.empty:
            keep = ['Player', 'CrdY', 'CrdR', 'Won%', 'Int', 'TklW']
            parts.append(prep_cols(df, f"{season}_misc", keep))

    # Merge all on Player
    out = None
    for p in parts:
        out = p if out is None else out.merge(p, on='Player', how='outer')

    # Derive xG+xA
    if out is not None and not out.empty:
        if f"{season}_std_xG" in out.columns and f"{season}_std_xAG" in out.columns:
            out[f"{season}_derived_xG_plus_xAG"] = pd.to_numeric(out[f"{season}_std_xG"], errors='coerce') + \
                                                   pd.to_numeric(out[f"{season}_std_xAG"], errors='coerce')
    # Deduplicate: keep only one row per player (if any duplicates exist, keep the first)
    if out is not None and not out.empty:
        out = out.drop_duplicates(subset=['Player'], keep='first')
    return out if out is not None else pd.DataFrame(columns=['Player'])

def scrape_transfermarkt_contracts(year: int) -> pd.DataFrame:
    url = TM_DETAILED_SQUAD.format(year=year)
    resp = requests.get(url, headers=TM_HEADERS, timeout=30)
    resp.raise_for_status()
    dfs = pd.read_html(resp.text)
    # Find the main squad table: usually the first large one with "Player" column
    main = None
    for df in dfs:
        df = flatten_columns(df)
        if 'Player' in df.columns and any(c.lower().startswith('contract') for c in df.columns.str.lower()):
            main = df
            break
    if main is None:
        candidates = []
        for df in dfs:
            df = flatten_columns(df)
            if 'Player' in df.columns:
                candidates.append(df)
        main = max(candidates, key=lambda d: d.shape[1]) if candidates else pd.DataFrame()

    if not main.empty:
        cols = [c for c in main.columns if c in ['Player', 'Position', 'Date of birth / Age', 'DOB_Age', 'Contract expires', 'Market value']]
        main = main[cols].copy()
        main.rename(columns={
            'Date of birth / Age': 'DOB_Age',
            'Contract expires': 'Contract_expires',
            'Market value': 'MarketValue'
        }, inplace=True)
        main['Player'] = main['Player'].map(normalize_name)
        if 'Contract_expires' in main.columns:
            main['Contract_expires'] = main['Contract_expires'].astype(str).str.replace(r'[^0-9\-/.]', '', regex=True)
        # Deduplicate: keep only one row per player
        main = main.drop_duplicates(subset=['Player'], keep='first')
    return main

def scrape_capology_wages(season: str) -> pd.DataFrame:
    url = CAPO_SALARIES[season]
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    dfs = pd.read_html(resp.text)

    table = None
    for df in dfs:
        df = flatten_columns(df)
        cols = [c.lower() for c in df.columns.astype(str)]
        if 'player' in cols and any('annual' in c and 'salary' in c for c in cols):
            table = df
            break
    if table is None:
        return pd.DataFrame()

    keep_cols = []
    for c in table.columns.astype(str):
        lc = c.lower()
        if lc == 'player' or 'position' in lc or 'annual' in lc or 'weekly' in lc or 'contract' in lc or 'age' in lc:
            keep_cols.append(c)
    out = table[keep_cols].copy()
    out.rename(columns={keep_cols[0]: 'Player'}, inplace=True)
    out['Player'] = out['Player'].map(normalize_name)
    # Deduplicate: keep only one row per player
    out = out.drop_duplicates(subset=['Player'], keep='first')
    return out

# -------- Orchestration --------
def main():
    frames = []
    for season in SEASONS:
        print(f"Scraping FBref for {season}...")
        df = scrape_fbref_for_season(season)
        if df is not None and not df.empty:
            # Deduplicate: keep only one row per player
            df = df.drop_duplicates(subset=['Player'], keep='first')
            frames.append(df)
            df.to_csv(os.path.join(INT_DIR, f"fbref_{season.replace('-', '')}.csv"), index=False)
        else:
            print(f"[WARN] No FBref player tables parsed for {season}")

    fbref_merged = None
    for df in frames:
        fbref_merged = df if fbref_merged is None else fbref_merged.merge(df, on='Player', how='outer')

    # Deduplicate after merge
    if fbref_merged is not None and not fbref_merged.empty:
        fbref_merged = fbref_merged.drop_duplicates(subset=['Player'], keep='first')

    # Contracts
    try:
        contracts_2024 = scrape_transfermarkt_contracts(2024)
    except Exception as e:
        print("[WARN] Transfermarkt 2024 scrape failed:", e)
        contracts_2024 = pd.DataFrame()

    try:
        contracts_2025 = scrape_transfermarkt_contracts(2025)
    except Exception as e:
        print("[WARN] Transfermarkt 2025 scrape failed:", e)
        contracts_2025 = pd.DataFrame()

    if not contracts_2024.empty or not contracts_2025.empty:
        contracts = pd.concat(
            [contracts_2024.assign(Season='2024-2025'), contracts_2025.assign(Season='2025-2026')],
            ignore_index=True
        )
    else:
        contracts = pd.DataFrame(columns=['Player','Season'])
    # Deduplicate: keep only one row per player per season
    contracts = contracts.drop_duplicates(subset=['Player', 'Season'], keep='first')
    contracts.to_csv(os.path.join(INT_DIR, "transfermarkt_contracts.csv"), index=False)

    # Wages - Use manually scraped comprehensive wage data
    try:
        wage_file = os.path.join(INT_DIR, "Como_Wage_Breakdown_2425_2526_Cleaned.csv")
        if os.path.exists(wage_file):
            wages_raw = pd.read_csv(wage_file)
            print(f"Loaded comprehensive wage data: {wage_file}")
            print(f"Wage data shape: {wages_raw.shape}")
            
            # Process wage data to match expected format
            wages = wages_raw[['Player_Clean', 'Season', 'Gross_PW_EUR', 'Gross_PY_EUR', 'Position', 'Age']].copy()
            wages.rename(columns={'Player_Clean': 'Player', 'Gross_PW_EUR': 'Weekly_Gross_EUR', 'Gross_PY_EUR': 'Yearly_Gross_EUR'}, inplace=True)
            
            # Convert season format to match other data (2024-25 -> 2024-2025, 2025-26 -> 2025-2026)
            wages['Season'] = wages['Season'].str.replace('2024-25', '2024-2025').str.replace('2025-26', '2025-2026')
            
        else:
            # Fallback to old scraper if manual data not available
            print("[WARN] Manual wage data not found, falling back to Capology scraper")
            try:
                wages_2425 = scrape_capology_wages("2024-2025")
            except Exception as e:
                print("[WARN] Capology 2024-25 scrape failed:", e)
                wages_2425 = pd.DataFrame()

            try:
                wages_2526 = scrape_capology_wages("2025-2026")
            except Exception as e:
                print("[WARN] Capology 2025-26 scrape failed:", e)
                wages_2526 = pd.DataFrame()

            if not wages_2425.empty or not wages_2526.empty:
                wages = pd.concat(
                    [wages_2425.assign(Season='2024-2025'), wages_2526.assign(Season='2025-2026')],
                    ignore_index=True
                )
            else:
                wages = pd.DataFrame(columns=['Player','Season'])
                
    except Exception as e:
        print(f"[ERROR] Failed to process wage data: {e}")
        wages = pd.DataFrame(columns=['Player','Season'])
        
    # Deduplicate: keep only one row per player per season
    if not wages.empty:
        wages = wages.drop_duplicates(subset=['Player', 'Season'], keep='first')
        wages.to_csv(os.path.join(INT_DIR, "capology_wages.csv"), index=False)

    # Merge all
    out = fbref_merged if fbref_merged is not None else pd.DataFrame(columns=['Player'])
    out = out.merge(contracts, on=['Player'], how='left', suffixes=('', '_tm'))
    out = out.merge(wages, on=['Player', 'Season'], how='left', suffixes=('', '_capo'))

    # Deduplicate after all merges: keep only one row per player
    # Sort by Season to prefer 2025-2026 (more recent) data
    if 'Season' in out.columns:
        out['Season_priority'] = out['Season'].map({'2025-2026': 1, '2024-2025': 2})
        out = out.sort_values(['Player', 'Season_priority']).drop_duplicates(subset=['Player'], keep='first')
        out = out.drop(columns=['Season_priority'])
    else:
        out = out.drop_duplicates(subset=['Player'], keep='first')

    # Recompute simple labels
    pos_cols = [c for c in out.columns if c.endswith('_std_Pos')]
    if pos_cols:
        out['Latest_Pos'] = out[pos_cols[-1]]

    def pos_bucket(val):
        if pd.isna(val): return None
        s = str(val).upper()
        if any(k in s for k in ['GK']): return 'GK'
        if any(k in s for k in ['CB','RB','LB','DF','FB','WB']): return 'DF'
        if any(k in s for k in ['DM','AM','MF','CM']): return 'MF'
        if any(k in s for k in ['FW','ST','W','LW','RW']): return 'FW'
        return None
    out['Latest_Pos4'] = out['Latest_Pos'].map(pos_bucket) if 'Latest_Pos' in out.columns else None

    # Age features
    age_cols = [c for c in out.columns if c.endswith('_std_Age')]
    if age_cols:
        out['Age_latest'] = out[age_cols[-1]]

    # Minutes features
    out['Minutes_2425'] = pd.to_numeric(out.get('2024-2025_std_Min', pd.Series([None]*len(out))), errors='coerce')
    out['Minutes_2526'] = pd.to_numeric(out.get('2025-2026_std_Min', pd.Series([None]*len(out))), errors='coerce')

    # KPI features
    out['xG_plus_xAG_2425'] = pd.to_numeric(out.get('2024-2025_derived_xG_plus_xAG', pd.Series([None]*len(out))), errors='coerce')
    out['xG_plus_xAG_2526'] = pd.to_numeric(out.get('2025-2026_derived_xG_plus_xAG', pd.Series([None]*len(out))), errors='coerce')

    for season in SEASONS:
        # Progressive passes (total) + per 90
        prog = pd.to_numeric(out.get(f'{season}_pass_Prog', pd.Series([None]*len(out))), errors='coerce')
        nineties = pd.to_numeric(out.get(f'{season}_std_90s', pd.Series([None]*len(out))), errors='coerce')
        out[f'PrgPasses_{season[-2:]}'] = prog
        out[f'PrgPasses_per90_{season[-2:]}'] = (prog / nineties).where((nineties > 0) & prog.notna())

        # Final third touches (total) + per 90
        f3 = pd.to_numeric(out.get(f'{season}_poss_Att 3rd', pd.Series([None]*len(out))), errors='coerce')
        out[f'FinalThirdTouches_{season[-2:]}'] = f3
        out[f'FinalThirdTouches_per90_{season[-2:]}'] = (f3 / nineties).where((nineties > 0) & f3.notna())

    # Final deduplication: ensure only one row per player
    # Sort by Season to prefer 2025-2026 (more recent) data
    if 'Season' in out.columns:
        out['Season_priority'] = out['Season'].map({'2025-2026': 1, '2024-2025': 2})
        out = out.sort_values(['Player', 'Season_priority']).drop_duplicates(subset=['Player'], keep='first')
        out = out.drop(columns=['Season_priority'])
    else:
        out = out.drop_duplicates(subset=['Player'], keep='first')

    # Export
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(OUT_DIR, "como_agecurve_wide.csv"), index=False)
    print("Wrote:", os.path.join(OUT_DIR, "como_agecurve_wide.csv"))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
