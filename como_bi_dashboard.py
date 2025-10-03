"""
Como 1907 Business Intelligence Dashboard
========================================

A comprehensive Streamlit dashboard for Como 1907 stakeholders
consolidating key insights from the EDA analysis.

Wage Data Policy (read me):
- Capology's live pages are unstable (layout shifts, net vs gross ambiguity, duplicate rows).
- We treat 'capology_manual' (club-curated CSV) as the *only* authoritative wage source.
- Optional fallback to scraped Capology is available behind a flag, but disabled by default.

Author: Iishaan Shekhar
Date: October 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # kept for parity; not strictly required
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import unicodedata
import re
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Como 1907 Squad Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS for professional styling
# -----------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2980b9;
        margin: 1rem 0;
    }
    .recommendation-box {
        background-color: #e8f8f5;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #27ae60;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; white-space: pre-wrap; background-color: #f0f2f6;
        border-radius: 4px 4px 0 0; gap: 1px; padding: 0 20px;
    }
    .stTabs [aria-selected="true"] { background-color: #1f77b4; color: white; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
def _normalize_name(s: str) -> str:
    """
    Robust player name normalizer:
    - lowercases
    - strips accents
    - removes punctuation & extra spaces
    """
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _pick_first_nonnull(series: pd.Series):
    """Return first non-null value in a Series or np.nan"""
    for v in series:
        if pd.notna(v):
            return v
    return np.nan

def _coerce_numeric(x):
    """Convert common currency strings to float (EUR). Accepts 1,234 or '€1,234'."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace("€", "").replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return np.nan

def normalize_manual_wages(df):
    df = df.copy()
    rename_map = {
        'Weekly EUR':'Weekly_Gross_EUR',
        'Weekly':'Weekly_Gross_EUR',
        'Yearly EUR':'Yearly_Gross_EUR',
        'Annual':'Yearly_Gross_EUR',
    }
    for k,v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df = df.rename(columns={k:v})
    for col in ['Weekly_Gross_EUR','Yearly_Gross_EUR']:
        if col in df.columns:
            df[col] = df[col].apply(_coerce_numeric)
    return df

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_data():
    """
    Load datasets. We DO load scraped capology for optional debugging/fallback,
    but we DO NOT trust or use it unless the user toggles fallback on.
    """
    data_path = Path("data")

    como_agecurve = pd.read_csv(data_path / "como_agecurve_wide.csv")
    fbref_2425 = pd.read_csv(data_path / "intermediate" / "fbref_20242025.csv")
    fbref_2526 = pd.read_csv(data_path / "intermediate" / "fbref_20252026.csv")
    transfermarkt = pd.read_csv(data_path / "intermediate" / "transfermarkt_contracts.csv")

    # Scraped capology (UNRELIABLE) – loaded, but not used unless fallback enabled
    try:
        capology_scraped = pd.read_csv(data_path / "intermediate" / "capology_wages.csv")
    except Exception:
        capology_scraped = pd.DataFrame()

    # Authoritative manual wage file (club-curated; can contain 24/25 and 25/26)
    capology_manual = pd.read_csv(
        data_path / "intermediate" / "Como_Wage_Breakdown_2425_2526_Cleaned.csv"
    )

    return como_agecurve, fbref_2425, fbref_2526, transfermarkt, capology_scraped, capology_manual

# -----------------------------
# Domain transforms
# -----------------------------
def standardize_position(pos_str):
    if pd.isna(pos_str):
        return "Unknown"
    pos_str = str(pos_str).upper()
    if "GK" in pos_str:
        return "GK"
    elif "DF" in pos_str or "DEF" in pos_str or "BACK" in pos_str:
        return "DF"
    elif "MF" in pos_str or "MID" in pos_str:
        return "MF"
    elif "FW" in pos_str or "FOR" in pos_str or "STRIKER" in pos_str:
        return "FW"
    else:
        return "Unknown"

def extract_market_value(value):
    if pd.isna(value) or value == "-":
        return 0.0
    s = str(value).lower().replace("€", "").strip()
    mult = 1.0
    if s.endswith("m"):
        mult = 1_000_000.0
        s = s[:-1]
    elif s.endswith("k"):
        mult = 1_000.0
        s = s[:-1]
    try:
        return float(s) * mult
    except Exception:
        # Accept "1.2" as millions if input was like "1.2m" earlier
        try:
            return float(s)
        except Exception:
            return 0.0

def extract_age_from_string(age_str):
    if pd.isna(age_str):
        return np.nan
    if isinstance(age_str, (int, float)):
        return float(age_str)
    s = str(age_str)
    if "-" in s:
        return pd.to_numeric(s.split("-")[0], errors="coerce")
    return pd.to_numeric(s, errors="coerce")

# -----------------------------
# Wage master builder (authoritative)
# -----------------------------
def build_wage_master(perf_df: pd.DataFrame,
                      manual_df: pd.DataFrame,
                      scraped_df: pd.DataFrame,
                      season_label: str,
                      enable_fallback_to_scraped: bool = False) -> pd.DataFrame:
    """
    Join authoritative wages onto performance frame using name keys.
    Priority: capology_manual ONLY. If a player is missing there and the
    fallback flag is True, we *then* consider scraped capology.

    Expected manual_df columns (flexible, we infer):
      - Player or Name
      - Season (e.g., '2024/25', '2025/26' OR '2425'/'2526' OR '24-25'/'25-26')
      - Weekly_Gross_EUR, Yearly_Gross_EUR
      - Position (optional)
    """

    df = perf_df.copy()

    # Create robust join keys
    name_cols_perf = [c for c in df.columns if c.lower() in {"player", "name", "fbref_name"}]
    perf_name_col = name_cols_perf[0] if name_cols_perf else "Player"
    df["join_key"] = df[perf_name_col].apply(_normalize_name)

    # Manual wage frame normalization
    m = manual_df.copy()
    # Identify columns
    manual_name_col = None
    for cand in ["Player", "Name", "player_name", "Player_Name"]:
        if cand in m.columns:
            manual_name_col = cand
            break
    if manual_name_col is None:
        # Last resort: pick first object column
        manual_name_col = m.select_dtypes(include="object").columns[0]

    # Season parsing – accept many shapes
    if "Season" not in m.columns:
        # Try to infer from file or columns
        # If absent, we will assign 'ALL' and ignore season filtering
        m["Season"] = "ALL"

    # Normalize season label provided by the UI (e.g., "2024/25", "2025/26", "ALL")
    def _norm_season(s):
        if pd.isna(s):
            return "ALL"
        s = str(s).strip()
        s = s.replace(" ", "").replace("-", "/")
        # Map a few common patterns
        if re.fullmatch(r"\d{4}/\d{2}", s):
            return s  # e.g., 2024/25
        if re.fullmatch(r"\d{2}/\d{2}", s):
            return f"20{s[:2]}/" + s[-2:]
        if re.fullmatch(r"\d{2}\d{2}", s):  # 2425 -> 2024/25
            return f"20{s[:2]}/" + s[-2:]
        return s

    m["Season_norm"] = m["Season"].apply(_norm_season)
    season_label_norm = _norm_season(season_label)

    # Build join keys & coerce numerics
    m["join_key"] = m[manual_name_col].apply(_normalize_name)
    # Standardize wage columns
    if "Weekly_Gross_EUR" not in m.columns and "Weekly EUR" in m.columns:
        m = m.rename(columns={"Weekly EUR": "Weekly_Gross_EUR"})
    if "Yearly_Gross_EUR" not in m.columns and "Yearly EUR" in m.columns:
        m = m.rename(columns={"Yearly EUR": "Yearly_Gross_EUR"})

    for col in ["Weekly_Gross_EUR", "Yearly_Gross_EUR"]:
        if col in m.columns:
            m[col] = m[col].apply(_coerce_numeric)

    # Deduplicate manual rows by (join_key, Season_norm) preferring non-null wages
    m = (
        m.sort_values(by=["Weekly_Gross_EUR", "Yearly_Gross_EUR"], ascending=False)
         .groupby(["join_key", "Season_norm"], as_index=False)
         .agg({
             manual_name_col: "first",
             "Weekly_Gross_EUR": "first",
             "Yearly_Gross_EUR": "first",
             "Position": _pick_first_nonnull if "Position" in m.columns else (lambda x: np.nan)
         })
    )

    # Filter to season
    if season_label_norm != "ALL" and "Season_norm" in m.columns:
        m = m[m["Season_norm"] == season_label_norm]

    # Left join onto performance
    df = df.merge(
        m[["join_key", "Weekly_Gross_EUR", "Yearly_Gross_EUR"]],
        on="join_key",
        how="left",
        validate="m:1"
    )

    # Optional fallback to scraped capology for missing wages
    if enable_fallback_to_scraped and isinstance(scraped_df, pd.DataFrame) and not scraped_df.empty:
        s = scraped_df.copy()
        # Try to find a name column
        scraped_name_col = None
        for cand in ["Player", "Name", "player_name"]:
            if cand in s.columns:
                scraped_name_col = cand
                break
        if scraped_name_col is None and len(s.select_dtypes(include="object").columns) > 0:
            scraped_name_col = s.select_dtypes(include="object").columns[0]
        if scraped_name_col is not None:
            s["join_key"] = s[scraped_name_col].apply(_normalize_name)
            # Try to detect season
            if "Season" in s.columns:
                s["Season_norm"] = s["Season"].apply(_norm_season)
                if season_label_norm != "ALL":
                    s = s[s["Season_norm"] == season_label_norm]
            # Coerce numeric col names
            for col_guess in ["Weekly_Gross_EUR", "Weekly EUR", "Weekly"]:
                if col_guess in s.columns:
                    s["Weekly_Fallback"] = s[col_guess].apply(_coerce_numeric)
                    break
            for col_guess in ["Yearly_Gross_EUR", "Annual", "Yearly EUR", "Yearly"]:
                if col_guess in s.columns:
                    s["Yearly_Fallback"] = s[col_guess].apply(_coerce_numeric)
                    break
            s = s.groupby("join_key", as_index=False).agg({
                "Weekly_Fallback": "max",
                "Yearly_Fallback": "max"
            })
            # Fill only where manual is missing
            for base, fb in [("Weekly_Gross_EUR", "Weekly_Fallback"),
                             ("Yearly_Gross_EUR", "Yearly_Fallback")]:
                if fb in s.columns:
                    df = df.merge(s[["join_key", fb]], on="join_key", how="left")
                    df[base] = np.where(df[base].isna(), df[fb], df[base])
                    df.drop(columns=[fb], inplace=True, errors="ignore")

    return df

# -----------------------------
# KPIs & Scoring
# -----------------------------
def create_wage_analysis(df):
    required = ['Weekly_Gross_EUR']
    if not all(col in df.columns for col in required):
        return "No wage column(s) present"
    wage_data = df.dropna(subset=['Weekly_Gross_EUR'])
    if wage_data.empty:
        return "No wage data available"
    total_weekly = wage_data['Weekly_Gross_EUR'].sum()
    total_yearly = (wage_data['Yearly_Gross_EUR'].sum()
                    if 'Yearly_Gross_EUR' in wage_data.columns
                    else total_weekly * 52)
    return {
        'total_players': int(wage_data.shape[0]),
        'total_weekly': total_weekly,
        'total_yearly': total_yearly,
        'avg_weekly': wage_data['Weekly_Gross_EUR'].mean(),
        'median_weekly': wage_data['Weekly_Gross_EUR'].median(),
        'top_earners': wage_data.nlargest(
            10, 'Weekly_Gross_EUR'
        )[[c for c in ['Player','Position','Position_Standard',
                       'Weekly_Gross_EUR','Yearly_Gross_EUR'] if c in wage_data.columns]]
    }


def create_performance_score(df):
    df = df.copy()
    df["Performance_Score"] = 0.0
    if "Age_latest" in df.columns:
        df["Age_numeric"] = df["Age_latest"].apply(extract_age_from_string)
    if "Minutes_2425" in df.columns:
        max_minutes = max(1, float(df["Minutes_2425"].max()))
        df["Minutes_Score"] = (df["Minutes_2425"] / max_minutes) * 40
        df["Performance_Score"] += df["Minutes_Score"]
    if "Age_numeric" in df.columns:
        bins = []
        for age in df["Age_numeric"]:
            if pd.isna(age):
                bins.append(0)
            elif 23 <= age <= 30:
                bins.append(30)
            elif 20 <= age < 23 or 30 < age <= 33:
                bins.append(20)
            elif age < 20:
                bins.append(15)
            else:
                bins.append(10)
        df["Age_Score"] = bins
        df["Performance_Score"] += df["Age_Score"]
    if "MarketValue" in df.columns:
        df["Numeric_Market_Value"] = df["MarketValue"].apply(extract_market_value)
        max_val = max(1.0, float(df["Numeric_Market_Value"].max()))
        df["Market_Value_Score"] = (df["Numeric_Market_Value"] / max_val) * 30
        df["Performance_Score"] += df["Market_Value_Score"]
    return df

# -----------------------------
# App
# -----------------------------
def main():
    st.markdown('<h1 class="main-header">⚽ Como 1907 Squad Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")

    with st.spinner("Loading squad data..."):
        como_agecurve, fbref_2425, fbref_2526, transfermarkt, capology_scraped, capology_manual = load_data()

    # Process performance base
    df_performance = como_agecurve.copy()
    capology_manual = normalize_manual_wages(capology_manual)

    # build a robust join key (your normalizer)
    def _norm_name(s):
        s = '' if pd.isna(s) else str(s).strip().lower()
        s = unicodedata.normalize("NFKD", s)
        s = ''.join(ch for ch in s if not unicodedata.combining(ch))
        return re.sub(r'[^a-z\s]',' ', s).split()
        # or return a cleaned string; you just need consistency on both sides

    df_performance['join_key'] = df_performance['Player'].apply(lambda s: ' '.join(_norm_name(s)))

    name_col = 'Player' if 'Player' in capology_manual.columns else 'Name'
    capology_manual['join_key'] = capology_manual[name_col].apply(lambda s: ' '.join(_norm_name(s)))

    # (optional) filter manual by season here if your file mixes seasons
    # capology_manual = capology_manual[capology_manual['Season'] == '2024/25']

    wage_cols = [c for c in ['Weekly_Gross_EUR','Yearly_Gross_EUR'] if c in capology_manual.columns]
    df_performance_with_wages = df_performance.merge(
        capology_manual[['join_key'] + wage_cols],
        on='join_key', how='left', validate='m:1'
    )

    wage_analysis = create_wage_analysis(df_performance_with_wages)


    # Ensure Player column exists
    if "Player" not in df_performance.columns:
        # try a best-effort rename
        for cand in ["Name", "FBRef_Name", "player"]:
            if cand in df_performance.columns:
                df_performance = df_performance.rename(columns={cand: "Player"})
                break

    df_performance["Position_Standard"] = df_performance.get("Latest_Pos4", pd.Series(index=df_performance.index)).apply(standardize_position)
    if "Minutes_2425" in df_performance.columns:
        df_performance = df_performance[df_performance["Minutes_2425"].fillna(0) >= 90]  # Filter for sufficient minutes
    df_performance = create_performance_score(df_performance)

    # -----------------------------
    # Sidebar controls
    # -----------------------------
    st.sidebar.title("📊 Dashboard Controls")

    # Season selector – drives wage join
    # Offer common variants; 'ALL' keeps all manual wages (useful for squad-wide view)
    season_choice = st.sidebar.selectbox(
        "Wage Season (capology_manual)",
        options=["2024/25", "2025/26", "ALL"],
        index=0
    )
    enable_fallback = st.sidebar.toggle("Allow fallback to scraped Capology for missing wages (not recommended)", value=False,
                                        help="Disabled by default to avoid polluting curated wages with noisy scraped values.")

    # Build authoritative wage master and join
    df_perf_with_wages = build_wage_master(
        perf_df=df_performance,
        manual_df=capology_manual,
        scraped_df=capology_scraped,
        season_label=season_choice,
        enable_fallback_to_scraped=enable_fallback
    )

    # Filters
    positions = ["All"] + sorted(df_perf_with_wages["Position_Standard"].dropna().unique().tolist())
    selected_position = st.sidebar.selectbox("Select Position", positions)

    if "Age_numeric" in df_perf_with_wages.columns and df_perf_with_wages["Age_numeric"].notna().any():
        age_min = int(np.nanmin(df_perf_with_wages["Age_numeric"]))
        age_max = int(np.nanmax(df_perf_with_wages["Age_numeric"]))
        age_range = st.sidebar.slider("Age Range", min_value=age_min, max_value=age_max, value=(age_min, age_max))
    else:
        age_range = (18, 40)

    min_minutes = int(df_perf_with_wages["Minutes_2425"].max()) if "Minutes_2425" in df_perf_with_wages.columns else 0
    min_minutes = st.sidebar.slider("Minimum Minutes", min_value=0, max_value=min_minutes, value=0)

    # Apply filters
    filtered_df = df_perf_with_wages.copy()
    if "Age_numeric" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["Age_numeric"].between(age_range[0], age_range[1], inclusive="both")
        ]
    if "Minutes_2425" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Minutes_2425"].fillna(0) >= min_minutes]
    if selected_position != "All":
        filtered_df = filtered_df[filtered_df["Position_Standard"] == selected_position]

    # Wage analysis (authoritative)
    wage_analysis = create_wage_analysis(filtered_df)

    # -----------------------------
    # Tabs
    # -----------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Squad Overview",
        "⚽ Performance Analysis",
        "💰 Salary Analysis",
        "📈 Key Insights",
        "🎯 Recommendations",
        "📋 Detailed Reports"
    ])

    with tab1:
        st.markdown('<h2 class="section-header">Squad Overview</h2>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Players", len(filtered_df))
        with col2:
            if "Age_numeric" in filtered_df.columns:
                st.metric("Average Age", f"{filtered_df['Age_numeric'].mean():.1f} years")
            else:
                st.metric("Average Age", "N/A")
        with col3:
            if "Minutes_2425" in filtered_df.columns:
                st.metric("Total Minutes", f"{filtered_df['Minutes_2425'].sum():,.0f}")
            else:
                st.metric("Total Minutes", "N/A")
        with col4:
            st.metric("Avg Performance Score", f"{filtered_df['Performance_Score'].mean():.1f}")

        # Position distribution & Age distribution
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Position Distribution")
            pos_counts = filtered_df["Position_Standard"].value_counts()
            fig = px.pie(values=pos_counts.values, names=pos_counts.index, title="Players by Position")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Age Distribution")
            if "Age_numeric" in filtered_df.columns:
                fig = px.histogram(filtered_df, x="Age_numeric", nbins=15, title="Age Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Age data not available")

        # Top performers
        st.subheader("Top Performers")
        display_cols = ["Player", "Position_Standard", "Minutes_2425", "Performance_Score"]
        if "Age_numeric" in filtered_df.columns:
            display_cols.insert(2, "Age_numeric")
        top_performers = filtered_df.nlargest(10, "Performance_Score")[display_cols]
        st.dataframe(top_performers, use_container_width=True)

    with tab2:
        st.markdown('<h2 class="section-header">Performance Analysis</h2>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Performance Score by Position")
            pos_perf = filtered_df.groupby("Position_Standard")["Performance_Score"].mean().reset_index()
            fig = px.bar(pos_perf, x="Position_Standard", y="Performance_Score", title="Average Performance Score by Position")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Minutes vs Age")
            if "Age_numeric" in filtered_df.columns and "Minutes_2425" in filtered_df.columns:
                fig = px.scatter(
                    filtered_df,
                    x="Age_numeric", y="Minutes_2425",
                    color="Position_Standard", size="Performance_Score",
                    title="Minutes Played vs Age"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Age or minutes data not available")

        st.subheader("Position-Specific Analysis")
        for pos in ["GK", "DF", "MF", "FW"]:
            pos_data = filtered_df[filtered_df["Position_Standard"] == pos]
            if len(pos_data) > 0:
                with st.expander(f"{pos} - {len(pos_data)} players"):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if "Age_numeric" in pos_data.columns:
                            st.metric("Average Age", f"{pos_data['Age_numeric'].mean():.1f}")
                        else:
                            st.metric("Average Age", "N/A")
                    with c2:
                        if "Minutes_2425" in pos_data.columns:
                            st.metric("Total Minutes", f"{pos_data['Minutes_2425'].sum():,.0f}")
                        else:
                            st.metric("Total Minutes", "N/A")
                    with c3:
                        st.metric("Avg Performance", f"{pos_data['Performance_Score'].mean():.1f}")

                    display_cols = ["Player", "Minutes_2425", "Performance_Score"]
                    if "Age_numeric" in pos_data.columns:
                        display_cols.insert(1, "Age_numeric")
                    top_pos = pos_data.nlargest(5, "Performance_Score")[display_cols]
                    st.dataframe(top_pos, use_container_width=True)

    with tab3:
        st.markdown('<h2 class="section-header">Salary Analysis (Manual Source)</h2>', unsafe_allow_html=True)
        st.caption("Source: club-curated `Como_Wage_Breakdown_2425_2526_Cleaned.csv`. Scraped Capology is ignored unless you explicitly enable fallback in the sidebar.")
        if isinstance(wage_analysis, dict):
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Players with Wages", f"{wage_analysis['total_players']}")
            with c2:
                st.metric("Weekly Wage Bill", f"€{wage_analysis['total_weekly']:,.0f}")
            with c3:
                st.metric("Annual Wage Bill", f"€{wage_analysis['total_yearly']:,.0f}")
            with c4:
                st.metric("Avg Weekly Wage", f"€{wage_analysis['avg_weekly']:,.0f}")

            # Top earners
            st.subheader("🏆 Top Earners")
            te = wage_analysis["top_earners"].copy()
            if "Weekly_Gross_EUR" in te.columns:
                te["Weekly Gross EUR"] = te["Weekly_Gross_EUR"].apply(lambda x: f"€{x:,.0f}")
                te.drop(columns=["Weekly_Gross_EUR"], inplace=True)
            if "Yearly_Gross_EUR" in te.columns:
                te["Yearly Gross EUR"] = te["Yearly_Gross_EUR"].apply(lambda x: f"€{x:,.0f}")
                te.drop(columns=["Yearly_Gross_EUR"], inplace=True)
            st.dataframe(te, use_container_width=True)

            # Salary by position
            st.subheader("📊 Average Weekly Salary by Position")
            wage_filtered = filtered_df.dropna(subset=["Weekly_Gross_EUR"])
            if not wage_filtered.empty:
                salary_stats = wage_filtered.groupby("Position_Standard")["Weekly_Gross_EUR"].agg(["count", "mean", "median"]).round(0)
                salary_display = pd.DataFrame({
                    "Count": salary_stats["count"],
                    "Average": salary_stats["mean"].apply(lambda x: f"€{x:,.0f}"),
                    "Median": salary_stats["median"].apply(lambda x: f"€{x:,.0f}")
                })
                st.dataframe(salary_display, use_container_width=True)

                st.subheader("📈 Weekly Salary Distribution")
                fig = px.histogram(wage_filtered, x="Weekly_Gross_EUR",
                                   title="Weekly Salary Distribution",
                                   labels={"Weekly_Gross_EUR": "Weekly Salary (EUR)", "count": "Number of Players"})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("💰 Wage data not available. Check the manual CSV or the join keys.")

    with tab4:
        st.markdown('<h2 class="section-header">Key Insights</h2>', unsafe_allow_html=True)
        st.subheader("Squad Balance Analysis")
        c1, c2 = st.columns(2)
        with c1:
            if "Age_numeric" in filtered_df.columns:
                young = (filtered_df["Age_numeric"] < 23).sum()
                prime = filtered_df["Age_numeric"].between(23, 30, inclusive="both").sum()
                vet = (filtered_df["Age_numeric"] > 30).sum()
            else:
                young = prime = vet = 0
            age_data = pd.DataFrame({"Category": ["Young (<23)", "Prime (23-30)", "Veteran (>30)"],
                                     "Count": [young, prime, vet]})
            fig = px.pie(age_data, values="Count", names="Category", title="Age Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            pos_data = filtered_df["Position_Standard"].value_counts().reset_index()
            pos_data.columns = ["Position", "Count"]
            fig = px.bar(pos_data, x="Position", y="Count", title="Position Distribution")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Key Insights")
        insights = [
            f"📊 **Squad Size**: {len(filtered_df)} players analyzed",
            f"⏱️ **Total Minutes**: {int(filtered_df.get('Minutes_2425', pd.Series()).sum()):,} minutes played",
            f"🏆 **Top Performer**: {filtered_df.loc[filtered_df['Performance_Score'].idxmax(), 'Player']}"
        ]
        if "Age_numeric" in filtered_df.columns and filtered_df["Age_numeric"].notna().any():
            insights.insert(1, f"🎂 **Average Age**: {filtered_df['Age_numeric'].mean():.1f} years")
            youngest = filtered_df.loc[filtered_df["Age_numeric"].idxmin()]
            oldest = filtered_df.loc[filtered_df["Age_numeric"].idxmax()]
            insights.extend([
                f"🌟 **Youngest Player**: {youngest['Player']} ({youngest['Age_numeric']:.1f} years)",
                f"👴 **Most Experienced**: {oldest['Player']} ({oldest['Age_numeric']:.1f} years)"
            ])
        for i in insights:
            st.markdown(f'<div class="insight-box">{i}</div>', unsafe_allow_html=True)

    with tab5:
        st.markdown('<h2 class="section-header">Strategic Recommendations</h2>', unsafe_allow_html=True)
        st.subheader("Squad Management")
        recs = [
            "🎯 **Starting XI**: Bias to highest performance scores within prime age windows.",
            "🔄 **Rotation**: Manage high-minute players; aim to flatten fatigue spikes across congested periods.",
            "📈 **Development**: Ring-fence minutes and IP for U23 prospects with upward trend lines.",
            "👥 **Leadership**: Assign mentoring responsibilities to >30 cohort; formalize handovers.",
            "⚖️ **Balance**: Maintain GK/DF/MF/FW numbers and wage share within pre-agreed caps."
        ]
        for r in recs:
            st.markdown(f'<div class="recommendation-box">{r}</div>', unsafe_allow_html=True)

        st.subheader("Position-Specific Recommendations")
        for pos in ["GK", "DF", "MF", "FW"]:
            pos_data = filtered_df[filtered_df["Position_Standard"] == pos]
            if len(pos_data) > 0:
                with st.expander(f"{pos} Recommendations"):
                    top_player = pos_data.loc[pos_data["Performance_Score"].idxmax()]
                    st.write(f"**Top Performer**: {top_player['Player']} (Score: {top_player['Performance_Score']:.1f})")
                    if "Age_numeric" in pos_data.columns and pos_data["Age_numeric"].notna().any():
                        avg_age = pos_data["Age_numeric"].mean()
                        if avg_age < 25:
                            st.write("🔵 **Development Phase**: Emphasize technical/tactical reps & managed exposure.")
                        elif avg_age > 30:
                            st.write("🟡 **Experience Phase**: Leverage leadership; succession plan active.")
                        else:
                            st.write("🟢 **Prime Phase**: Optimize for consistency & combinations.")

    with tab6:
        st.markdown('<h2 class="section-header">Detailed Reports</h2>', unsafe_allow_html=True)
        st.subheader("Player Comparison")
        player_options = filtered_df["Player"].dropna().tolist()
        default_sel = player_options[:3] if len(player_options) >= 3 else player_options
        selected_players = st.multiselect("Select players to compare", player_options, default=default_sel)

        if selected_players:
            display_cols = ["Player", "Position_Standard", "Performance_Score"]
            if "Age_numeric" in filtered_df.columns:
                display_cols.insert(2, "Age_numeric")
            if "Minutes_2425" in filtered_df.columns:
                display_cols.insert(len(display_cols), "Minutes_2425")
            comparison_df = filtered_df[filtered_df["Player"].isin(selected_players)][display_cols]
            st.dataframe(comparison_df, use_container_width=True)
            fig = px.bar(comparison_df, x="Player", y="Performance_Score", title="Performance Score Comparison")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Export Data")
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv,
            file_name=f"como_squad_analysis_{season_choice}_{selected_position}_{age_range[0]}-{age_range[1]}.csv",
            mime="text/csv"
        )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>Como 1907 Squad Analysis Dashboard | Data Analysis Team | 2024–2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
