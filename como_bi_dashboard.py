"""
Como 1907 Business Intelligence Dashboard
========================================

A comprehensive Streamlit dashboard for Como 1907 stakeholders
consolidating key insights from the EDA analysis.

Wage Data Policy:
- We treat the club-curated manual wage file as the ONLY authoritative source.
- Required columns in manual file:
  ['Player','Position','Age','Country','Gross_PW_EUR','Gross_PY_EUR','Season','Player_Clean']
- Seasons in manual file are 'YYYY-YY' (e.g., '2024-25'); UI shows 'YYYY/YY'.

Author: Iishaan Shekhar
Date: October 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import unicodedata
import re
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Como 1907 Squad Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _normalize_name(s: str) -> str:
    """Deterministic name normalizer (accents/punct/whitespace)."""
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _season_dash_to_slash(s: str) -> str:
    """'2024-25' -> '2024/25' (UI canonical form)."""
    if pd.isna(s): return "ALL"
    s = str(s).strip()
    return s.replace("-", "/") if "-" in s else s

def extract_market_value(value):
    if pd.isna(value) or value == "-":
        return 0.0
    s = str(value).lower().replace("€", "").strip()
    mult = 1.0
    if s.endswith("m"): mult = 1_000_000.0; s = s[:-1]
    elif s.endswith("k"): mult = 1_000.0; s = s[:-1]
    try:
        return float(s) * mult
    except Exception:
        try: return float(s)
        except Exception: return 0.0

def extract_age_from_string(age_str):
    if pd.isna(age_str): return np.nan
    if isinstance(age_str, (int, float)): return float(age_str)
    s = str(age_str)
    if "-" in s:
        return pd.to_numeric(s.split("-")[0], errors="coerce")
    return pd.to_numeric(s, errors="coerce")

def standardize_position(pos_str):
    if pd.isna(pos_str): return "Unknown"
    pos_str = str(pos_str).upper()
    if "GK" in pos_str: return "GK"
    if any(k in pos_str for k in ["DF","DEF","BACK"]): return "DF"
    if any(k in pos_str for k in ["MF","MID"]): return "MF"
    if any(k in pos_str for k in ["FW","FOR","STRIKER"]): return "FW"
    return "Unknown"

# ----------------------------------------------------------------------
# Loaders (hard assumptions)
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Load datasets from project data folder.
    Assumes manual wage file exact schema:
    ['Player','Position','Age','Country','Gross_PW_EUR','Gross_PY_EUR','Season','Player_Clean']
    """
    data_path = Path("data")
    como_agecurve = pd.read_csv(data_path / "como_agecurve_wide.csv")
    fbref_2425 = pd.read_csv(data_path / "intermediate" / "fbref_20242025.csv")
    fbref_2526 = pd.read_csv(data_path / "intermediate" / "fbref_20252026.csv")
    transfermarkt = pd.read_csv(data_path / "intermediate" / "transfermarkt_contracts.csv")

    # Authoritative manual wages (no fuzzy matching; strict schema)
    manual = pd.read_csv(data_path / "intermediate" / "Como_Wage_Breakdown_2425_2526_Cleaned.csv")

    return como_agecurve, fbref_2425, fbref_2526, transfermarkt, manual

def validate_manual_schema(df: pd.DataFrame):
    required = [
        "Player","Position","Age","Country",
        "Gross_PW_EUR","Gross_PY_EUR","Season","Player_Clean"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Manual wage file is missing required columns: {missing}")
        st.stop()

# ----------------------------------------------------------------------
# Wage Master (strict)
# ----------------------------------------------------------------------
def build_wage_master_strict(perf_df: pd.DataFrame,
                             manual_df: pd.DataFrame,
                             season_choice_ui: str) -> pd.DataFrame:
    """
    Strict, no-fuzzy build:
    - Maps Gross_PW_EUR->Weekly_Gross_EUR, Gross_PY_EUR->Yearly_Gross_EUR
    - Season filtered by UI (UI uses 'YYYY/YY', manual uses 'YYYY-YY')
    - Joins perf.Player to manual.Player_Clean (fallback to manual.Player)
    """
    df = perf_df.copy()

    # Ensure Player exists
    if "Player" not in df.columns:
        for cand in ["Name", "FBRef_Name", "player"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "Player"})
                break
    if "Player" not in df.columns:
        st.error("Performance dataset lacks a Player column after normalization.")
        st.stop()

    # Manual schema validation
    validate_manual_schema(manual_df)

    # Normalize seasons for UI (manual stays in dash format internally)
    manual = manual_df.copy()
    manual["Season_UI"] = manual["Season"].apply(_season_dash_to_slash)

    # Filter by season (UI provides '2024/25', '2025/26', or 'ALL')
    if season_choice_ui != "ALL":
        manual = manual[manual["Season_UI"] == season_choice_ui]

    # Map wage columns to the app's expected names (no guessing)
    manual = manual.rename(columns={
        "Gross_PW_EUR": "Weekly_Gross_EUR",
        "Gross_PY_EUR": "Yearly_Gross_EUR"
    })

    # Build join keys: perf on Player; manual on Player_Clean (fallback Player)
    df["join_key"] = df["Player"].apply(_normalize_name)
    manual_key_col = "Player_Clean" if "Player_Clean" in manual.columns else "Player"
    manual["join_key"] = manual[manual_key_col].apply(_normalize_name)

    # Deduplicate manual on (join_key, Season_UI) preferring non-null wages
    manual = (manual.sort_values(by=["Weekly_Gross_EUR","Yearly_Gross_EUR"], ascending=False)
                    .groupby(["join_key","Season_UI"], as_index=False)
                    .agg({
                        manual_key_col: "first",
                        "Weekly_Gross_EUR": "first",
                        "Yearly_Gross_EUR": "first",
                        "Position": "first"
                    }))

    # Merge wages
    df = df.merge(
        manual[["join_key","Weekly_Gross_EUR","Yearly_Gross_EUR"]],
        on="join_key", how="left", validate="m:1"
    )

    return df

# ----------------------------------------------------------------------
# Scoring & Analysis
# ----------------------------------------------------------------------
def create_performance_score(df):
    df = df.copy()
    df["Performance_Score"] = 0.0
    if "Age_latest" in df.columns:
        df["Age_numeric"] = df["Age_latest"].apply(extract_age_from_string)
    if "Minutes_2425" in df.columns:
        max_minutes = max(1, float(df["Minutes_2425"].fillna(0).max()))
        df["Minutes_Score"] = (df["Minutes_2425"].fillna(0) / max_minutes) * 40
        df["Performance_Score"] += df["Minutes_Score"]
    if "Age_numeric" in df.columns:
        bins = []
        for age in df["Age_numeric"]:
            if pd.isna(age): bins.append(0)
            elif 23 <= age <= 30: bins.append(30)
            elif 20 <= age < 23 or 30 < age <= 33: bins.append(20)
            elif age < 20: bins.append(15)
            else: bins.append(10)
        df["Age_Score"] = bins
        df["Performance_Score"] += df["Age_Score"]
    if "MarketValue" in df.columns:
        df["Numeric_Market_Value"] = df["MarketValue"].apply(extract_market_value)
        max_val = max(1.0, float(df["Numeric_Market_Value"].max()))
        df["Market_Value_Score"] = (df["Numeric_Market_Value"] / max_val) * 30
        df["Performance_Score"] += df["Market_Value_Score"]
    return df

def create_wage_analysis(df):
    if "Weekly_Gross_EUR" not in df.columns:
        return "No wage column(s) present"
    wage_data = df.dropna(subset=["Weekly_Gross_EUR"])
    if wage_data.empty:
        return "No wage data available"
    total_weekly = wage_data["Weekly_Gross_EUR"].sum()
    total_yearly = (wage_data["Yearly_Gross_EUR"].sum()
                    if "Yearly_Gross_EUR" in wage_data.columns
                    else total_weekly * 52)
    avg_weekly = wage_data["Weekly_Gross_EUR"].mean()
    median_weekly = wage_data["Weekly_Gross_EUR"].median()
    pos_col = "Position_Standard" if "Position_Standard" in wage_data.columns else ("Position" if "Position" in wage_data.columns else None)
    show_cols = ["Player", "Weekly_Gross_EUR"]
    if pos_col: show_cols.insert(1, pos_col)
    if "Yearly_Gross_EUR" in wage_data.columns: show_cols.append("Yearly_Gross_EUR")
    top_earners = wage_data.nlargest(10, "Weekly_Gross_EUR")[show_cols]
    return {
        "total_players": int(wage_data["Player"].nunique() if "Player" in wage_data.columns else wage_data.shape[0]),
        "total_weekly": total_weekly,
        "total_yearly": total_yearly,
        "avg_weekly": avg_weekly,
        "median_weekly": median_weekly,
        "top_earners": top_earners
    }

# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------
def main():
    st.markdown('<h1 class="main-header">⚽ Como 1907 Squad Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Load
    try:
        with st.spinner("Loading squad data..."):
            como_agecurve, fbref_2425, fbref_2526, transfermarkt, capology_manual = load_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    # Performance base
    df_performance = como_agecurve.copy()
    if "Player" not in df_performance.columns:
        for cand in ["Name", "FBRef_Name", "player"]:
            if cand in df_performance.columns:
                df_performance = df_performance.rename(columns={cand: "Player"})
                break
    if "Player" not in df_performance.columns:
        st.error("Performance dataset lacks a Player column after normalization.")
        st.stop()

    df_performance["Position_Standard"] = df_performance.get("Latest_Pos4", pd.Series(index=df_performance.index)).apply(standardize_position)
    if "Minutes_2425" in df_performance.columns:
        df_performance = df_performance[df_performance["Minutes_2425"].fillna(0) >= 90]
    df_performance = create_performance_score(df_performance)

    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    # Derive available seasons from manual (convert to UI slash form)
    seasons_ui = sorted(capology_manual["Season"].astype(str).map(_season_dash_to_slash).unique().tolist())
    # Ensure we offer a reasonable default order (latest first)
    seasons_ui = sorted(seasons_ui)[::-1]
    season_options = seasons_ui if seasons_ui else ["2024/25","2025/26"]
    season_choice = st.sidebar.selectbox("Wage Season (manual)", options=season_options + ["ALL"], index=0)

    # Join wages (strict, no fuzzy matching)
    try:
        df_perf_with_wages = build_wage_master_strict(
            perf_df=df_performance,
            manual_df=capology_manual,
            season_choice_ui=season_choice
        )
    except Exception as e:
        st.error(f"Error building wage master: {e}")
        st.stop()

    # Filters
    positions = ["All"] + sorted(df_perf_with_wages["Position_Standard"].dropna().unique().tolist())
    selected_position = st.sidebar.selectbox("Select Position", positions)

    if "Age_numeric" in df_perf_with_wages.columns and df_perf_with_wages["Age_numeric"].notna().any():
        age_min = int(np.nanmin(df_perf_with_wages["Age_numeric"]))
        age_max = int(np.nanmax(df_perf_with_wages["Age_numeric"]))
        age_range = st.sidebar.slider("Age Range", min_value=age_min, max_value=age_max, value=(age_min, age_max))
    else:
        age_range = (18, 40)

    mm_max = int(df_perf_with_wages["Minutes_2425"].fillna(0).max()) if "Minutes_2425" in df_perf_with_wages.columns else 0
    min_minutes = st.sidebar.slider("Minimum Minutes", min_value=0, max_value=mm_max, value=0)

    filtered_df = df_perf_with_wages.copy()
    if "Age_numeric" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Age_numeric"].between(age_range[0], age_range[1], inclusive="both")]
    if "Minutes_2425" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Minutes_2425"].fillna(0) >= min_minutes]
    if selected_position != "All":
        filtered_df = filtered_df[filtered_df["Position_Standard"] == selected_position]

    # Wage analysis
    wage_analysis = create_wage_analysis(filtered_df)

    # Tabs
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
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Total Players", len(filtered_df))
        with c2:
            if "Age_numeric" in filtered_df.columns and filtered_df["Age_numeric"].notna().any():
                st.metric("Average Age", f"{filtered_df['Age_numeric'].mean():.1f} years")
            else:
                st.metric("Average Age", "N/A")
        with c3:
            if "Minutes_2425" in filtered_df.columns:
                st.metric("Total Minutes", f"{filtered_df['Minutes_2425'].sum():,.0f}")
            else:
                st.metric("Total Minutes", "N/A")
        with c4: st.metric("Avg Performance Score", f"{filtered_df['Performance_Score'].mean():.1f}")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Position Distribution")
            pos_counts = filtered_df["Position_Standard"].value_counts()
            fig = px.pie(values=pos_counts.values, names=pos_counts.index, title="Players by Position")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Age Distribution")
            if "Age_numeric" in filtered_df.columns and filtered_df["Age_numeric"].notna().any():
                fig = px.histogram(filtered_df, x="Age_numeric", nbins=15, title="Age Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Age data not available")

        st.subheader("Top Performers")
        display_cols = ["Player", "Position_Standard", "Minutes_2425", "Performance_Score"]
        if "Age_numeric" in filtered_df.columns: display_cols.insert(2, "Age_numeric")
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
                fig = px.scatter(filtered_df, x="Age_numeric", y="Minutes_2425",
                                 color="Position_Standard", size="Performance_Score",
                                 title="Minutes Played vs Age")
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
                        if "Age_numeric" in pos_data.columns and pos_data["Age_numeric"].notna().any():
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
                    if "Age_numeric" in pos_data.columns: display_cols.insert(1, "Age_numeric")
                    top_pos = pos_data.nlargest(5, "Performance_Score")[display_cols]
                    st.dataframe(top_pos, use_container_width=True)

    with tab3:
        st.markdown('<h2 class="section-header">Salary Analysis (Manual Source)</h2>', unsafe_allow_html=True)
        st.caption("Source: club-curated manual file. Columns: Gross_PW_EUR (weekly), Gross_PY_EUR (yearly).")
        if isinstance(wage_analysis, str):
            st.warning(f"💰 {wage_analysis}. Check the manual CSV, season filter, or join keys.")
            with st.expander("Wage Audit (first 25 rows)"):
                cols = [c for c in ["Player","Position_Standard","Weekly_Gross_EUR","Yearly_Gross_EUR"] if c in filtered_df.columns]
                if cols:
                    st.dataframe(filtered_df[cols].head(25), use_container_width=True)
                # Show unmatched: players with no weekly wage
                if "Weekly_Gross_EUR" in filtered_df.columns:
                    unmatched = filtered_df[filtered_df["Weekly_Gross_EUR"].isna()][["Player","Position_Standard"]].head(25)
                    if not unmatched.empty:
                        st.write("Players without matched wages (top 25):")
                        st.dataframe(unmatched, use_container_width=True)
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Players with Wages", f"{wage_analysis['total_players']}")
            with c2: st.metric("Weekly Wage Bill", f"€{wage_analysis['total_weekly']:,.0f}")
            with c3: st.metric("Annual Wage Bill", f"€{wage_analysis['total_yearly']:,.0f}")
            with c4: st.metric("Avg Weekly Wage", f"€{wage_analysis['avg_weekly']:,.0f}")

            st.subheader("🏆 Top Earners")
            te = wage_analysis["top_earners"].copy()
            if "Weekly_Gross_EUR" in te.columns:
                te["Weekly Gross EUR"] = te["Weekly_Gross_EUR"].apply(lambda x: f"€{x:,.0f}")
                te.drop(columns=["Weekly_Gross_EUR"], inplace=True)
            if "Yearly_Gross_EUR" in te.columns:
                te["Yearly Gross EUR"] = te["Yearly_Gross_EUR"].apply(lambda x: f"€{x:,.0f}")
                te.drop(columns=["Yearly_Gross_EUR"], inplace=True)
            st.dataframe(te, use_container_width=True)

            st.subheader("📊 Average Weekly Salary by Position")
            wage_filtered = filtered_df.dropna(subset=["Weekly_Gross_EUR"])
            if not wage_filtered.empty:
                salary_stats = wage_filtered.groupby("Position_Standard")["Weekly_Gross_EUR"].agg(["count","mean","median"]).round(0)
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
                st.info("No wage data available after filters.")

    with tab4:
        st.markdown('<h2 class="section-header">Key Insights</h2>', unsafe_allow_html=True)
        st.subheader("Squad Balance Analysis")
        c1, c2 = st.columns(2)
        with c1:
            if "Age_numeric" in filtered_df.columns and filtered_df["Age_numeric"].notna().any():
                young = (filtered_df["Age_numeric"] < 23).sum()
                prime = filtered_df["Age_numeric"].between(23, 30, inclusive="both").sum()
                vet = (filtered_df["Age_numeric"] > 30).sum()
            else:
                young = prime = vet = 0
            age_data = pd.DataFrame({"Category": ["Young (<23)","Prime (23-30)","Veteran (>30)"],
                                     "Count": [young, prime, vet]})
            fig = px.pie(age_data, values="Count", names="Category", title="Age Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            pos_data = filtered_df["Position_Standard"].value_counts().reset_index()
            pos_data.columns = ["Position", "Count"]
            fig = px.bar(pos_data, x="Position", y="Count", title="Position Distribution")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Key Insights")
        if not filtered_df.empty:
            insights = [
                f"📊 **Squad Size**: {len(filtered_df)} players analyzed",
                f"⏱️ **Total Minutes**: {int(filtered_df.get('Minutes_2425', pd.Series(dtype=float)).sum()):,} minutes played",
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
        else:
            st.info("No players left after filters.")

    with tab5:
        st.markdown('<h2 class="section-header">Strategic Recommendations</h2>', unsafe_allow_html=True)
        st.subheader("Squad Management")
        recs = [
            "🎯 **Starting XI**: Bias to highest performance scores within prime age windows.",
            "🔄 **Rotation**: Manage high-minute players; flatten fatigue spikes across congested periods.",
            "📈 **Development**: Ring-fence minutes & IP for U23 prospects with upward trend lines.",
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
                    else:
                        st.write("📊 **Analysis Phase**: Optimize performance inputs and role clarity.")

    with tab6:
        st.markdown('<h2 class="section-header">Detailed Reports</h2>', unsafe_allow_html=True)
        st.subheader("Player Comparison")
        player_options = filtered_df["Player"].dropna().tolist()
        default_sel = player_options[:3] if len(player_options) >= 3 else player_options
        selected_players = st.multiselect("Select players to compare", player_options, default=default_sel)
        if selected_players:
            display_cols = ["Player", "Position_Standard", "Performance_Score"]
            if "Age_numeric" in filtered_df.columns: display_cols.insert(2, "Age_numeric")
            if "Minutes_2425" in filtered_df.columns: display_cols.append("Minutes_2425")
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

# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Unhandled error: {e}")
        st.stop()
