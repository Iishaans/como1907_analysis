"""
Como 1907 Squad Analysis Dashboard
==================================

Streamlit business intelligence workspace for FC Como 1907 stakeholders.
Combines performance, wage, and market context to support strategic decisions.

Author: Iishaan Shekhar (updated by Codex assistant)
Date: October 2025
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Page configuration & styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Como 1907 Squad Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
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
    .insight-box {
        background-color: #e8f4fd;
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 5px solid #2980b9;
        margin: 0.75rem 0;
    }
    .recommendation-box {
        background-color: #e8f8f5;
        padding: 1.25rem;
        border-radius: 10px;
        border-left: 5px solid #27ae60;
        margin: 0.75rem 0;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data loading & preparation helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> Dict[str, pd.DataFrame]:
    """Load all required Como 1907 datasets from disk."""
    data_path = Path("data")
    intermediate = data_path / "intermediate"

    datasets: Dict[str, pd.DataFrame] = {}
    datasets["como_agecurve"] = pd.read_csv(data_path / "como_agecurve_wide.csv")
    datasets["fbref_2425"] = pd.read_csv(intermediate / "fbref_20242025.csv")
    datasets["fbref_2526"] = pd.read_csv(intermediate / "fbref_20252026.csv")
    datasets["transfermarkt"] = pd.read_csv(intermediate / "transfermarkt_contracts.csv")

    manual_path = intermediate / "Como_Wage_Breakdown_2425_2526_Cleaned.csv"
    if manual_path.exists():
        datasets["capology_manual"] = pd.read_csv(manual_path)
    else:
        datasets["capology_manual"] = pd.DataFrame()

    capology_path = intermediate / "capology_wages.csv"
    if capology_path.exists():
        datasets["capology_raw"] = pd.read_csv(capology_path)
    else:
        datasets["capology_raw"] = pd.DataFrame()

    return datasets


def standardize_position(pos_str: str) -> str:
    """Coerce free-text positional tags into standard lines."""
    if pd.isna(pos_str) or str(pos_str).strip() == "":
        return "Unknown"

    pos_str = str(pos_str).upper()
    if "GK" in pos_str:
        return "GK"
    if any(key in pos_str for key in ("DF", "DEF", "BACK")):
        return "DF"
    if any(key in pos_str for key in ("MF", "MID")):
        return "MF"
    if any(key in pos_str for key in ("FW", "FOR", "STRIKER")):
        return "FW"
    return "Unknown"


def extract_age_from_string(age_value) -> float:
    """Extract numeric age (years) from FBRef string formats like '29-170'."""
    if pd.isna(age_value):
        return np.nan

    if isinstance(age_value, (int, float, np.number)):
        return float(age_value)

    text = str(age_value).strip()
    if text == "":
        return np.nan

    text = text.replace("years", "").replace("yrs", "").replace("yo", "").strip()

    if "-" in text:
        parts = text.split("-")
        try:
            years = float(parts[0])
            days = float(parts[1]) if len(parts) > 1 else 0.0
            return years + days / 365.0
        except ValueError:
            pass

    cleaned = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    try:
        return float(cleaned) if cleaned else np.nan
    except ValueError:
        return np.nan


def parse_market_value(value) -> float:
    """Convert Transfermarkt-style market value strings into EUR."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip().replace("€", "").replace(",", "").lower()
    multiplier = 1.0
    if text.endswith("m"):
        multiplier = 1_000_000
        text = text[:-1]
    elif text.endswith("k"):
        multiplier = 1_000
        text = text[:-1]

    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def percentile_rank(series: pd.Series) -> pd.Series:
    """Return percentile rank (0-1) while handling empty inputs."""
    if series.dropna().empty:
        return pd.Series(0.0, index=series.index)
    return series.rank(pct=True, method="min")


def create_age_score(age: float) -> float:
    """Apply strategic weighting to age bands."""
    if pd.isna(age):
        return 12.0  # conservative default when age missing
    if 23 <= age <= 28:
        return 22.0
    if 20 <= age < 23 or 28 < age <= 31:
        return 18.0
    if age < 20:
        return 15.0
    if 31 < age <= 34:
        return 12.0
    return 8.0


def prepare_wage_table(raw_frames: List[pd.DataFrame], squad: pd.DataFrame) -> pd.DataFrame:
    """Aggregate wage tables from different capology sources for Como players."""
    relevant_frames = []
    squad_names = set(squad["Player"].unique())

    for frame in raw_frames:
        if frame.empty:
            continue
        temp = frame.copy()
        if "Player" not in temp.columns:
            continue
        temp = temp[temp["Player"].isin(squad_names)]
        rename_map = {
            "Weekly_Gross_EUR": "Weekly_Gross_EUR",
            "Weekly Wages": "Weekly_Gross_EUR",
            "Yearly_Gross_EUR": "Yearly_Gross_EUR",
            "Annual Wages": "Yearly_Gross_EUR",
            "Season": "Season",
        }
        temp = temp.rename(columns=rename_map)
        for col in ("Weekly_Gross_EUR", "Yearly_Gross_EUR"):
            if col in temp.columns:
                temp[col] = pd.to_numeric(temp[col], errors="coerce")
        relevant_frames.append(temp)

    if not relevant_frames:
        return pd.DataFrame(columns=["Player", "Weekly_Gross_EUR", "Yearly_Gross_EUR"])

    wages = pd.concat(relevant_frames, ignore_index=True)
    wages = wages.sort_values(by=["Season"], ascending=False)
    wages = wages.drop_duplicates(subset=["Player"], keep="first")
    return wages[[col for col in ["Player", "Weekly_Gross_EUR", "Yearly_Gross_EUR"] if col in wages.columns]]


def prepare_player_dataset(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Clean, enrich, and merge squad data for downstream analytics."""
    squad = data["como_agecurve"].copy()

    numeric_cols = [
        "Minutes_2425",
        "Minutes_2526",
        "xG_plus_xAG_2425",
        "xG_plus_xAG_2526",
        "PrgPasses_25",
        "PrgPasses_per90_25",
        "PrgPasses_26",
        "PrgPasses_per90_26",
        "FinalThirdTouches_25",
        "FinalThirdTouches_per90_25",
        "FinalThirdTouches_26",
        "FinalThirdTouches_per90_26",
        "Weekly_Gross_EUR",
        "Yearly_Gross_EUR",
    ]

    for col in numeric_cols:
        if col in squad.columns:
            squad[col] = pd.to_numeric(squad[col], errors="coerce")

    squad["Position_Standard"] = squad.get("Latest_Pos4", squad.get("Latest_Pos", "")).apply(standardize_position)

    squad["Age_numeric"] = squad.get("Age_latest", np.nan).apply(extract_age_from_string)
    if "Age" in squad.columns:
        squad["Age_numeric"] = squad["Age_numeric"].fillna(pd.to_numeric(squad["Age"], errors="coerce"))

    squad["MarketValue_EUR"] = squad.get("MarketValue").apply(parse_market_value)

    wages = prepare_wage_table([
        data.get("capology_manual", pd.DataFrame()),
        data.get("capology_raw", pd.DataFrame()),
    ], squad)

    if not wages.empty:
        squad = squad.merge(wages, on="Player", how="left", suffixes=("", "_wage"))
        for col in ("Weekly_Gross_EUR", "Yearly_Gross_EUR"):
            if f"{col}_wage" in squad.columns:
                squad[col] = squad[col].combine_first(squad[f"{col}_wage"])
                squad = squad.drop(columns=[f"{col}_wage"])

    minutes_cols = [col for col in ["Minutes_2425", "Minutes_2526"] if col in squad.columns]
    squad["Total_Minutes"] = squad[minutes_cols].fillna(0).sum(axis=1) if minutes_cols else 0
    squad["Minutes_Delta"] = squad.get("Minutes_2526", 0) - squad.get("Minutes_2425", 0)
    squad["xG_Delta"] = squad.get("xG_plus_xAG_2526", 0) - squad.get("xG_plus_xAG_2425", 0)

    if "Minutes_2425" in squad.columns:
        squad["Minutes_per_90"] = np.where(squad["Minutes_2425"] > 0, squad["Minutes_2425"] / 90.0, np.nan)
    if "Yearly_Gross_EUR" in squad.columns:
        squad["Cost_per_90"] = np.where(
            squad.get("Minutes_2425", 0) > 0,
            squad["Yearly_Gross_EUR"] / (squad.get("Minutes_2425") / 90.0),
            np.nan,
        )
    if {"MarketValue_EUR", "Yearly_Gross_EUR"}.issubset(squad.columns):
        squad["Wage_to_Value"] = np.where(
            squad["MarketValue_EUR"] > 0,
            squad["Yearly_Gross_EUR"] / squad["MarketValue_EUR"],
            np.nan,
        )

    return squad


def create_performance_score(df: pd.DataFrame) -> pd.DataFrame:
    """Blend availability, attacking output, and age profile into a composite score."""
    df = df.copy()

    df["Minutes_pct"] = percentile_rank(df.get("Minutes_2425", pd.Series(dtype=float)).fillna(0))

    attacking_cols = [
        df.get("xG_plus_xAG_2425", pd.Series(dtype=float)).fillna(0),
        df.get("xG_plus_xAG_2526", pd.Series(dtype=float)).fillna(0) * 0.6,
    ]
    df["Attacking_Index"] = sum(attacking_cols)
    df["Attacking_pct"] = percentile_rank(df["Attacking_Index"])

    progression_cols = [
        df.get("PrgPasses_25", pd.Series(dtype=float)).fillna(0),
        df.get("FinalThirdTouches_25", pd.Series(dtype=float)).fillna(0),
    ]
    df["Progression_Index"] = sum(progression_cols)
    df["Progression_pct"] = percentile_rank(df["Progression_Index"])

    df["Availability_pct"] = percentile_rank(df.get("Total_Minutes", pd.Series(dtype=float)).fillna(0))
    df["MarketValue_pct"] = percentile_rank(df.get("MarketValue_EUR", pd.Series(dtype=float)).fillna(0))

    df["Age_Score"] = df.get("Age_numeric", pd.Series(dtype=float)).apply(create_age_score)

    df["Performance_Score"] = (
        df["Minutes_pct"] * 32
        + df["Attacking_pct"] * 20
        + df["Progression_pct"] * 15
        + df["Availability_pct"] * 10
        + df["MarketValue_pct"] * 8
        + df["Age_Score"]
    )

    return df


def create_wage_analysis(df: pd.DataFrame) -> Dict[str, object]:
    """Generate wage KPIs and efficiency snapshots."""
    if "Weekly_Gross_EUR" not in df.columns:
        return {"message": "No wage data available"}

    wage_df = df.dropna(subset=["Weekly_Gross_EUR"]) if "Weekly_Gross_EUR" in df.columns else pd.DataFrame()
    if wage_df.empty:
        return {"message": "No wage data available"}

    total_weekly = wage_df["Weekly_Gross_EUR"].sum()
    total_yearly = wage_df.get("Yearly_Gross_EUR", pd.Series(dtype=float)).sum()
    if not total_yearly:
        total_yearly = total_weekly * 52

    wage_df["Cost_per_90"] = np.where(
        wage_df.get("Minutes_2425", 0) > 0,
        wage_df.get("Yearly_Gross_EUR", total_yearly) / (wage_df.get("Minutes_2425") / 90.0),
        np.nan,
    )

    top_earners = wage_df.nlargest(10, "Weekly_Gross_EUR")
    efficiency = wage_df.dropna(subset=["Cost_per_90"]).nsmallest(10, "Cost_per_90")
    pressure = wage_df.dropna(subset=["Cost_per_90"]).nlargest(10, "Cost_per_90")

    return {
        "total_players": len(wage_df),
        "total_weekly": total_weekly,
        "total_yearly": total_yearly,
        "avg_weekly": wage_df["Weekly_Gross_EUR"].mean(),
        "median_weekly": wage_df["Weekly_Gross_EUR"].median(),
        "top_earners": top_earners,
        "efficiency": efficiency,
        "pressure": pressure,
    }


def format_currency(value: float, prefix: str = "€", decimals: int = 0) -> str:
    if pd.isna(value):
        return "–"
    return f"{prefix}{value:,.{decimals}f}"


def generate_insights(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Craft qualitative insights and recruitment prompts from filtered data."""
    if df.empty:
        return [], []

    insights: List[str] = []
    recommendations: List[str] = []

    avg_age = df["Age_numeric"].mean()
    if not math.isnan(avg_age):
        insights.append(f"🎂 **Average age**: {avg_age:.1f} years")

    total_minutes = df.get("Minutes_2425", pd.Series(dtype=float)).sum()
    insights.append(f"⏱️ **League minutes covered**: {int(total_minutes):,} in 2024/25")

    if "Performance_Score" in df.columns and not df["Performance_Score"].dropna().empty:
        top_player = df.loc[df["Performance_Score"].idxmax()]
        insights.append(
            f"🏆 **Form talisman**: {top_player['Player']} (score {top_player['Performance_Score']:.1f})"
        )

    if "Cost_per_90" in df.columns and not df["Cost_per_90"].dropna().empty:
        best_value = df.loc[df["Cost_per_90"].idxmin()]
        insights.append(
            f"💶 **Value contract**: {best_value['Player']} at {format_currency(best_value['Cost_per_90'], decimals=0)} per 90"
        )

    veteran_core = df[(df["Age_numeric"] >= 31) & (df.get("Minutes_2425", 0) > 1200)]
    if not veteran_core.empty:
        names = ", ".join(veteran_core["Player"].head(3))
        recommendations.append(
            f"Succession planning needed for senior core ({names})."
        )

    high_potential = df[(df["Age_numeric"] <= 23) & (df["Performance_Score"] > df["Performance_Score"].median())]
    if not high_potential.empty:
        names = ", ".join(high_potential["Player"].head(3))
        recommendations.append(
            f"Prioritise development plans for emerging talents ({names})."
        )

    pressure_contracts = df[df["Cost_per_90"] > df["Cost_per_90"].median() * 1.3]
    if not pressure_contracts.empty:
        names = ", ".join(pressure_contracts["Player"].head(3))
        recommendations.append(
            f"Review wage efficiency for {names}."
        )

    return insights, recommendations


# -----------------------------------------------------------------------------
# Main Streamlit application
# -----------------------------------------------------------------------------
def main():
    st.markdown('<h1 class="main-header">⚽ Como 1907 Squad Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")

    with st.spinner("Loading squad intelligence..."):
        data = load_data()

    if not data:
        st.error("Datasets missing. Please verify CSV paths.")
        return

    squad = prepare_player_dataset(data)
    squad = create_performance_score(squad)
    squad = squad[squad.get("Minutes_2425", 0).fillna(0) >= 90].reset_index(drop=True)

    if squad.empty:
        st.warning("No players meet the minimum minutes threshold (90 minutes). Adjust filters below.")

    wage_analysis = create_wage_analysis(squad)

    # Sidebar filters
    st.sidebar.title("📊 Dashboard Controls")
    positions = ["All"] + sorted(squad["Position_Standard"].dropna().unique().tolist())
    selected_position = st.sidebar.selectbox("Select Position", positions)

    if squad["Age_numeric"].notna().any():
        age_min = int(math.floor(squad["Age_numeric"].min()))
        age_max = int(math.ceil(squad["Age_numeric"].max()))
    else:
        age_min, age_max = 18, 36
    age_range = st.sidebar.slider("Age Range", min_value=age_min, max_value=age_max, value=(age_min, age_max))

    minutes_ceiling = int(squad.get("Minutes_2425", pd.Series(dtype=float)).max() or 0)
    min_minutes = st.sidebar.slider("Minimum Minutes", min_value=0, max_value=max(90, minutes_ceiling), value=90, step=30)

    filtered = squad.copy()
    filtered = filtered[filtered.get("Minutes_2425", 0).fillna(0) >= min_minutes]
    if selected_position != "All":
        filtered = filtered[filtered["Position_Standard"] == selected_position]
    if "Age_numeric" in filtered.columns and filtered["Age_numeric"].notna().any():
        mask = (filtered["Age_numeric"] >= age_range[0]) & (filtered["Age_numeric"] <= age_range[1])
        mask |= filtered["Age_numeric"].isna()
        filtered = filtered[mask]

    tab_overview, tab_performance, tab_salary, tab_insights = st.tabs([
        "📊 Squad Overview",
        "⚽ Performance Analysis",
        "💰 Salary Analysis",
        "📈 Strategic Insights",
    ])

    # ------------------------------------------------------------------
    # Squad Overview
    # ------------------------------------------------------------------
    with tab_overview:
        st.markdown('<h2 class="section-header">Squad Overview</h2>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Players", len(filtered))
        if filtered["Age_numeric"].notna().any():
            col2.metric("Average Age", f"{filtered['Age_numeric'].mean():,.1f} years")
        else:
            col2.metric("Average Age", "N/A")
        if "Minutes_2425" in filtered.columns:
            col3.metric("Total Minutes (24/25)", f"{filtered['Minutes_2425'].sum():,.0f}")
        else:
            col3.metric("Total Minutes", "N/A")
        col4.metric("Avg Performance Score", f"{filtered['Performance_Score'].mean():.1f}" if not filtered.empty else "N/A")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.subheader("Position Distribution")
            if not filtered.empty:
                pos_counts = filtered["Position_Standard"].value_counts().reset_index()
                pos_counts.columns = ["Position", "Players"]
                fig = px.pie(pos_counts, values="Players", names="Position", title="Players by Position")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No players available for chart.")

        with chart_col2:
            st.subheader("Age Distribution")
            if filtered["Age_numeric"].notna().any():
                fig = px.histogram(
                    filtered.dropna(subset=["Age_numeric"]),
                    x="Age_numeric",
                    nbins=15,
                    title="Age Distribution",
                    labels={"Age_numeric": "Age"},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Age data not available")

        st.subheader("Top Performers")
        display_cols = ["Player", "Position_Standard", "Minutes_2425", "Performance_Score"]
        if "Age_numeric" in filtered.columns:
            display_cols.insert(2, "Age_numeric")
        if "xG_plus_xAG_2425" in filtered.columns:
            display_cols.append("xG_plus_xAG_2425")
        top_performers = filtered.nlargest(10, "Performance_Score")[display_cols] if not filtered.empty else pd.DataFrame(columns=display_cols)
        st.dataframe(top_performers, use_container_width=True)

        st.subheader("Market Value vs Performance")
        mv_df = filtered.dropna(subset=["MarketValue_EUR", "Performance_Score"])
        if not mv_df.empty:
            fig = px.scatter(
                mv_df,
                x="MarketValue_EUR",
                y="Performance_Score",
                color="Position_Standard",
                size="Minutes_2425",
                hover_name="Player",
                labels={"MarketValue_EUR": "Market Value (€)", "Performance_Score": "Performance Score"},
                title="Squad Value Curve",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Market value data not available for the selected cohort.")

    # ------------------------------------------------------------------
    # Performance Analysis
    # ------------------------------------------------------------------
    with tab_performance:
        st.markdown('<h2 class="section-header">Performance Analysis</h2>', unsafe_allow_html=True)

        perf_col1, perf_col2 = st.columns(2)
        with perf_col1:
            st.subheader("Performance Score by Position")
            if not filtered.empty:
                pos_perf = filtered.groupby("Position_Standard")["Performance_Score"].mean().reset_index()
                fig = px.bar(pos_perf, x="Position_Standard", y="Performance_Score", title="Average Performance Score by Role")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No players to visualise")

        with perf_col2:
            st.subheader("Minutes vs Age")
            if filtered["Age_numeric"].notna().any() and "Minutes_2425" in filtered.columns:
                fig = px.scatter(
                    filtered,
                    x="Age_numeric",
                    y="Minutes_2425",
                    color="Position_Standard",
                    size="Performance_Score",
                    hover_name="Player",
                    labels={"Age_numeric": "Age", "Minutes_2425": "Minutes (24/25)"},
                    title="Minutes Played vs Age",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Age or minutes data unavailable")

        st.subheader("Season-on-Season Minutes Shift")
        if {"Minutes_2425", "Minutes_2526"}.issubset(filtered.columns):
            season_df = filtered.dropna(subset=["Minutes_2425", "Minutes_2526"])
            if not season_df.empty:
                fig = px.scatter(
                    season_df,
                    x="Minutes_2425",
                    y="Minutes_2526",
                    color="Position_Standard",
                    hover_name="Player",
                    labels={"Minutes_2425": "2024/25 Minutes", "Minutes_2526": "2025/26 Minutes"},
                    title="Playing Time Trajectory",
                )
                fig.add_trace(
                    go.Scatter(
                        x=[season_df["Minutes_2425"].min(), season_df["Minutes_2425"].max()],
                        y=[season_df["Minutes_2425"].min(), season_df["Minutes_2425"].max()],
                        mode="lines",
                        name="Parity line",
                        line=dict(color="#7f8c8d", dash="dash"),
                        showlegend=True,
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Insufficient overlapping minutes data to compare seasons.")
        else:
            st.info("2025/26 minutes not available in dataset.")

        st.subheader("Position-Specific Dashboards")
        for pos in ["GK", "DF", "MF", "FW"]:
            pos_data = filtered[filtered["Position_Standard"] == pos]
            if len(pos_data) == 0:
                continue
            with st.expander(f"{pos} – {len(pos_data)} players"):
                exp_col1, exp_col2, exp_col3 = st.columns(3)
                if pos_data["Age_numeric"].notna().any():
                    exp_col1.metric("Avg Age", f"{pos_data['Age_numeric'].mean():.1f}")
                else:
                    exp_col1.metric("Avg Age", "N/A")
                if "Minutes_2425" in pos_data.columns:
                    exp_col2.metric("Minutes 24/25", f"{pos_data['Minutes_2425'].sum():,.0f}")
                else:
                    exp_col2.metric("Minutes 24/25", "N/A")
                exp_col3.metric("Avg Score", f"{pos_data['Performance_Score'].mean():.1f}")

                display_cols = ["Player", "Minutes_2425", "Performance_Score", "Attacking_Index", "Progression_Index"]
                if "Cost_per_90" in pos_data.columns:
                    display_cols.append("Cost_per_90")
                st.dataframe(pos_data.sort_values("Performance_Score", ascending=False)[display_cols], use_container_width=True)

    # ------------------------------------------------------------------
    # Salary Analysis
    # ------------------------------------------------------------------
    with tab_salary:
        st.markdown('<h2 class="section-header">Salary Analysis</h2>', unsafe_allow_html=True)
        st.info("💰 Wage data blends Como internal tracking with Capology references.")

        if wage_analysis.get("message"):
            st.warning(wage_analysis["message"])
        else:
            sal_col1, sal_col2, sal_col3, sal_col4 = st.columns(4)
            sal_col1.metric("Players with Wages", f"{wage_analysis['total_players']}")
            sal_col2.metric("Weekly Wage Bill", format_currency(wage_analysis["total_weekly"]))
            sal_col3.metric("Annual Wage Bill", format_currency(wage_analysis["total_yearly"]))
            sal_col4.metric("Avg Weekly Wage", format_currency(wage_analysis["avg_weekly"]))

            st.subheader("🏆 Top Earners")
            top_earners = wage_analysis["top_earners"].copy()
            top_earners["Weekly_Gross_EUR"] = top_earners["Weekly_Gross_EUR"].apply(format_currency)
            if "Yearly_Gross_EUR" in top_earners.columns:
                top_earners["Yearly_Gross_EUR"] = top_earners["Yearly_Gross_EUR"].apply(format_currency)
            st.dataframe(top_earners[[col for col in top_earners.columns if col in ("Player", "Position_Standard", "Weekly_Gross_EUR", "Yearly_Gross_EUR")]], use_container_width=True)

            st.subheader("📊 Average Weekly Salary by Position")
            wage_filtered = filtered.dropna(subset=["Weekly_Gross_EUR"]) if "Weekly_Gross_EUR" in filtered.columns else pd.DataFrame()
            if not wage_filtered.empty:
                salary_stats = wage_filtered.groupby("Position_Standard")["Weekly_Gross_EUR"].agg(["count", "mean", "median"]).round(0)
                salary_stats = salary_stats.rename(columns={"count": "Players", "mean": "Average", "median": "Median"})
                salary_stats["Average"] = salary_stats["Average"].apply(format_currency)
                salary_stats["Median"] = salary_stats["Median"].apply(format_currency)
                st.dataframe(salary_stats, use_container_width=True)
            else:
                st.info("No wage data for current filter selection.")

            st.subheader("⚖️ Wage Efficiency (Cost per 90 Minutes)")
            eff_df = wage_analysis.get("efficiency", pd.DataFrame()).copy()
            eff_df["Cost_per_90"] = eff_df["Cost_per_90"].apply(lambda v: format_currency(v, decimals=0))
            eff_cols = [col for col in ["Player", "Position_Standard", "Minutes_2425", "Cost_per_90"] if col in eff_df.columns]
            st.dataframe(eff_df[eff_cols], use_container_width=True)

            st.subheader("⚠️ Wage Pressure Watchlist")
            pressure_df = wage_analysis.get("pressure", pd.DataFrame()).copy()
            if not pressure_df.empty:
                pressure_df["Cost_per_90"] = pressure_df["Cost_per_90"].apply(lambda v: format_currency(v, decimals=0))
                pressure_cols = [col for col in ["Player", "Position_Standard", "Minutes_2425", "Cost_per_90"] if col in pressure_df.columns]
                st.dataframe(pressure_df[pressure_cols], use_container_width=True)
            else:
                st.info("No high-pressure contracts identified with current filters.")

            st.subheader("📈 Weekly Salary Distribution")
            if not wage_filtered.empty:
                fig = px.histogram(
                    wage_filtered,
                    x="Weekly_Gross_EUR",
                    nbins=12,
                    title="Weekly Salary Distribution",
                    labels={"Weekly_Gross_EUR": "Weekly Salary (EUR)", "count": "Players"},
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No salary data to plot.")

    # ------------------------------------------------------------------
    # Strategic Insights
    # ------------------------------------------------------------------
    with tab_insights:
        st.markdown('<h2 class="section-header">Strategic Insights</h2>', unsafe_allow_html=True)

        insights, recommendations = generate_insights(filtered)
        if insights:
            for insight in insights:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        else:
            st.info("Insights will populate once players match the filters above.")

        if recommendations:
            st.subheader("Recommended Actions")
            for rec in recommendations:
                st.markdown(f'<div class="recommendation-box">{rec}</div>', unsafe_allow_html=True)

        st.subheader("Succession Planning Flags")
        veteran_flags = filtered[(filtered["Age_numeric"] >= 30) & (filtered.get("Minutes_2425", 0) > 900)]
        if not veteran_flags.empty:
            display_cols = [col for col in ["Player", "Age_numeric", "Minutes_2425", "Performance_Score", "Cost_per_90"] if col in veteran_flags.columns]
            st.dataframe(veteran_flags.sort_values("Age_numeric", ascending=False)[display_cols], use_container_width=True)
        else:
            st.info("No senior regulars in the current filter.")

        st.subheader("Emerging Talent Radar")
        emerging = filtered[(filtered["Age_numeric"] <= 23) & (filtered["Performance_Score"] >= filtered["Performance_Score"].quantile(0.6))]
        if not emerging.empty:
            display_cols = [col for col in ["Player", "Age_numeric", "Minutes_2425", "Attacking_Index", "Progression_Index", "Performance_Score"] if col in emerging.columns]
            st.dataframe(emerging.sort_values("Performance_Score", ascending=False)[display_cols], use_container_width=True)
        else:
            st.info("No emerging talents based on the current cut-offs.")

    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #7f8c8d;'>
        <p>Como 1907 Squad Analysis Dashboard | Data Analysis Team | 2024–2025</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
