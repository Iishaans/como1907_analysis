"""
Como 1907 Squad Analysis Dashboard
==================================

A comprehensive Streamlit dashboard for FC Como 1907 stakeholders,
providing performance analysis, wage insights, and strategic recommendations.

Author: Iishaan Shekhar
Date: October 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Como 1907 Squad Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
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
    .insight-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2980b9;
        position: 1rem 0;
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
        height: 50px; 
        white-space: pre-wrap; 
        background-color: #f0f2f6; 
        border-radius: 4px 4px 0 0; 
        gap: 1px; 
        padding: 0 20px; 
    }
    .stTabs [aria-selected="true"] { 
        background-color: #1f77b4; 
        color: white; 
    }
</style>
""", unsafe_allow_html=True)

# Data loading function
@st.cache_data
def load_data():
    """Load Como 1907 datasets"""
    try:
        data_path = Path('data')
        
        # Load main integrated dataset
        como_agecurve = pd.read_csv(data_path / 'como_agecurve_wide.csv')
        
        # Load intermediate datasets
        fbref_2425 = pd.read_csv(data_path / 'intermediate' / 'fbref_20242025.csv')
        fbref_2526 = pd.read_csv(data_path / 'intermediate' / 'fbref_20252026.csv')
        transfermarkt = pd.read_csv(data_path / 'intermediate' / 'transfermarkt_contracts.csv')
        
        # Load wage data
        try:
            capology_manual = pd.read_csv(data_path / 'intermediate' / 'Como_Wage_Breakdown_2425_2526_Cleaned.csv')
        except FileNotFoundError:
            capology_manual = pd.DataFrame()
            
        return como_agecurve, fbref_2425, fbref_2526, transfermarkt, capology_manual
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

# Utility functions
def standardize_position(pos_str):
    """Standardize position strings"""
    if pd.isna(pos_str):
        return 'Unknown'
    pos_str = str(pos_str).upper()
    if 'GK' in pos_str:
        return 'GK'
    elif 'DF' in pos_str or 'DEF' in pos_str or 'BACK' in pos_str:
        return 'DF'
    elif 'MF' in pos_str or 'MID' in pos_str:
        return 'MF'
    elif 'FW' in pos_str or 'FOR' in pos_str or 'STRIKER' in pos_str:
        return 'FW'
    else:
        return 'Unknown'

def extract_age_from_string(age_str):
    """Extract numeric age from age string format"""
    if pd.isna(age_str):
        return None
    try:
        if isinstance(age_str, sstr) and '-' in age_str:
            age_part = age_str.split('-')[0]
            return float(age_part)
        elif isinstance(age_str, (int, float)):
            return float(age_str)
        else:
            return None
    except:
        return None

def create_performance_score(df):
    """Create comprehensive performance scores for players"""
    df = df.copy()
    df['Performance_Score'] = 0
    
    # Minutes contribution (40% of score)
    if 'Minutes_2425' in df.columns:
        max_minutes = df['Minutes_2425'].max()
        if max_minutes > 0:
            df['Minutes_Score'] = (df['Minutes_2425'] / max_minutes) * 40
            df['Performance_Score'] += df['Minutes_Score']
    
    # Age optimization (30% of score)
    if 'Age_latest' in df.columns:
        df['Age_numeric'] = df['Age_latest'].apply(extract_age_from_string)
        
    if 'Age_numeric' in df.columns:
        df['Age_Score'] = 0
        for idx, row in df.iterrows():
            age = row['Age_numeric']
            if pd.notna(age):
                if 23 <= age <= 30:
                    df.loc[idx, 'Age_Score'] = 30  # Prime age
                elif 20 <= age < 23 or 30 < age <= 33:
                    df.loc[idx, 'Age_Score'] = 20  # Good age
                elif age < 20:
                    df.loc[idx, 'Age_Score'] = 15  # Young prospect
                else:
                    df.loc[idx, 'Age_Score'] = 10  # Veteran
        df['Performance_Score'] += df['Age_Score']
    
    return df

def create_wage_analysis(df):
    """Create comprehensive wage analysis"""
    if 'Weekly_Gross_EUR' not in df.columns:
        return "No wage data available"
        
    wage_data = df.dropna(subset=['Weekly_Gross_EUR'])
    if wage_data.empty:
        return "No wage data available"
    
    # Basic statistics
    total_weekly = wage_data['Weekly_Gross_EUR'].sum()
    total_yearly = wage_data['Yearly_Gross_EUR'].sum() if 'Yearly_Gross_EUR' in wage_data.columns and wage_data['Yearly_Gross_EUR'].sum() > 0 else total_weekly * 52
    
    avg_weekly = wage_data['Weekly_Gross_EUR'].mean()
    median_weekly = wage_data['Weekly_Gross_EUR'].median()
    
    # Top earners
    display_cols = ['Player', 'Weekly_Gross_EUR']
    if 'Position_Standard' in wage_data.columns:
        display_cols.insert(1, 'Position_Standard')
    if 'Yearly_Gross_EUR' in wage_data.columns:
        display_cols.append('Yearly_Gross_EUR')
        
    top_earners = wage_data.nlargest(10, 'Weekly_Gross_EUR')[display_cols]
    
    return {
        'total_players': len(wage_data),
        'total_weekly': total_weekly,
        'total_yearly': total_yearly,
        'avg_weekly': avg_weekly,
        'median_weekly': median_weekly,
        'top_earners': top_earners
    }

# Main application
def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">⚽ Como 1907 Squad Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading squad data..."):
        data_results = load_data()
        como_agecurve, fbref_2425, fbref_2526, transfermarkt, capology_manual = data_results
    
    if como_agecurve is None:
        st.error("Failed to load data. Please check your data files.")
        return
    
    # Data processing
    df_performance = como_agecurve.copy()
    df_performance['Position_Standard'] = df_performance['Latest_Pos4'].apply(standardize_position)
    
    # Filter for players with sufficient playing time
    if 'Minutes_2425' in df_performance.columns:
        df_performance = df_performance[df_performance['Minutes_2425'].fillna(0) >= 90]
    
    df_performance = create_performance_score(df_performance)
    
    # Create wage analysis
    wage_analysis = create_wage_analysis(df_performance)
    
    # Sidebar controls
    st.sidebar.title("📊 Dashboard Controls")
    
    # Position filter
    positions = ['All'] + sorted(df_performance['Position_Standard'].unique().tolist())
    selected_position = st.sidebar.selectbox("Select Position", positions)
    
    # Age range filter
    if 'Age_numeric' in df_performance.columns and df_performance['Age_numeric'].notna().any():
        age_min = int(df_performance['Age_numeric'].min())
        age_max = int(df_performance['Age_numeric'].max())
        age_range = st.sidebar.slider(
            "Age Range",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max)
        )
    else:
        age_range = (18, 40)
    
    # Minutes filter
    mm_max = int(df_performance['Minutes_2425'].max()) if 'Minutes_2425' in df_performance.columns else 0
    min_minutes = st.sidebar.slider("Minimum Minutes", min_value=0, max_value=mm_max, value=0)
    
    # Apply filters
    filtered_df = df_performance.copy()
    
    if 'Age_numeric' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['Age_numeric'] >= age_range[0]) & 
            (filtered_df['Age_numeric'] <= age_range[1])
        ]
    
    if 'Minutes_2425' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Minutes_2425'].fillna(0) >= min_minutes]
    
    if selected_position != 'All':
        filtered_df = filtered_df[filtered_df['Position_Standard'] == selected_position]
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Squad Overview", 
        "⚽ Performance Analysis",
        "💰 Salary Analysis",
        "📈 Key Insights"
    ])
    
    # Squad Overview Tab
    with tab1:
        st.markdown('<h2 class="section-header">Squad Overview</h2>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Players", len(filtered_df))
        
        with col2:
            if 'Age_numeric' in filtered_df.columns and filtered_df['Age_numeric'].notna().any():
                st.metric("Average Age", f"{filtered_df['Age_numeric'].mean():.,1f} years")
            else:
                st.metric("Average Age", "N/A")
        
        with col3:
            if 'Minutes_2425' in filtered_df.columns:
                st.metric("Total Minutes", f"{filtered_df['Minutes_2425'].sum():,.0f}")
            else:
                st.metric("Total Minutes", "N/A")
        
        with col4:
            st.metric("Avg Performance Score", f"{filtered_df['Performance_Score'].mean():.1f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Position Distribution")
            pos_counts = filtered_df['Position_Standard'].value_counts()
            fig = px.pie(values=pos_counts.values, names=pos_counts.index, 
                        title="Players by Position")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Age Distribution")
            if 'Age_numeric' in filtered_df.columns and filtered_df['Age_numeric'].notna().any():
                fig = px.histogram(filtered_df, x='Age_numeric', nbins=15, 
                                  title="Age Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Age data not available")
        
        # Top performers
        st.subheader("Top Performers")
        display_cols = ['Player', 'Position_Standard', 'Minutes_2425', 'Performance_Score']
        if 'Age_numeric' in filtered_df.columns:
            display_cols.insert(2, 'Age_numeric')
        top_performers = filtered_df.nlargest(10, 'Performance_Score')[display_cols]
        st.dataframe(top_performers, use_container_width=True)
    
    # Performance Analysis Tab
    with tab2:
        st.markdown('<h2 class="section-header">Performance Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Score by Position")
            pos_performance = filtered_df.groupby('Position_Standard')['Performance_Score'].mean().reset_index()
            fig = px.bar(pos_performance, x='Position_Standard', y='Performance_Score',
                        title="Average Performance Score by Position")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Minutes vs Age")
            if 'Age_numeric' in filtered_df.columns and 'Minutes_2425' in filtered_df.columns:
                fig = px.scatter(filtered_df, x='Age_numeric', y='Minutes_2425',
                                color='Position_Standard', size='Performance_Score',
                                title="Minutes Played vs Age")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Age or minutes data not available")
        
        # Position-specific analysis
        st.subheader("Position-Specific Analysis")
        
        for pos in ['GK', 'DF', 'MF', 'FW']:
            pos_data = filtered_df[filtered_df['Position_Standard'] == pos]
            if len(pos_data) > 0:
                with st.expander(f"{pos} - {len(pos_data)} players"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'Age_numeric' in pos_data.columns and pos_data['Age_numeric'].notna().any():
                            st.metric("Average Age", f"{pos_data['Age_numeric'].mean():.1f}")
                        else:
                            st.metric("Average Age", "N/A")
                    
                    with col2:
                        if "Minutes_2425" in pos_data.columns:
                            st.metric("Total Minutes", f"{pos_data['Minutes_2425'].sum():,.0f}")
                        else:
                            st.metric("Total Minutes", "N/A")
                    
                    with col3:
                        st.metric("Avg Performance", f"{pos_data['Performance_Score'].mean():.1f}")
                    
                    display_cols = ["Player", "Minutes_2425", "Performance_Score"]
                    if "Age_numeric" in pos_data.columns:
                        display_cols.insert(1, "Age_numeric")
                    top_pos = pos_data.nlargest(5, "Performance_Score")[display_cols]
                    st.dataframe(top_pos, use_container_width=True)
    
    # Salary Analysis Tab
    with tab3:
        st.markdown('<h2 class="section-header">Salary Analysis</h2>', unsafe_allow_html=True)
        st.info("💰 Wage data sourced from manual Capology breakdown files")
        
        if isinstance(wage_analysis, dict):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Players with Wages", f"{wage_analysis['total_players']}")
            
            with col2:
                st.metric("Weekly Wage Bill", f"€{wage_analysis['total_weekly']:,.0f}")
            
            with col3:
                st.metric("Annual Wage Bill", f"€{wage_analysis['total_yearly']:,.0f}")
            
            with col4:
                st.metric("Avg Weekly Wage", f"€{wage_analysis['avg_weekly']:,.0f}")
            
            # Top earners
            st.subheader("🏆 Top Earners")
            wage_filtered = filtered_df.dropna(subset=['Weekly_Gross_EUR'])
            if not wage_filtered.empty:
                top_earners_display = wage_analysis['top_earners'].copy()
                if 'Weekly_Gross_EUR' in top_earners_display.columns:
                    top_earners_display['Weekly Gross (EUR)'] = top_earners_display['Weekly_Gross_EUR'].apply(lambda x: f"€{x:,.0f}")
                    top_earners_display = top_earners_display.drop(columns=['Weekly_Gross_EUR'])
                st.dataframe(top_earners_display, use_container_width=True)
                
                # Position salary analysis
                st.subheader("📊 Average Weekly Salary by Position")
                salary_stats = wage_filtered.groupby('Position_Standard')['Weekly_Gross_EUR'].agg(['count', 'mean', 'median']).round(0)
                salary_display = pd.DataFrame({
                    'Count': salary_stats['count'],
                    'Average': salary_stats['mean'].apply(lambda x: f"€{x:,.0f}"),
                    'Median': salary_stats['median'].apply(lambda x: f"€{x:,.0f}")
                })
                st.dataframe(salary_display, use_container_width=True)
                
                # Salary distribution chart
                st.subheader("📈 Weekly Salary Distribution")
                fig = px.histogram(wage_filtered, x='Weekly_Gross_EUR', 
                                 title="Weekly Salary Distribution",
                                 labels={'Weekly_Gross_EUR': 'Weekly Salary (EUR)', 'count': 'Number of Players'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"{wage_analysis}")
    
    # Key Insights Tab
    with tab4:
        st.markdown('<h2 class="section-header">Key Insights</h2>', unsafe_allow_html=True)
        
        if not filtered_df.empty:
            insights = [
                f"📊 **Squad Size**: {len(filtered_df)} players analyzed",
                f"⏱️ **Total Minutes**: {int(filtered_df.get('Minutes_2425', pd.Series(dtype=float)).sum()):,} minutes played",
                f"🏆 **Top Performer**: {filtered_df.loc[filtered_df['Performance_Score'].idxmax(), 'Player']}"
            ]
            
            if "Age_numeric" in filtered_df.columns and filtered_df["Age_numeric"].notna().any():
                insights.insert(1, f"🎂 **Average Age**: {filtered_df['Age_numeric'].mean():.1f} years")
            
            for insight in insights:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
        else:
            st.info("No players left after filters.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>Como 1907 Squad Analysis Dashboard | Data Analysis Team | 2024–2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()