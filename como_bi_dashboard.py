"""
Como 1907 Business Intelligence Dashboard
========================================

A comprehensive Streamlit dashboard for Como 1907 stakeholders
consolidating key insights from the EDA analysis.

Author: Data Analysis Team
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and process all datasets"""
    try:
        data_path = Path('data')
        
        # Load datasets
        como_agecurve = pd.read_csv(data_path / 'como_agecurve_wide.csv')
        fbref_2425 = pd.read_csv(data_path / 'intermediate' / 'fbref_20242025.csv')
        fbref_2526 = pd.read_csv(data_path / 'intermediate' / 'fbref_20252026.csv')
        transfermarkt = pd.read_csv(data_path / 'intermediate' / 'transfermarkt_contracts.csv')
        capology = pd.read_csv(data_path / 'intermediate' / 'capology_wages.csv')
        
        return como_agecurve, fbref_2425, fbref_2526, transfermarkt, capology
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

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

def extract_market_value(value):
    """Extract numeric market values from strings"""
    if pd.isna(value) or value == '-':
        return 0
    try:
        value_str = str(value).replace('€', '').replace('m', '').replace('k', '')
        if 'k' in str(value):
            return float(value_str) * 0.001
        return float(value_str)
    except:
        return 0

def extract_age_from_string(age_str):
    """Extract numeric age from age string format like '21-013' or '30-236'"""
    if pd.isna(age_str):
        return None
    try:
        # Handle string format like '21-013' or '30-236'
        if isinstance(age_str, str) and '-' in age_str:
            age_part = age_str.split('-')[0]
            return float(age_part)
        # Handle numeric values
        elif isinstance(age_str, (int, float)):
            return float(age_str)
        else:
            return None
    except:
        return None

def create_wage_analysis(df):
    """Create comprehensive wage analysis"""
    wage_data = df.dropna(subset=['Weekly_Gross_EUR'])
    
    if wage_data.empty:
        return "No wage data available"
    
    # Basic statistics
    total_weekly = wage_data['Weekly_Gross_EUR'].sum()
    total_yearly = wage_data['Yearly_Gross_EUR'].sum()
    avg_weekly = wage_data['Weekly_Gross_EUR'].mean()
    median_weekly = wage_data['Weekly_Gross_EUR'].median()
    
    # Top earners
    top_earners = wage_data.nlargest(10, 'Weekly_Gross_EUR')
    
    return {
        'total_players': len(wage_data),
        'total_weekly': total_weekly,
        'total_yearly': total_yearly,
        'avg_weekly': avg_weekly,
        'median_weekly': median_weekly,
        'top_earners': top_earners
    }

def create_performance_score(df):
    """Create performance scores for players"""
    df = df.copy()
    df['Performance_Score'] = 0
    
    # Convert age to numeric
    if 'Age_latest' in df.columns:
        df['Age_numeric'] = df['Age_latest'].apply(extract_age_from_string)
    
    # Minutes contribution
    if 'Minutes_2425' in df.columns:
        max_minutes = df['Minutes_2425'].max()
        df['Minutes_Score'] = (df['Minutes_2425'] / max_minutes) * 40
        df['Performance_Score'] += df['Minutes_Score']
    
    # Age factor
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
    
    # Market value factor
    if 'MarketValue' in df.columns:
        df['Numeric_Market_Value'] = df['MarketValue'].apply(extract_market_value)
        if df['Numeric_Market_Value'].max() > 0:
            max_value = df['Numeric_Market_Value'].max()
            df['Market_Value_Score'] = (df['Numeric_Market_Value'] / max_value) * 30
            df['Performance_Score'] += df['Market_Value_Score']
    
    return df

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">⚽ Como 1907 Squad Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading squad data..."):
        como_agecurve, fbref_2425, fbref_2526, transfermarkt, capology = load_data()
    
    if como_agecurve is None:
        st.error("Failed to load data. Please check your data files.")
        return
    
    # Process data
    df_performance = como_agecurve.copy()
    df_performance['Position_Standard'] = df_performance['Latest_Pos4'].apply(standardize_position)
    df_performance = df_performance[df_performance['Minutes_2425'] >= 90]  # Filter for sufficient minutes
    df_performance = create_performance_score(df_performance)
    
    # Use performance data with existing wage columns
    # The wage data is already included in como_agecurve from the builder
    df_performance_with_wages = df_performance.copy()
    
    # Create wage analysis with existing wage data
    wage_analysis = create_wage_analysis(df_performance_with_wages)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    # Position filter
    positions = ['All'] + list(df_performance['Position_Standard'].unique())
    selected_position = st.sidebar.selectbox("Select Position", positions)
    
    # Age range filter
    if 'Age_numeric' in df_performance.columns:
        age_min = int(df_performance['Age_numeric'].min())
        age_max = int(df_performance['Age_numeric'].max())
        age_range = st.sidebar.slider(
            "Age Range",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max)
        )
    else:
        age_range = (18, 40)  # Default range
    
    # Minutes filter
    min_minutes = st.sidebar.slider(
        "Minimum Minutes",
        min_value=0,
        max_value=int(df_performance['Minutes_2425'].max()),
        value=0
    )
    
    # Apply filters
    if 'Age_numeric' in df_performance.columns:
        filtered_df = df_performance[
            (df_performance['Age_numeric'] >= age_range[0]) & 
            (df_performance['Age_numeric'] <= age_range[1]) &
            (df_performance['Minutes_2425'] >= min_minutes)
        ]
    else:
        filtered_df = df_performance[df_performance['Minutes_2425'] >= min_minutes]
    
    if selected_position != 'All':
        filtered_df = filtered_df[filtered_df['Position_Standard'] == selected_position]
    
    # Main content tabs
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
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Players", len(filtered_df))
        
        with col2:
            if 'Age_numeric' in filtered_df.columns:
                st.metric("Average Age", f"{filtered_df['Age_numeric'].mean():.1f} years")
            else:
                st.metric("Average Age", "N/A")
        
        with col3:
            st.metric("Total Minutes", f"{filtered_df['Minutes_2425'].sum():,.0f}")
        
        with col4:
            st.metric("Avg Performance Score", f"{filtered_df['Performance_Score'].mean():.1f}")
        
        # Position distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Position Distribution")
            pos_counts = filtered_df['Position_Standard'].value_counts()
            fig = px.pie(values=pos_counts.values, names=pos_counts.index, 
                        title="Players by Position")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Age Distribution")
            if 'Age_numeric' in filtered_df.columns:
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
    
    with tab2:
        st.markdown('<h2 class="section-header">Performance Analysis</h2>', unsafe_allow_html=True)
        
        # Performance by position
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Score by Position")
            pos_performance = filtered_df.groupby('Position_Standard')['Performance_Score'].mean().reset_index()
            fig = px.bar(pos_performance, x='Position_Standard', y='Performance_Score',
                        title="Average Performance Score by Position")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Minutes vs Age")
            if 'Age_numeric' in filtered_df.columns:
                fig = px.scatter(filtered_df, x='Age_numeric', y='Minutes_2425',
                               color='Position_Standard', size='Performance_Score',
                               title="Minutes Played vs Age")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Age data not available")
        
        # Position-specific analysis
        st.subheader("Position-Specific Analysis")
        
        for pos in ['GK', 'DF', 'MF', 'FW']:
            pos_data = filtered_df[filtered_df['Position_Standard'] == pos]
            if len(pos_data) > 0:
                with st.expander(f"{pos} - {len(pos_data)} players"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'Age_numeric' in pos_data.columns:
                            st.metric("Average Age", f"{pos_data['Age_numeric'].mean():.1f}")
                        else:
                            st.metric("Average Age", "N/A")
                    
                    with col2:
                        st.metric("Total Minutes", f"{pos_data['Minutes_2425'].sum():,.0f}")
                    
                    with col3:
                        st.metric("Avg Performance", f"{pos_data['Performance_Score'].mean():.1f}")
                    
                    # Top performers in position
                    display_cols = ['Player', 'Minutes_2425', 'Performance_Score']
                    if 'Age_numeric' in pos_data.columns:
                        display_cols.insert(1, 'Age_numeric')
                    top_pos = pos_data.nlargest(5, 'Performance_Score')[display_cols]
                    st.dataframe(top_pos, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="section-header">Salary Analysis</h2>', unsafe_allow_html=True)
        
        if isinstance(wage_analysis, dict):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("/59 Players", f"{wage_analysis['total_players']}")
            with col2:
                st.metric("Weekly Wage Bill", f"€{wage_analysis['total_weekly']:,.0f}")
            with col3:
                st.metric("Annual Wage Bill", f"€{wage_analysis['total_yearly']:,.0f}")
            with col4:
                st.metric("Avg Weekly Wage", f"€{wage_analysis['avg_weekly']:,.0f}")
            
            # Top earners table
            st.subheader("🏆 Top Earners")
            top_earners_display = wage_analysis['top_earners'][['Player', 'Position', 'Weekly_Gross_EUR', 'Yearly_Gross_EUR']].copy()
            top_earners_display['Weekly Gross EUR'] = top_earners_display['Weekly_Gross_EUR'].apply(lambda x: f"€{x:,.0f}")
            top_earners_display['Yearly Gross EUR'] = top_earners_display['Yearly_Gross_EUR'].apply(lambda x: f"€{x:,.0f}")
            top_earners_display = top_earners_display.drop(columns=['Weekly_Gross_EUR', 'Yearly_Gross_EUR'])
            st.dataframe(top_earners_display, use_container_width=True)
            
            # Position salary analysis
            st.subheader("📊 Average Salary by Position")
            wage_filtered = df_performance_with_wages.dropna(subset=['Weekly_Gross_EUR'])
            if not wage_filtered.empty:
                salary_stats = wage_filtered.groupby('Position_Standard')['Weekly_Gross_EUR'].agg(['count', 'mean', 'median']).round(0)
                salary_display = pd.DataFrame({
                    'Count': salary_stats['count'],
                    'Average': salary_stats['mean'].apply(lambda x: f"€{x:,.0f}"),
                    'Median': salary_stats['median'].apply(lambda x: f"€{x:,.0f}")
                })
                st.dataframe(salary_display, use_container_width=True)
                
                # Salary distribution chart
                st.subheader("📈 Salary Distribution")
                fig = px.histogram(wage_filtered, x='Weekly_Gross_EUR', 
                                 title="Weekly Salary Distribution",
                                 labels={'Weekly_Gross_EUR': 'Weekly Salary (EUR)', 'count': 'Number of Players'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("💰 Wage data not available - using manual Capology data file")
            
    with tab4:
        st.markdown('<h2 class="section-header">Key Insights</h2>', unsafe_allow_html=True)
        
        # Squad balance analysis
        st.subheader("Squad Balance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age balance
            if 'Age_numeric' in filtered_df.columns:
                young_players = len(filtered_df[filtered_df['Age_numeric'] < 23])
                prime_players = len(filtered_df[(filtered_df['Age_numeric'] >= 23) & (filtered_df['Age_numeric'] <= 30)])
                veteran_players = len(filtered_df[filtered_df['Age_numeric'] > 30])
            else:
                young_players = prime_players = veteran_players = 0
            
            age_data = pd.DataFrame({
                'Category': ['Young (<23)', 'Prime (23-30)', 'Veteran (>30)'],
                'Count': [young_players, prime_players, veteran_players]
            })
            
            fig = px.pie(age_data, values='Count', names='Category', 
                         title="Age Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Position balance
            pos_data = pd.DataFrame({
                'Position': filtered_df['Position_Standard'].value_counts().index,
                'Count': filtered_df['Position_Standard'].value_counts().values
            })
            
            fig = px.bar(pos_data, x='Position', y='Count',
                        title="Position Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Key insights
        st.subheader("Key Insights")
        
        insights = [
            f"📊 **Squad Size**: {len(filtered_df)} players analyzed",
            f"⏱️ **Total Minutes**: {filtered_df['Minutes_2425'].sum():,.0f} minutes played",
            f"🏆 **Top Performer**: {filtered_df.loc[filtered_df['Performance_Score'].idxmax(), 'Player']}"
        ]
        
        if 'Age_numeric' in filtered_df.columns:
            insights.insert(1, f"🎂 **Average Age**: {filtered_df['Age_numeric'].mean():.1f} years")
            insights.extend([
                f"🌟 **Youngest Player**: {filtered_df.loc[filtered_df['Age_numeric'].idxmin(), 'Player']} ({filtered_df['Age_numeric'].min():.1f} years)",
                f"👴 **Most Experienced**: {filtered_df.loc[filtered_df['Age_numeric'].idxmax(), 'Player']} ({filtered_df['Age_numeric'].max():.1f} years)"
            ])
        
        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<h2 class="section-header">Strategic Recommendations</h2>', unsafe_allow_html=True)
        
        # Squad management recommendations
        st.subheader("Squad Management")
        
        recommendations = [
            "🎯 **Starting XI Selection**: Focus on players with highest performance scores and appropriate age profiles",
            "🔄 **Rotation Strategy**: Implement effective rotation to prevent burnout and maintain freshness",
            "📈 **Development Focus**: Prioritize young players (under 23) for technical and tactical development",
            "👥 **Leadership**: Utilize veteran players (over 30) for leadership and mentoring roles",
            "⚖️ **Balance**: Maintain optimal age distribution across all positions"
        ]
        
        for rec in recommendations:
            st.markdown(f'<div class="recommendation-box">{rec}</div>', unsafe_allow_html=True)
        
        # Position-specific recommendations
        st.subheader("Position-Specific Recommendations")
        
        for pos in ['GK', 'DF', 'MF', 'FW']:
            pos_data = filtered_df[filtered_df['Position_Standard'] == pos]
            if len(pos_data) > 0:
                with st.expander(f"{pos} Recommendations"):
                    top_player = pos_data.loc[pos_data['Performance_Score'].idxmax()]
                    st.write(f"**Top Performer**: {top_player['Player']} (Score: {top_player['Performance_Score']:.1f})")
                    
                    # Age-specific recommendations
                    if 'Age_numeric' in pos_data.columns:
                        avg_age = pos_data['Age_numeric'].mean()
                        if avg_age < 25:
                            st.write("🔵 **Development Phase**: Focus on technical and tactical development")
                        elif avg_age > 30:
                            st.write("🟡 **Experience Phase**: Utilize for leadership and mentoring")
                        else:
                            st.write("🟢 **Prime Phase**: Optimize performance and consistency")
                    else:
                        st.write("📊 **Analysis Phase**: Focus on performance optimization")
    
    with tab6:
        st.markdown('<h2 class="section-header">Detailed Reports</h2>', unsafe_allow_html=True)
        
        # Player comparison
        st.subheader("Player Comparison")
        
        # Select players to compare
        player_options = filtered_df['Player'].tolist()
        selected_players = st.multiselect("Select players to compare", player_options, default=player_options[:3])
        
        if selected_players:
            display_cols = ['Player', 'Position_Standard', 'Minutes_2425', 'Performance_Score']
            if 'Age_numeric' in filtered_df.columns:
                display_cols.insert(2, 'Age_numeric')
            comparison_df = filtered_df[filtered_df['Player'].isin(selected_players)][display_cols]
            st.dataframe(comparison_df, use_container_width=True)
            
            # Comparison chart
            fig = px.bar(comparison_df, x='Player', y='Performance_Score',
                        title="Performance Score Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        # Export data
        st.subheader("Export Data")
        
        if st.button("Download Filtered Data as CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"como_squad_analysis_{selected_position}_{age_range[0]}-{age_range[1]}.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>Como 1907 Squad Analysis Dashboard | Data Analysis Team | 2024</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
