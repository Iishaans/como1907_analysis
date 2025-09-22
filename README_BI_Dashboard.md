# Como 1907 Business Intelligence Dashboard

## Overview
A comprehensive Streamlit dashboard that consolidates key insights from the Como 1907 squad analysis for stakeholder presentations and decision-making.

## Features

### 📊 **Squad Overview**
- Key performance metrics and statistics
- Position distribution analysis
- Age distribution visualization
- Top performers identification

### ⚽ **Performance Analysis**
- Position-specific performance scoring
- Minutes vs Age correlation analysis
- Performance trends by position
- Detailed player comparisons

### 📈 **Key Insights**
- Squad balance analysis (age and position)
- Performance indicators
- Statistical summaries
- Data-driven insights

### 🎯 **Strategic Recommendations**
- Squad management guidance
- Position-specific recommendations
- Development focus areas
- Transfer strategy insights

### 📋 **Detailed Reports**
- Player comparison tools
- Export functionality
- Custom filtering options
- Comprehensive data tables

## Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Quick Launch (Recommended)**
   ```bash
   python launch_dashboard.py
   ```

3. **Manual Launch**
   ```bash
   streamlit run como_bi_dashboard.py
   ```

4. **Access the Dashboard**
   - Open your browser to `http://localhost:8501`
   - The dashboard will automatically load your data

## Usage

### Dashboard Navigation
- **Sidebar Controls**: Filter by position, age range, and minimum minutes
- **Tab Navigation**: Switch between different analysis sections
- **Interactive Elements**: Click, hover, and explore visualizations

### Key Features
- **Real-time Filtering**: Adjust filters to focus on specific player groups
- **Interactive Charts**: Hover for details, click to explore
- **Export Capabilities**: Download filtered data as CSV
- **Responsive Design**: Works on desktop and mobile devices

## Data Requirements

The dashboard expects the following data files in the `data/` directory:
- `como_agecurve_wide.csv` - Main squad data
- `intermediate/fbref_20242025.csv` - 2024-25 season data
- `intermediate/fbref_20252026.csv` - 2025-26 season data
- `intermediate/transfermarkt_contracts.csv` - Market values
- `intermediate/capology_wages.csv` - Wage data

## Performance Metrics

### Scoring System
- **Minutes Contribution** (40%): Based on playing time
- **Age Factor** (30%): Prime age optimization
- **Market Value** (30%): Transfer market assessment

### Position Analysis
- **Goalkeepers**: Focus on consistency and experience
- **Defenders**: Emphasis on defensive contribution
- **Midfielders**: Balance of creativity and control
- **Forwards**: Attacking threat and goal contribution

## Customization

### Styling
- Professional Como 1907 branding
- Responsive design elements
- Custom CSS for enhanced presentation

### Data Processing
- Automatic position standardization
- Market value extraction and normalization
- Performance score calculation
- Age curve analysis

## Stakeholder Presentation

### Key Sections for Presentations
1. **Executive Summary**: High-level squad overview
2. **Performance Analysis**: Detailed player assessments
3. **Strategic Insights**: Data-driven recommendations
4. **Action Items**: Specific next steps

### Export Options
- CSV downloads for further analysis
- Screenshot capabilities for presentations
- Interactive exploration during meetings

## Technical Details

### Built With
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data processing and analysis
- **Matplotlib/Seaborn**: Statistical plotting

### Performance
- Cached data loading for fast performance
- Optimized visualizations
- Responsive design for all devices

## Support

For technical support or feature requests, contact the Data Analysis Team.

---

**Como 1907 Squad Analysis Dashboard**  
Data Analysis Team | 2024
