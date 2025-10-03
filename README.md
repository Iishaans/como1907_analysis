# FC Como 1907: Comprehensive Squad Analytics Project

## ⚽ Project Overview

This repository contains a comprehensive football analytics project focused on **FC Como 1907**, analyzing the club's squad composition, performance metrics, and strategic positioning as they navigate their recent rise to Serie A. The project provides data-driven insights for squad management, recruitment strategy, and long-term sustainability planning.

### Club Context
FC Como 1907 has experienced a remarkable resurgence in recent years, climbing from Serie C to Serie A through strategic recruitment and tactical innovation. This analysis examines their current squad through multiple analytical lenses to support data-driven decision making in their Serie A campaign.

---

## 🎯 Analytical Framework

The project employs three complementary analytical lenses to provide a holistic view of Como's squad:

### 1. 📊 Performance Analysis
- **Player Metrics**: Individual performance statistics, playing time analysis, and contribution assessment
- **Positional Breakdown**: Position-specific performance evaluation (GK, DF, MF, FW)
- **Tactical Contributions**: Progressive passing, final third touches, and attacking involvement
- **Market Value Correlation**: Performance-to-value analysis for recruitment insights

### 2. 🎂 Age Analysis
- **Squad Age Distribution**: Understanding the age profile across positions
- **Career Stage Balance**: Identifying young prospects, prime performers, and experienced leaders
- **Long-term Sustainability**: Assessing squad development trajectory and succession planning
- **Age Curve Modeling**: Predicting performance peaks and decline phases

### 3. 💰 Salary/Wage Analysis
- **Wage Bill Allocation**: Distribution of financial resources across positions and players
- **Cost Efficiency**: Performance-per-euro analysis for value assessment
- **Market Alignment**: Comparing wages to market values and league standards
- **Budget Optimization**: Identifying high and low ROI players for strategic decisions

---

## 📁 Repository Structure

```
como1907_analysis/
├── 📊 como_analysis.ipynb          # Comprehensive EDA and analysis notebook
├── 🏗️ como_agecurve_builder.py     # Data collection and processing pipeline
├── 🎛️ como_bi_dashboard.py         # Interactive Streamlit dashboard
├── 🚀 launch_dashboard.py          # Dashboard launcher with checks
├── 🧪 test_dashboard.py            # Testing and validation scripts
├── 📋 requirements.txt             # Python dependencies
├── 📖 README.md                    # This file
│
├── 📁 data/                        # Data storage
│   ├── como_agecurve_wide.csv      # Master dataset (59 players)
│   └── intermediate/               # Raw data sources
│       ├── fbref_20242025.csv      # FBref 2024-25 season data
│       ├── fbref_20252026.csv      # FBref 2025-26 season data
│       ├── transfermarkt_contracts.csv # Market values and contracts
│       └── capology_wages.csv      # Wage and salary data
│
└── 📚 Documentation/
    ├── README_BI_Dashboard.md      # Dashboard usage guide
    ├── README_como_agecurve_builder.md # Data pipeline documentation
    └── FIX_SUMMARY.md              # Technical issue resolutions
```

---

## 🔬 Methodology

### Data Sources
- **FBref**: Player statistics, minutes played, positional data, and performance metrics
- **Transfermarkt**: Market values, contract information, and player valuations
- **Manual Wage Data**: Comprehensive salary breakdown for 35 players (Capology web scraper temporarily unavailable)
- **Time Period**: 2024-25 and 2025-26 seasons (current and previous)

### Key Metrics Applied
- **Performance Metrics**: Minutes per 90, xG+xAG, progressive passes, final third touches
- **Age Metrics**: Career stage classification, prime age analysis, development potential
- **Financial Metrics**: Market value, wage efficiency, cost per performance unit
- **Positional Metrics**: Role-specific performance indicators and tactical contributions

### Analytical Approaches
- **Descriptive Analytics**: Squad composition, distribution analysis, summary statistics
- **Comparative Analytics**: Position-based comparisons, age group analysis, performance benchmarking
- **Predictive Analytics**: Age curve modeling, performance projection, career trajectory analysis
- **Prescriptive Analytics**: Strategic recommendations, recruitment priorities, squad optimization

---

## 🏆 Key Insights & Recommendations

### Squad Composition Analysis
- **Total Squad Size**: 59 players analyzed
- **Positional Distribution**: 11 Defenders, 9 Forwards, 6 Midfielders, 4 Goalkeepers
- **Playing Time**: 37,428 total minutes in 2024-25, 4,938 minutes in 2025-26 (early season)
- **Data Quality**: Fully deduplicated dataset with comprehensive player coverage

### Performance Insights
- **Top Performers**: Players with highest minutes and performance scores identified
- **Positional Strengths**: Areas of squad depth and potential gaps
- **Tactical Contributions**: Progressive passing and attacking involvement analysis
- **Development Potential**: Young players showing promise for future development

### Age Profile Analysis
- **Squad Balance**: Mix of experienced veterans and emerging talents
- **Career Stages**: Identification of players in different development phases
- **Succession Planning**: Key positions requiring future reinforcement
- **Long-term Sustainability**: Age curve analysis for strategic planning

### Financial Efficiency
- **Total Wage Bill**: €45.2m annual salary expenditure (59.3% squad coverage)
- **Market Value Analysis**: Player valuations and market positioning
- **Wage Optimization**: Cost-effective performers and potential overpayments
- **ROI Assessment**: Performance per euro spent analysis (cost per minute analysis)
- **Budget Allocation**: Strategic distribution of financial resources by position
- **Top Earner Analysis**: Maxence Caqueret leads at €128k/week, Dele Alli follows at €107k/week

### Strategic Recommendations
1. **Squad Management**: Focus on developing young talent while maintaining experience
2. **Recruitment Strategy**: Target specific positions and age profiles for optimization
3. **Performance Monitoring**: Implement regular assessment of key performance indicators
4. **Financial Planning**: Optimize wage structure for maximum efficiency
5. **Long-term Vision**: Plan for squad evolution and Serie A sustainability

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Required packages: `streamlit`, `pandas`, `numpy`, `plotly`, `matplotlib`, `seaborn`

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd como1907_analysis

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

#### Interactive Dashboard
```bash
# Launch the comprehensive BI dashboard
python launch_dashboard.py

# Or manually
streamlit run como_bi_dashboard.py
```

#### Jupyter Notebook Analysis
```bash
# Open the comprehensive analysis notebook
jupyter notebook como_analysis.ipynb
```

#### Data Pipeline
```bash
# Rebuild the dataset from source
python como_agecurve_builder.py
```

---

## 📈 Dashboard Features

The interactive Streamlit dashboard provides:

- **Squad Overview**: Key metrics, position distribution, and top performers
- **Performance Analysis**: Position-specific scoring and player comparisons
- **Key Insights**: Data-driven insights and statistical summaries
- **Strategic Recommendations**: Actionable guidance for squad management
- **Detailed Reports**: Player comparison tools and export functionality

### Dashboard Sections
1. **Squad Overview**: High-level metrics and visualizations
2. **Performance Analysis**: Detailed player and position analysis
3. **Key Insights**: Statistical summaries and trends
4. **Strategic Recommendations**: Data-driven guidance
5. **Detailed Reports**: Export and comparison tools

---

## 🎯 Target Audience

This analysis is designed for:
- **Sports Data Analysts**: Comprehensive statistical analysis and methodology
- **Football Operations Staff**: Strategic insights for squad management
- **Recruitment Teams**: Data-driven player evaluation and market analysis
- **Club Management**: High-level insights for strategic decision making
- **Technical Staff**: Detailed performance metrics and tactical analysis

---

## 📊 Data Quality & Validation

- **Completeness**: 59 players with comprehensive data coverage
- **Accuracy**: Validated against multiple data sources
- **Consistency**: Standardized metrics and unified player identification
- **Timeliness**: Current season data with historical context
- **Deduplication**: Clean, single-row-per-player dataset

---

## 🔮 Future Enhancements

- **Real-time Data Integration**: Live performance tracking and updates
- **Advanced Analytics**: Machine learning models for performance prediction
- **Comparative Analysis**: League-wide benchmarking and competitive analysis
- **Tactical Analysis**: Formation-specific performance evaluation
- **Injury Analysis**: Availability and fitness impact assessment

---

## 📞 Support & Contact

For questions, suggestions, or collaboration opportunities:
- **Technical Issues**: Check the documentation files for troubleshooting
- **Data Questions**: Review the methodology section for metric definitions
- **Analysis Requests**: Consider the analytical framework for new insights

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*This analysis represents a comprehensive approach to football analytics, combining performance data, age analysis, and financial metrics to provide actionable insights for FC Como 1907's continued success in Serie A.*
