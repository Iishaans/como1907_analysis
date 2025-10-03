# FC Como 1907: Comprehensive Squad Analytics Project

## ⚽ Project Overview

This repository contains a comprehensive football analytics project focused on **FC Como 1907**, analyzing the club's squad composition, performance metrics, and strategic positioning as they navigate their rise to Serie A. The project provides data-driven insights for squad management, recruitment strategy, and long-term sustainability planning.

### Club Context
FC Como 1907 has experienced a remarkable resurgence, climbing to Serie A through strategic recruitment and tactical innovation. This analysis examines their current squad through multiple analytical lenses to support data-driven decision making in their Serie A campaign.

---

## 🎯 Analytical Framework

The project employs three complementary analytical lenses to provide a holistic view of Como's squad:

### 1. 📊 Performance Analysis
- **Player Metrics**: Individual performance statistics, playing time analysis, and contribution assessment
- **Positional Breakdown**: Position-specific performance evaluation (GK, DF, MF, FW)
- **Tactical Contributions**: Minutes played, performance scores, and playing time distribution
- **Performance Scoring**: Composite scoring system based on minutes, age optimization, and market value

### 2. 🎂 Age Analysis
- **Squad Age Distribution**: Understanding the age profile across positions
- **Career Stage Balance**: Identifying young prospects, prime performers, and experienced leaders
- **Age Optimization Scoring**: Prime age bonus (23-30) vs prospect penalties
- **Long-term Sustainability**: Assessing squad development trajectory

### 3. 💰 Salary/Wage Analysis
- **Wage Bill Analysis**: Total weekly/annual expenditure breakdown
- **Top Earners Analysis**: Highest-paid players and salary distribution
- **Position-Based Salary Analysis**: Average wages by playing position
- **Cost Efficiency**: Performance vs salary correlation analysis
- **Data Source**: Manual Capology wage breakdown (scraped data excluded due to instability)

---

## 📁 Repository Structure

```
como1907_analysis/
├── 📊 como_bi_dashboard.py         # Main Streamlit dashboard
├── 🚀 launch_dashboard.py         # Dashboard launcher with checks
├── 🧪 test_dashboard.py           # Comprehensive testing suite
├── 📋 requirements.txt           # Python dependencies
├── 📖 README.md                  # This comprehensive guide
│
├── 📁 data/                      # Data storage
│   ├── como_agecurve_wide.csv   # Master dataset (59-61 players)
│   └── intermediate/            # Raw data sources
│       ├── Como_Wage_Breakdown_2425_2526_Cleaned.csv  # Manual wage data
│       ├── fbref_20242025.csv   # FBref 2024-25 season data
│       ├── fbref_20252026.csv   # FBref 2025-26 season data
│       ├── transfermarkt_contracts.csv  # Market values
│       └── capology_wages.csv   # Backup scraped wages
│
├── 📚 Documentation/
│   ├── README_BI_Dashboard.md    # Dashboard-specific documentation
│   ├── README_como_agecurve_builder.md  # Data pipeline docs
│   └── FIX_SUMMARY.md           # Technical issue resolutions
│
└── 🌐 Website/                  # Next.js application
    ├── src/pages/index.js       # Main dashboard page
    ├── components/              # Reusable UI components
    ├── lib/data.js              # Data processing utilities
    └── package.json             # Node.js dependencies
```

---

## 🔬 Data Sources & Methodology

### Data Sources
- **como_agecurve_wide.csv**: Master dataset with integrated data from all sources
- **FBref**: Player statistics, minutes played, positional data, and performance metrics
- **Transfermarkt**: Market values, contract information, and player valuations
- **Manual Wage Data**: Comprehensive salary breakdown for 35+ players (authoritative source)
- **Time Period**: 2024-25 and 2025-26 seasons coverage

### Data Quality
- **✅ Fully Deduplicated**: No duplicate players after sophisticated merging
- **✅ Comprehensive Coverage**: 59 players with varying data completeness
- **✅ Wage Data**: 35+ players with complete salary information (59.3% coverage)
- **✅ Performance Data**: All players with sufficient minutes (90+ minutes threshold)
- **✅ Validation**: Automated quality checks and error handling

### Key Metrics Applied
- **Performance Metrics**: Minutes per season, positional performance, tactical contributions
- **Age Metrics**: Career stage analysis, prime age identification (23-30), prospect value
- **Financial Metrics**: Weekly/annual gross wages, cost efficiency ratios, market alignment
- **Positional Metrics**: GK/DF/MF/FW specific analysis and recommendations

---

## 🏆 Key Insights & Strategic Recommendations

### Financial Insights
- **Total Wage Bill**: €45.2m annual salary expenditure (calculated from €868k weekly)
- **Top Earners**: Maxence Caqueret leads at €128k/week, Dele Alli follows at €107k/week
- **Position Distribution**: Midfielders average highest salaries (€32.6k/week)
- **Cost Efficiency**: Significant variation in performance-to-cost ratios across squad

### Performance Insights
- **Active Squad**: 32+ players with sufficient playing time (90+ minutes)
- **Position Balance**: Strong coverage across all position groups
- **Age Distribution**: Optimal mix of youth prospects and experienced veterans
- **Development Potential**: Multiple U23 players with high upside

### Strategic Recommendations
1. **Starting XI Optimization**: Focus on highest performance scores within prime age windows
2. **Rotation Management**: Implement effective rotation to prevent fatigue and maintain freshness
3. **Development Pipeline**: Prioritize U23 talent with high performance scores
4. **Leadership Structure**: Leverage 30+ veterans for mentoring and succession planning
5. **Financial Efficiency**: Optimize wage allocation based on performance-to-cost ratios
6. **Recruitment Strategy**: Target positions with age gaps or salary inefficiencies

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.9+
- Node.js 18+ (for Next.js app)
- Required Python packages: `streamlit`, `pandas`, `numpy`, `plotly`

### Streamlit Dashboard (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Launch with health checks
python launch_dashboard.py

# Or launch directly
streamlit run como_bi_dashboard.py
```
**Dashboard will be available at:** http://localhost:8501

### Next.js Application
```bash
# Navigate to website directory
cd Website

# Install dependencies
npm install

# Start development server
npm run dev
```
**Website will be available at:** http://localhost:3000

### Testing
```bash
# Run comprehensive test suite
python test_dashboard.py
```

---

## 📊 Dashboard Features

### Streamlit Dashboard
- **📊 Squad Overview**: Key metrics, position distribution, top performers
- **⚽ Performance Analysis**: Position-specific scoring and analysis
- **💰 Salary Analysis**: Comprehensive wage breakdown and insights
- **📈 Key Insights**: Data-driven strategic insights and recommendations
- **🎛️ Interactive Controls**: Position, age, and minutes filters
- **📈 Visualizations**: Interactive charts and graphs

### Next.js Website
- **🎨 Modern UI**: Shadcn/UI components for professional presentation
- **📱 Responsive Design**: Works on desktop, tablet, and mobile
- **🔍 Interactive Filters**: Real-time data filtering and analysis
- **📊 Rich Visualizations**: Advanced charts powered by modern libraries
- **⚡ Performance**: Fast loading and smooth interactions

---

## 🎯 Target Audience

This analysis is designed for:
- **Sports Data Analysts**: Comprehensive statistical analysis and methodology
- **Football Operations Staff**: Strategic insights for squad management
- **Recruitment Teams**: Data-driven player evaluation and market analysis
- **Club Management**: High-level insights for strategic decision making
- **Technical Staff**: Detailed performance metrics and tactical analysis
- **Stakeholders**: Clear visualizations for presentations and meetings

---

## 🔄 Data Updates & Maintenance

### Data Pipeline
1. **FBref Scraping**: Automated season data extraction
2. **Manual Wage Updates**: Club-curated salary data maintenance
3. **Integration Process**: Automated deduplication and merging
4. **Quality Validation**: Comprehensive data quality checks

### Regular Maintenance
- **Season Updates**: New FBref data integration
- **Wage Reviews**: Manual salary data updates
- **Performance Recalibration**: Scoring algorithm refinements
- **Dashboard Enhancements**: New features and visualizations

---

## 🧪 Testing & Validation

### Automated Testing
- **Data Quality Tests**: Schema validation and completeness checks
- **Function Testing**: Core analytical functions verification
- **Visualization Testing**: Chart generation and interaction tests
- **Dashboard Testing**: End-to-end dashboard functionality

### Manual Validation
- **Data Accuracy**: Cross-reference with original sources
- **Business Logic**: Validate analytical approaches and scoring algorithms
- **Visualization Quality**: Ensure charts and insights are clear and accurate

---

## 🌐 Web Application (Next.js)

The project includes a modern Next.js web application with Shadcn/UI components, providing the same analytical capabilities as the Streamlit dashboard but with enhanced user experience and professional presentation suitable for stakeholder meetings.

### Key Features
- **Professional UI**: Clean, modern interface with Shadcn components
- **Responsive Design**: Mobile-first approach with adaptive layouts
- **Interactive Analytics**: Real-time filtering and data exploration
- **Export Capabilities**: Data download and report generation
- **Performance Optimized**: Fast loading and smooth interactions

---

## 📞 Support & Contributing

### Technical Support
- **Data Issues**: Check data pipeline documentation and validation reports
- **Dashboard Problems**: Use testing suite to identify specific issues
- **Feature Requests**: Review documentation and existing capabilities

### Contributing
- **Data Updates**: Follow established pipeline and validation procedures
- **Feature Development**: Maintain consistency with existing analytical framework
- **Documentation**: Keep README files updated with changes

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎉 Acknowledgments

- **FC Como 1907**: For the club data and analytical requirements
- **FBref**: Comprehensive football statistics and data
- **Transfermarkt**: Market valuations and contract information
- **Capology**: Professional salary and wage data
- **Data Analysis Team**: For methodology development and implementation

---

*This analysis represents a comprehensive approach to modern football analytics, combining performance data, age analysis, and financial metrics to provide actionable insights for FC Como 1907's continued success in Serie A.*