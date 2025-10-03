#!/usr/bin/env python3
"""
Como 1907 BI Dashboard Launcher
===============================

This script launches the Como 1907 Business Intelligence Dashboard
for stakeholder presentations and analysis.

Usage:
    python launch_dashboard.py

The dashboard will be available at: http://localhost:8501
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = ['streamlit', 'pandas', 'numpy', 'plotly']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_data_files():
    """Check if required data files exist"""
    data_path = Path('data')
    required_files = [
        'como_agecurve_wide.csv',  # Main integrated dataset
        'intermediate/Como_Wage_Breakdown_2425_2526_Cleaned.csv'  # Manual wage data
    ]
    
    optional_files = [
        'intermediate/fbref_20242025.csv',
        'intermediate/fbref_20252026.csv', 
        'intermediate/transfermarkt_contracts.csv',
        'intermediate/capology_wages.csv'  # Scraped data (may be empty)
    ]
    
    missing_files = []
    for file in required_files:
        if not (data_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required data files: {', '.join(missing_files)}")
        print("Please ensure all data files are in the correct locations.")
        return False
    
    # Check optional files
    optional_missing = []
    for file in optional_files:
        if not (data_path / file).exists():
            optional_missing.append(file)
    
    if optional_missing:
        print(f"⚠️  Missing optional data files: {', '.join(optional_missing)}")
        print("Dashboard will still run but with reduced functionality.")
    
    return True

def check_data_quality():
    """Check data quality and provide insights"""
    try:
        import pandas as pd
        data_path = Path('data')
        
        # Check main dataset
        como_df = pd.read_csv(data_path / 'como_agecurve_wide.csv')
        print(f"📊 Main dataset loaded: {como_df.shape[0]} players, {como_df.shape[1]} columns")
        
        # Check wage coverage
        if 'Weekly_Gross_EUR' in como_df.columns:
            wage_coverage = como_df['Weekly_Gross_EUR'].notna().sum()
            print(f"💰 Wage data coverage: {wage_coverage}/{len(como_df)} players ({wage_coverage/len(como_df)*100:.1f}%)")
        
        # Check performance data
        if 'Minutes_2425' in como_df.columns:
            active_players = len(como_df[como_df['Minutes_2425'] >= 90])
            print(f"⚽ Active players (>90 min): {active_players}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Data quality check failed: {e}")
        return False

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    try:
        print("🚀 Launching Como 1907 BI Dashboard...")
        print("📊 Dashboard will be available at: http://localhost:8501")
        print("🔄 Press Ctrl+C to stop the dashboard")
        print("-" * 50)
        
        # Launch streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "como_bi_dashboard.py"])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main launcher function"""
    print("⚽ Como 1907 Business Intelligence Dashboard")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check data files
    if not check_data_files():
        return
    
    # Check data quality
    check_data_quality()
    
    print("\n✅ All checks passed!")
    print()
    
    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main()