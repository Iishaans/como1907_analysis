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
        'como_agecurve_wide.csv',
        'intermediate/fbref_20242025.csv',
        'intermediate/fbref_20252026.csv',
        'intermediate/transfermarkt_contracts.csv',
        'intermediate/capology_wages.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not (data_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing data files: {', '.join(missing_files)}")
        print("Please ensure all data files are in the correct locations.")
        return False
    
    return True

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
    
    print("✅ All requirements satisfied!")
    print("✅ Data files found!")
    print()
    
    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main()
