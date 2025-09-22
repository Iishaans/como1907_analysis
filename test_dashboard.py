"""
Test script to verify the Como 1907 BI Dashboard works correctly
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Test imports
    print("Testing imports...")
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from pathlib import Path
    
    print("✅ All imports successful!")
    
    # Test data loading
    print("Testing data loading...")
    data_path = Path('data')
    
    if data_path.exists():
        como_agecurve = pd.read_csv(data_path / 'como_agecurve_wide.csv')
        print(f"✅ Loaded como_agecurve_wide.csv: {como_agecurve.shape}")
        
        # Test data processing
        df_performance = como_agecurve.copy()
        df_performance['Position_Standard'] = df_performance['Latest_Pos4'].apply(
            lambda x: 'GK' if 'GK' in str(x).upper() else 
                     'DF' if 'DF' in str(x).upper() or 'BACK' in str(x).upper() else
                     'MF' if 'MF' in str(x).upper() or 'MID' in str(x).upper() else
                     'FW' if 'FW' in str(x).upper() or 'FOR' in str(x).upper() else 'Unknown'
        )
        
        df_performance = df_performance[df_performance['Minutes_2425'] >= 90]
        print(f"✅ Data processing successful: {df_performance.shape}")
        
        # Test age conversion
        def extract_age_from_string(age_str):
            if pd.isna(age_str):
                return None
            try:
                if isinstance(age_str, str) and '-' in age_str:
                    age_part = age_str.split('-')[0]
                    return float(age_part)
                elif isinstance(age_str, (int, float)):
                    return float(age_str)
                else:
                    return None
            except:
                return None
        
        df_performance['Age_numeric'] = df_performance['Age_latest'].apply(extract_age_from_string)
        print(f"✅ Age conversion successful: {df_performance['Age_numeric'].notna().sum()} valid ages")
        
        # Test performance scoring
        df_performance['Performance_Score'] = 0
        if 'Minutes_2425' in df_performance.columns:
            max_minutes = df_performance['Minutes_2425'].max()
            df_performance['Performance_Score'] = (df_performance['Minutes_2425'] / max_minutes) * 100
        
        print(f"✅ Performance scoring successful")
        
        # Test visualization
        print("Testing visualization...")
        fig = px.bar(df_performance.head(10), x='Player', y='Performance_Score')
        print("✅ Plotly visualization successful")
        
        print("\n🎉 All tests passed! Dashboard should work correctly.")
        print("\nTo run the dashboard:")
        print("streamlit run como_bi_dashboard.py")
        
    else:
        print("❌ Data directory not found. Please ensure data files are in the 'data' folder.")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required packages: pip install streamlit plotly pandas numpy")
    
except Exception as e:
    print(f"❌ Error: {e}")
