"""
Test script to verify the Como 1907 BI Dashboard works correctly
Updated for comprehensive testing with resolved data sets.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all required imports"""
    try:
        print("Testing imports...")
        import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go
        from pathlib import Path
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_loading():
    """Test data loading and basic validation"""
    try:
        print("\nTesting data loading...")
        import pandas as pd
        from pathlib import Path
        
        data_path = Path('data')
        
        if not data_path.exists():
            print("❌ Data directory not found.")
            return False
        
        # Load main dataset
        como_agecurve = pd.read_csv(data_path / 'como_agecurve_wide.csv')
        print(f"✅ Loaded como_agecurve_wide.csv: {como_agecurve.shape}")
        
        # Check required columns
        required_cols = ['Player', 'Minutes_2425', 'Latest_Pos4']
        missing_cols = [col for col in required_cols if col not in como_agecurve.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        print("✅ All required columns present")
        
        # Load intermediate datasets
        try:
            fbref_2425 = pd.read_csv(data_path / 'intermediate' / 'fbref_20242025.csv')
            print(f"✅ Loaded fbref_20242025.csv: {fbref_2425.shape}")
        except FileNotFoundError:
            print("⚠️  fbref_20242025.csv not found (optional)")
        
        try:
            transfermarkt = pd.read_csv(data_path / 'intermediate' / 'transfermarkt_contracts.csv')
            print(f"✅ Loaded transfermarkt_contracts.csv: {transfermarkt.shape}")
        except FileNotFoundError:
            print("⚠️  transfermarkt_contracts.csv not found (optional)")
        
        # Check wage data
        try:
            wage_data = pd.read_csv(data_path / 'intermediate' / 'Como_Wage_Breakdown_2425_2526_Cleaned.csv')
            print(f"✅ Loaded manual wage data: {wage_data.shape}")
        except FileNotFoundError:
            print("⚠️  Manual wage data not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        return False

def test_data_processing():
    """Test data processing functions"""
    try:
        print("\nTesting data processing...")
        
        # Import data processing functions
        import pandas as pd
        df = pd.read_csv('data/como_agecurve_wide.csv')
        
        # Test position standardization
        def standardize_position(pos_str):
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
        
        df['Position_Standard'] = df['Latest_Pos4'].apply(standardize_position)
        print(f"✅ Position standardization successful")
        
        # Test performance scoring
        df['Performance_Score'] = 0
        
        if 'Minutes_2425' in df.columns:
            max_minutes = df['Minutes_2425'].max()
            if max_minutes > 0:
                df['Minutes_Score'] = (df['Minutes_2425'] / max_minutes) * 40
                df['Performance_Score'] += df['Minutes_Score']
        print("✅ Performance scoring successful")
        
        # Test filtering
        if 'Minutes_2425' in df.columns:
            filtered_df = df[df['Minutes_2425'].fillna(0) >= 90]
            print(f"✅ Filtering successful: {len(filtered_df)}/{len(df)} players after filter")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing error: {e}")
        return False

def test_wage_analysis():
    """Test wage analysis functionality"""
    try:
        print("\nTesting wage analysis...")
        
        import pandas as pd
        df = pd.read_csv('data/como_agecurve_wide.csv')
        
        # Check wage data presence
        if 'Weekly_Gross_EUR' in df.columns:
            wage_data = df.dropna(subset=['Weekly_Gross_EUR'])
            if len(wage_data) > 0:
                print(f"✅ Wage analysis successful: {len(wage_data)} players with wage data")
                print(f"   Total weekly wage bill: €{wage_data['Weekly_Gross_EUR'].sum():,.0f}")
                
                # Test top earners
                top_earner = wage_data.nlargest(1, 'Weekly_Gross_EUR')
                if len(top_earner) > 0:
                    print(f"   Top earner: {top_earner.iloc[0]['Player']}")
                return True
            else:
                print("⚠️  No wage data available after filtering")
                return False
        else:
            print("⚠️  Weekly_Gross_EUR column not found")
            return False
            
    except Exception as e:
        print(f"❌ Wage analysis error: {e}")
        return False

def test_visualization():
    """Test plotting and visualization"""
    try:
        print("\nTesting visualization...")
        
        import pandas as pd
        import plotly.express as px
        
        # Load sample data
        df = pd.read_csv('data/como_agecurve_wide.csv')
        
        # Test basic plots
        df_test = df.head(10)
        
        # Test bar chart
        fig1 = px.bar(df_test, x='Player', y='Minutes_2425', title="Minutes Played")
        print("✅ Bar chart creation successful")
        
        # Test scatter plot
        fig2 = px.scatter(df_test, x='Minutes_2425', y='Minutes_2526', title="Minutes Comparison")
        print("✅ Scatter plot creation successful")
        
        # Test position distribution (if data available)
        if 'Latest_Pos4' in df.columns:
            pos_counts = df['Latest_Pos4'].value_counts()
            fig3 = px.pie(values=pos_counts.values, names=pos_counts.index, 
                         title="Position Distribution")
            print("✅ Pie chart creation successful")
        
        print("✅ All visualization tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False

def test_dashboard_functions():
    """Test dashboard-specific functions"""
    try:
        print("\nTesting dashboard functions...")
        
        # Try to import dashboard functions
        sys.path.append('.')
        try:
            from como_bi_dashboard import create_performance_score, create_wage_analysis, standardize_position
            print("✅ Dashboard functions imported successfully")
            
            # Test with sample data
            import pandas as pd
            df = pd.read_csv('data/como_agecurve_wide.csv')
            
            # Test performance scoring
            df_scored = create_performance_score(df.head(20))
            print("✅ Performance scoring function works")
            
            # Test wage analysis
            wage_result = create_wage_analysis(df.head(20))
            print("✅ Wage analysis function works")
            
            return True
            
        except ImportError as e:
            print(f"⚠️  Could not import dashboard functions: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Dashboard function test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Como 1907 BI Dashboard Testing Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_loading,
        test_data_processing,
        test_wage_analysis,
        test_visualization,
        test_dashboard_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard should work correctly.")
        print("\nTo run the dashboard:")
        print("python launch_dashboard.py")
        print("or")
        print("streamlit run como_bi_dashboard.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()