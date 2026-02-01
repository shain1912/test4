import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Load env for API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

from src.db import get_all_interviews
from src.analysis import SemanticAnalyzer

print("--- Starting Debug Analysis ---")

# 1. Check Data
try:
    data = get_all_interviews()
    print(f"1. Database Loaded. Rows: {len(data)}")
    if len(data) == 0:
        print("!! CRITICAL: No data in DB.")
        sys.exit(1)
except Exception as e:
    print(f"!! CRITICAL: DB Error: {e}")
    sys.exit(1)

df = pd.DataFrame(data)
target_col = 'issue_text' if 'issue_text' in df.columns else 'issue'
print(f"2. Target Text Column: {target_col}")

# 2. Check Analyzer Init
print("3. Initializing SemanticAnalyzer...")
if not api_key:
    print("!! CRITICAL: OPENAI_API_KEY is missing.")
    sys.exit(1)

try:
    analyzer = SemanticAnalyzer(api_key=api_key)
    print("   -> Analyzer Initialized.")
except Exception as e:
    print(f"!! CRITICAL: Analyzer Init Failed: {e}")
    sys.exit(1)

# 3. Run Analysis
print("4. Running process_and_analyze (n_dimensions=3)...")
try:
    result_df = analyzer.process_and_analyze(df, text_column=target_col, n_dimensions=3)
    print("   -> Analysis Returned.")
except Exception as e:
    print(f"!! CRITICAL: Analysis Phase Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Inspect Result
print("\n--- Inspection Results ---")
print(f"Columns: {result_df.columns.tolist()}")

required_cols = ['x', 'y', 'z', 'topic_label', 'cluster']
missing = [c for c in required_cols if c not in result_df.columns]

if missing:
    print(f"!! CRITICAL: Missing columns in result: {missing}")
    print("This explains why the dashboard shows nothing.")
else:
    print("SUCCESS: All required columns (x, y, z, topic_label) are present.")
    print("\nSample Data (First 3 rows):")
    print(result_df[required_cols].head(3))
    
    print("\nCluster Distribution:")
    print(result_df['topic_label'].value_counts())

print("\n--- End Debug ---")
