import sqlite3
import os
import pandas as pd

# Path as defined in src/db.py
DB_PATH = os.path.join(os.path.dirname(__file__), 'src', 'interviews.db')

print(f"Checking {os.path.abspath(DB_PATH)}...")
if os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM interviews LIMIT 5", conn)
        print(f"Found 'interviews' table with {len(df)} rows (showing top 5).")
        if not df.empty:
            print("Columns:", df.columns.tolist())
            print("Sample Data:", df.iloc[0].to_dict())
        else:
            print("Table is empty (0 rows).")
    except Exception as e:
        print(f"Error reading DB: {e}")
    conn.close()
else:
    print(f"File not found at {DB_PATH}")
