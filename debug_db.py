import os
import sys

# Add src to path just in case
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from src.db import DB_PATH
    print(f"Path defined in src.db: {os.path.abspath(DB_PATH)}")
except ImportError:
    # Manual check if import fails
    print("Could not import src.db")
    DB_PATH = os.path.join(os.path.dirname(__file__), 'interviews.db')

candidates = [
    os.path.abspath("interviews.db"),
    os.path.abspath("src/interviews.db"),
    os.path.abspath("../interviews.db"),
    DB_PATH
]

print("\n--- Checking Candidates ---")
for cand in set(candidates): # set to allow duplicates
    if os.path.exists(cand):
        size = os.path.getsize(cand)
        print(f"[FOUND] {cand} (Size: {size} bytes)")
        import sqlite3
        conn = sqlite3.connect(cand)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT count(*) FROM interviews")
            count = cursor.fetchone()[0]
            print(f"   -> Row count: {count}")
        except Exception as e:
            print(f"   -> Error checking rows: {e}")
        conn.close()
    else:
        print(f"[MISSING] {cand}")
