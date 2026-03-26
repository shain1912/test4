import sqlite3
import os

DB_PATH = '/home/shain/dev/test4/interviews.db'

def init():
    print(f"Initializing database at {DB_PATH}")
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            print("Removed existing invalid DB file.")
        except Exception as e:
            print(f"Error removing existing file: {e}")
            
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DROP TABLE IF EXISTS interviews')
        c.execute('''
            CREATE TABLE interviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                issue_text TEXT,
                severity_score INTEGER,
                primary_category TEXT,
                location_bucket TEXT,
                evidence_span TEXT,
                raw_log TEXT
            )
        ''')
        conn.commit()
        conn.close()
        print("✅ DB Init Success!")
    except Exception as e:
        print(f"❌ DB Init Failed: {e}")

if __name__ == '__main__':
    init()
