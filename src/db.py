import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'interviews.db')

def init_db():
    # Force reset for the new schema
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Simple migration: drop table if exists
    c.execute('DROP TABLE IF EXISTS interviews')
    c.execute('''
        CREATE TABLE interviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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

def insert_interview(data):
    """
    Inserts interview data into the database.
    Expects data matching the new InterviewInfo structure.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract fields safely
    issue_text = data.get('issue_text', '')
    severity_score = data.get('severity_score', 0)
    primary_category = data.get('primary_category', '')
    location_bucket = data.get('location_bucket', '')
    evidence_span = data.get('evidence_span', '')
    
    # Save full data blob as raw_log
    raw_log = json.dumps(data, ensure_ascii=False) # Store everything just in case

    c.execute('''
        INSERT INTO interviews (
            timestamp, issue_text, severity_score, primary_category, 
            location_bucket, evidence_span, raw_log
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp, issue_text, severity_score, primary_category,
        location_bucket, evidence_span, raw_log
    ))
    
    conn.commit()
    conn.close()

def get_all_interviews():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM interviews ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]
