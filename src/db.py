import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'interviews.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS interviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            location TEXT,
            urban_element TEXT,
            issue TEXT,
            solution_type TEXT,
            solution_detail TEXT,
            solution_logic TEXT,
            primary_value TEXT,
            willingness_to_pay TEXT,
            raw_log TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_interview(data):
    """
    Inserts interview data into the database.
    Accepts a dictionary 'data' which can be flat (InterviewInfo style) 
    or nested (old style support).
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Try flat access first, fall back to nested if needed (or just keys)
    location = data.get('location', '')
    urban_element = data.get('urban_element', '')
    issue = data.get('issue', '')
    
    # Handle Solution (Support both flat and nested 'proposed_solution')
    if 'proposed_solution' in data and isinstance(data['proposed_solution'], dict):
        sol = data['proposed_solution']
        solution_type = sol.get('type', '')
        solution_detail = sol.get('detail', '')
        solution_logic = sol.get('user_logic', '')
    else:
        solution_type = data.get('solution_type', '')
        solution_detail = data.get('solution_detail', '')
        solution_logic = data.get('solution_logic', '')
    
    # Handle Value (Support both flat and nested 'value_analysis')
    if 'value_analysis' in data and isinstance(data['value_analysis'], dict):
        val = data['value_analysis']
        primary_value = val.get('primary_value', '')
        willingness_to_pay = val.get('willingness_to_pay', '')
    else:
        primary_value = data.get('primary_value', '')
        willingness_to_pay = data.get('willingness_to_pay', '')
    
    # Save full data blob as raw_log for debugging
    raw_log = json.dumps(data, ensure_ascii=False) 

    c.execute('''
        INSERT INTO interviews (
            timestamp, location, urban_element, issue, 
            solution_type, solution_detail, solution_logic, 
            primary_value, willingness_to_pay, raw_log
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp, location, urban_element, issue,
        solution_type, solution_detail, solution_logic,
        primary_value, willingness_to_pay, raw_log
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
