import sqlite3
import json
import os
import uuid
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'interviews.db')


def init_db():
    """Initialize database with new schema supporting multiple issues per session."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Force reset for the new schema
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


def migrate_db():
    """Add session_id column if it doesn't exist (for existing databases)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Check if session_id column exists
    c.execute("PRAGMA table_info(interviews)")
    columns = [col[1] for col in c.fetchall()]

    if 'session_id' not in columns:
        c.execute('ALTER TABLE interviews ADD COLUMN session_id TEXT')
        conn.commit()

    conn.close()


def generate_session_id():
    """Generate a unique session ID."""
    return str(uuid.uuid4())[:8]


def insert_interview(data, session_id=None):
    """
    Inserts interview data into the database.
    Expects data matching the InterviewInfo structure.

    Args:
        data: Dictionary with interview fields
        session_id: Optional session ID to group multiple issues from same interview
    """
    # Ensure migration is done
    migrate_db()

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Extract fields safely
    issue_text = data.get('issue_text', '')
    severity_score = data.get('severity_score', 0)
    primary_category = data.get('primary_category', '')
    location_bucket = data.get('location_bucket', '')
    evidence_span = data.get('evidence_span', '')

    # Generate session_id if not provided
    if session_id is None:
        session_id = generate_session_id()

    # Save full data blob as raw_log
    raw_log = json.dumps(data, ensure_ascii=False)

    c.execute('''
        INSERT INTO interviews (
            session_id, timestamp, issue_text, severity_score, primary_category,
            location_bucket, evidence_span, raw_log
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        session_id, timestamp, issue_text, severity_score, primary_category,
        location_bucket, evidence_span, raw_log
    ))

    conn.commit()
    conn.close()

    return session_id


def insert_multiple_issues(issues, session_id=None):
    """
    Insert multiple issues from the same interview session.

    Args:
        issues: List of dictionaries with interview fields
        session_id: Optional session ID (will be generated if not provided)

    Returns:
        session_id used for all issues
    """
    if not issues:
        return None

    if session_id is None:
        session_id = generate_session_id()

    for issue in issues:
        insert_interview(issue, session_id=session_id)

    return session_id


def get_all_interviews():
    """Get all interviews ordered by most recent first."""
    migrate_db()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM interviews ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_interviews_by_session(session_id):
    """Get all issues from a specific interview session."""
    migrate_db()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM interviews WHERE session_id = ? ORDER BY id", (session_id,))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]
