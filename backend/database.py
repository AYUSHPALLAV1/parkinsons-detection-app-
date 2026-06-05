import sqlite3
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path=None):
        # On Render/cloud, use /tmp (writable). Locally use project root.
        if db_path is None:
            env_path = os.environ.get('DB_PATH')
            if env_path:
                db_path = env_path
            else:
                db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'parkinsons_detection.db')

        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for assessments
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                voice_prob REAL,
                eye_prob REAL,
                handwriting_prob REAL,
                final_prob REAL,
                verdict TEXT,
                user_id TEXT
            )
        ''')
        
        # Table for feature logs (optional, for history table)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assessment_id INTEGER,
                module TEXT,
                features_json TEXT,
                FOREIGN KEY (assessment_id) REFERENCES assessments (id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def save_assessment(self, voice_prob, eye_prob, handwriting_prob, final_prob, verdict, user_id='anonymous'):
        """Save a new assessment to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO assessments (voice_prob, eye_prob, handwriting_prob, final_prob, verdict, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (voice_prob, eye_prob, handwriting_prob, final_prob, verdict, user_id))
        
        assessment_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return assessment_id

    def get_history(self, limit=10):
        """Get recent assessments."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM assessments ORDER BY timestamp DESC LIMIT ?', (limit,))
        history = cursor.fetchall()
        
        conn.close()
        return history
