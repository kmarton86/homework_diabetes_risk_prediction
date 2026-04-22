import sqlite3
import pandas as pd

import config
from ml.dataset import load_data  # állítsd a pontos path-ra

# DB_PATH = "diabetes.db"

def init_db(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. ALWAYS CREATE TABLE FIRST
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS diabetes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age REAL,
            sex REAL,
            bmi REAL,
            bp REAL,
            s1 REAL,
            s2 REAL,
            s3 REAL,
            s4 REAL,
            s5 REAL,
            s6 REAL,
            target REAL
        )
    """)

    conn.commit()

    # 2. SAFE CHECK (after table guaranteed to exist)
    cursor.execute("SELECT COUNT(*) FROM diabetes")
    count = cursor.fetchone()[0]

    # 3. populate only if empty
    if count == 0:
        print("Loading dataset into DB...")

        X, y = load_data()

        X_values = X.values
        y_values = y.values

        for i in range(len(X_values)):
            row = list(X_values[i]) + [y_values[i]]

            cursor.execute("""
                INSERT INTO diabetes (
                    age, sex, bmi, bp,
                    s1, s2, s3, s4, s5, s6, target
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, row)

        conn.commit()
        print("Dataset stored in DB")

    else:
        print("DB already initialized")

    conn.close()

def load_from_db(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM diabetes", conn)
    conn.close()
    return df

if __name__ == "__main__":
    DB_PATH = config.DB_PATH
    init_db(DB_PATH)