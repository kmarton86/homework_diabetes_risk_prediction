import sqlite3

DB_FILE = "diabetes.db"

def init_db():
    """Inicializálja az adatbázist és létrehozza a patients táblát."""

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
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
    conn.close()

if __name__ == "__main__":
    init_db()