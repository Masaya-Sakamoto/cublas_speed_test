import sqlite3

def initialize_parameter_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            M INTEGER,
            N INTEGER,
            K INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def insert_parameter(db_path, M, N, K):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO parameters (M, N, K) VALUES (?, ?, ?)', (M, N, K))
    parameter_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return parameter_id
