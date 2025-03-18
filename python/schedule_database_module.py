import sqlite3

def initialize_schedule_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS schedule (
            exec_id INTEGER PRIMARY KEY AUTOINCREMENT,
            M INTEGER,
            N INTEGER,
            K INTEGER,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_schedule(db_path, M_min, M_max, N_min, N_max, K_min, K_max):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for M in range(M_min, M_max + 1, 32):
        for N in range(N_min, N_max + 1, 32):
            for K in range(K_min, K_max + 1, 32):
                cursor.execute('INSERT INTO schedule (M, N, K, status) VALUES (?, ?, ?, ?)', (M, N, K, "unexecuted"))
    conn.commit()
    conn.close()
