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

def initialize_parameters(db_path, m_list, n_list, k_list):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for M in m_list:
        for N in n_list:
            for K in k_list:
                cursor.execute('INSERT INTO parameters (M, N, K) VALUES (?, ?, ?)', (M, N, K))
    conn.commit()
    conn.close()
