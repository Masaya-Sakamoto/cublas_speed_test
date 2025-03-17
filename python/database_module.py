import sqlite3

def initialize_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            M INTEGER,
            N INTEGER,
            K INTEGER,
            iterations INTEGER,
            execution_time_avg REAL,
            execution_time_err REAL,
            device_copy_time_avg REAL,
            device_copy_time_err REAL,
            host_copy_time_avg REAL,
            host_copy_time_err REAL
        )
    ''')
    conn.commit()
    conn.close()

def store_results(db_path, M, N, K, iterations, results):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO results (
            M, N, K, iterations, execution_time_avg, execution_time_err,
            device_copy_time_avg, device_copy_time_err, host_copy_time_avg, host_copy_time_err
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (M, N, K, iterations, results['execution_time_avg'], results['execution_time_err'],
          results['device_copy_time_avg'], results['device_copy_time_err'],
          results['host_copy_time_avg'], results['host_copy_time_err']))
    conn.commit()
    conn.close()
