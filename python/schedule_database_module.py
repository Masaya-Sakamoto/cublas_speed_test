import sqlite3

def initialize_schedule_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS schedule (
            exec_id INTEGER PRIMARY KEY AUTOINCREMENT,
            parameter_id INTEGER,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def check_schedule(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM schedule')
    schedule_exists = cursor.fetchone()[0] > 0
    conn.close()
    return schedule_exists

def insert_schedule(db_path, parameter_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO schedule (parameter_id, status) VALUES (?, ?)', (parameter_id, "unexecuted"))
    conn.commit()
    conn.close()

def list_unregistered_parameters(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id FROM parameters
        WHERE id NOT IN (SELECT parameter_id FROM schedule)
    ''')
    unregistered_parameters = cursor.fetchall()
    conn.close()
    return [row[0] for row in unregistered_parameters]
