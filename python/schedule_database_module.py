import sqlite3

def initialize_schedule_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS schedule (
            exec_id INTEGER PRIMARY KEY AUTOINCREMENT,
            parameter_id INTEGER,
            status TEXT,
            priority INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def check_schedule(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT exec_id, parameter_id FROM schedule WHERE status = "unexecuted" ORDER BY priority DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows

def insert_schedule(db_path, parameter_id, priority=0):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO schedule (parameter_id, status, priority) VALUES (?, ?, ?)', (parameter_id, "unexecuted", priority))
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

def get_parameters_by_id(db_path, parameter_id):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT M, N, K FROM parameters WHERE id = ?', (parameter_id,))
        result = cursor.fetchone()
        
        if result:
            M, N, K = result
            return {'M': M, 'N': N, 'K': K}
        else:
            return None  # ID not found in database
        
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return None
    
    finally:
        if conn:
            conn.close()
