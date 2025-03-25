import sqlite3
import yaml

def read_critical_params(program_name):
    with open('critical_params.yml', 'r') as file:
        params = yaml.safe_load(file)
        priority = params.get('priority')
    return priority

def update_priority():
    new_priority = read_critical_params()

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

def insert_schedules(db_path, parameter_id_list, priority_list):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    #insert_list = [(parameter_id, "unexecuted", priority) for parameter_id in parameter_id_list for priority in priority_list]
    insert_list = [(parameter_id, "unexecuted", priority) for parameter_id, priority in zip(parameter_id_list, priority_list)]
    cursor.executemany('INSERT INTO schedule (parameter_id, status, priority) VALUES (?, ?, ?)', insert_list)
    conn.commit()
    conn.close()

def list_unregistered_parameters(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, M, N, K FROM parameters
        WHERE id NOT IN (SELECT parameter_id FROM schedule)
    ''')
    unregistered_parameters = cursor.fetchall()
    conn.close()
    return unregistered_parameters

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

def register_parameters_to_schedule(db_path):
    unregistered_parameters = list_unregistered_parameters(db_path)
    if len(unregistered_parameters) == 0:
        return 0
    try:
        sorted_parameters = sorted(unregistered_parameters, key=lambda x: x[1] * x[2] * x[3])
        insert_schedules(
            db_path,
            [parameter[0] for parameter in sorted_parameters],
            [0 for _ in range(len(sorted_parameters))])
    except sqlite3.Error as e:
        print(f"SQLite error occurred: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        return 1
    return 0
    
