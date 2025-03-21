import sqlite3

def insert_raw_data(db_path, parameter_id, execution_time, host_to_device_time, device_to_host_time):
    # TODO: Use a dictionary with parameter_id as the key for the function argument to support bulk inserts
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO raw_data (
            parameter_id, execution_time, host_to_device_time, device_to_host_time
        ) VALUES (?, ?, ?, ?)
    ''', (parameter_id, execution_time, host_to_device_time, device_to_host_time))
    conn.commit()
    conn.close()

def extract_raw_data(db_path, parameter_id):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT execution_time, host_to_device_time, device_to_host_time
        FROM raw_data
        WHERE parameter_id = ?
    ''', (parameter_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows

def parse_and_insert_raw_data(db_path, parameter_id, output):
    for line in output.strip().split('\n'):
        execution_time, host_to_device_time, device_to_host_time = map(float, line.split(','))
        insert_raw_data(db_path, parameter_id, execution_time, host_to_device_time, device_to_host_time)
