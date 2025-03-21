import sqlite3

def insert_raw_data(db_path, parameter_id, execution_time, host_to_device_time, device_to_host_time):
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
