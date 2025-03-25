import sqlite3
import numpy as np
from raw_data_module import extract_raw_data

def initialize_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parameter_id INTEGER,
            iterations INTEGER,
            execution_time_avg REAL,
            execution_time_err REAL,
            device_copy_time_avg REAL,
            device_copy_time_err REAL,
            host_copy_time_avg REAL,
            host_copy_time_err REAL,
            UNIQUE(parameter_id)
        )
    ''')
    conn.commit()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS raw_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parameter_id INTEGER,
            execution_time REAL,
            host_to_device_time REAL,
            device_to_host_time REAL
        )
    ''')
    conn.commit()
    conn.close()

def store_results(db_path, parameter_id, results):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO results (
            parameter_id, iterations, execution_time_avg, execution_time_err,
            device_copy_time_avg, device_copy_time_err, host_copy_time_avg, host_copy_time_err
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(parameter_id) DO UPDATE SET
            iterations=excluded.iterations,
            execution_time_avg=excluded.execution_time_avg,
            execution_time_err=excluded.execution_time_err,
            device_copy_time_avg=excluded.device_copy_time_avg,
            device_copy_time_err=excluded.device_copy_time_err,
            host_copy_time_avg=excluded.host_copy_time_avg,
            host_copy_time_err=excluded.host_copy_time_err
    ''', (parameter_id, results["num_of_samples"], results['execution_time_avg'], results['execution_time_err'],
          results['device_copy_time_avg'], results['device_copy_time_err'],
          results['host_copy_time_avg'], results['host_copy_time_err']))
    conn.commit()
    conn.close()

def aggregate_results(db_path, parameter_id):
    rows = extract_raw_data(db_path, parameter_id)
    execution_times = np.array([row[0] for row in rows])
    host_to_device_times = np.array([row[1] for row in rows])
    device_to_host_times = np.array([row[2] for row in rows])

    results = {
        'num_of_samples': len(rows),
        'execution_time_avg': np.mean(execution_times),
        'execution_time_err': np.std(execution_times, ddof=1),
        'device_copy_time_avg': np.mean(host_to_device_times),
        'device_copy_time_err': np.std(host_to_device_times, ddof=1),
        'host_copy_time_avg': np.mean(device_to_host_times),
        'host_copy_time_err': np.std(device_to_host_times, ddof=1)
    }

    return results
