import sqlite3

def extract_unexecuted_rows(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT exec_id, parameter_id FROM schedule WHERE status = "unexecuted"')
    rows = cursor.fetchall()
    conn.close()
    return rows

def update_status(db_path, exec_id, status):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('UPDATE results SET status = ? WHERE exec_id = ?', (status, exec_id))
    conn.commit()
    conn.close()
