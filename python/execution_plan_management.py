import sqlite3

def update_status(db_path, exec_id, status):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('UPDATE schedule SET status = ? WHERE exec_id = ?', (status, exec_id))
    conn.commit()
    conn.close()
