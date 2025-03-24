import sqlite3

def initialize_settings(db_path, program_settings):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value INTEGER
        )
    ''')
    cursor.execute("INSERT INTO settings (key, value) VALUES (?, ?);", ("M_MINIMUM", program_settings["M_limit"][0]))
    cursor.execute("INSERT INTO settings (key, value) VALUES (?, ?);", ("M_MAXIMUM", program_settings["M_limit"][1]))
    cursor.execute("INSERT INTO settings (key, value) VALUES (?, ?);", ("N_MINIMUM", program_settings["N_limit"][0]))
    cursor.execute("INSERT INTO settings (key, value) VALUES (?, ?);", ("N_MAXIMUM", program_settings["N_limit"][1]))
    cursor.execute("INSERT INTO settings (key, value) VALUES (?, ?);", ("K_MINIMUM", program_settings["K_limit"][0]))
    cursor.execute("INSERT INTO settings (key, value) VALUES (?, ?);", ("K_MAXIMUM", program_settings["K_limit"][1]))
    conn.commit()
    conn.close()

def initialize_parameter_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            M INTEGER,
            N INTEGER,
            K INTEGER,
            UNIQUE(M, N, K)
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
    conn = None
    cursor = None
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Prepare parameters as a list of tuples
        params = [(M, N, K) for M in m_list for N in n_list for K in k_list]
        
        
        # Insert all parameters efficiently, updating if they exist
        cursor.executemany('''
            INSERT INTO parameters (M, N, K)
            VALUES (?, ?, ?)
            ON CONFLICT(M, N, K) DO UPDATE SET
            M = excluded.M,
            N = excluded.N,
            K = excluded.K
        ''', params)
        
        # Commit the transaction
        conn.commit()
        
    except sqlite3.Error as e:
        print(f"An error occurred while connecting to SQLite: {e}")
        if conn is not None:
            # Roll back any changes if something went wrong
            conn.rollback()
        raise
        
    except Exception as e:
        print(f"An error occurred while executing SQL statement: {e}")
        print(f"Last attempted parameters: {params[-1] if params else 'No parameters'}")
        if conn is not None:
            # Roll back any changes if something went wrong
            conn.rollback()
        raise
        
    finally:
        # Close cursor and connection regardless of success or failure
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()
