import sqlite3
import logging

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

def initialize_settings(db_path, program_settings):
    required_keys = ["M_limit", "N_limit", "K_limit"]

    # 設定のバリデーション
    for key in required_keys:
        if key not in program_settings or not isinstance(program_settings[key], (list, tuple)) or len(program_settings[key]) != 2:
            logging.error(f"Invalid or missing key '{key}' in program settings.")
            raise ValueError(f"Invalid or missing key '{key}' in program settings. Expected a list or tuple of two integers.")

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # テーブル作成
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                "key" TEXT PRIMARY KEY,
                "value" INTEGER
            )
        ''')

        # 設定データの挿入
        cursor.executemany(
            "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value;",
            [
                ("M_MINIMUM", program_settings["M_limit"][0]),
                ("M_MAXIMUM", program_settings["M_limit"][1]),
                ("N_MINIMUM", program_settings["N_limit"][0]),
                ("N_MAXIMUM", program_settings["N_limit"][1]),
                ("K_MINIMUM", program_settings["K_limit"][0]),
                ("K_MAXIMUM", program_settings["K_limit"][1]),
            ]
        )

        conn.commit()
    except sqlite3.DatabaseError as e:
        if conn:
            conn.rollback()
        logging.error(f"Database error while initializing settings: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in initialize_settings: {e}")
        raise
    finally:
        if conn:
            conn.close()

def initialize_parameter_database(db_path):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parameters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                M INTEGER,
                N INTEGER,
                K INTEGER,
                divisions INTEGER,
                UNIQUE(M, N, K, divisions)
            )
        ''')
        conn.commit()
    except sqlite3.DatabaseError as e:
        if conn:
            conn.rollback()
        logging.error(f"Database error while initializing parameter database: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in initialize_parameter_database: {e}")
        raise
    finally:
        if conn:
            conn.close()

def insert_parameter(db_path, M, N, K, divisions):
    conn = None
    cursor = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO parameters (M, N, K, divisions) VALUES (?, ?, ?, ?)', (M, N, K, divisions))
        parameter_id = cursor.lastrowid
        conn.commit()
        return parameter_id
    except sqlite3.DatabaseError as e:
        if conn:
            conn.rollback()
        logging.error(f"Database error while inserting parameter: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in insert_parameter: {e}")
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def initialize_parameters(db_path, m_list, n_list, k_list, divisions_list):
    conn = None
    cursor = None
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Prepare parameters as a list of tuples
        params = [(M, N, K, divisions) for M in m_list for N in n_list for K in k_list for divisions in divisions_list]
        
        # Insert all parameters efficiently, updating if they exist
        cursor.executemany('''
            INSERT INTO parameters (M, N, K, divisions)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(M, N, K) DO UPDATE SET
            M = excluded.M,
            N = excluded.N,
            K = excluded.K,
            divisions = excluded.divisions
        ''', params)
        
        # Commit the transaction
        conn.commit()
        
    except sqlite3.DatabaseError as e:
        if conn:
            conn.rollback()
        logging.error(f"Database error while initializing parameters: {e}")
        raise
        
    except Exception as e:
        logging.error(f"Unexpected error in initialize_parameters: {e}")
        if params:
            logging.error(f"Last attempted parameters: {params[-1]}")
        if conn:
            conn.rollback()
        raise
        
    finally:
        # Close cursor and connection regardless of success or failure
        if cursor:
            cursor.close()
        if conn:
            conn.close()
