import sys
from execution_module import execute_program
from parsing_module import parse_output
from database_module import initialize_database, store_results
from execution_plan_management import extract_unexecuted_rows, update_status
from schedule_database_module import initialize_schedule_database, insert_schedule

def validate_parameters(M, N, K):
    if (M * K) % 32 != 0 or (K * N) % 32 != 0 or (M * N) % 32 != 0:
        raise ValueError("M*K, K*N, and M*N must all be multiples of 32")

def main():
    if len(sys.argv) != 8:
        print("Usage: python main_program.py <db_path> <program_path> <M_min> <M_max> <N_min> <N_max> <K_min> <K_max> <iterations>")
        sys.exit(1)
    
    db_path = sys.argv[1]
    program_path = sys.argv[2]
    M_min = int(sys.argv[3])
    M_max = int(sys.argv[4])
    N_min = int(sys.argv[5])
    N_max = int(sys.argv[6])
    K_min = int(sys.argv[7])
    K_max = int(sys.argv[8])
    iterations = int(sys.argv[9])
    
    initialize_database(db_path)
    
    initialize_schedule_database(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM schedule')
    schedule_exists = cursor.fetchone()[0] > 0
    conn.close()

    if not schedule_exists:
        insert_schedule(db_path, M_min, M_max, N_min, N_max, K_min, K_max)

    unexecuted_rows = extract_unexecuted_rows(db_path)
    for row in unexecuted_rows:
        exec_id, M, N, K = row
        try:
            validate_parameters(M, N, K)
            update_status(db_path, exec_id, "in_progress")
            output = execute_program(program_path, M, N, K, iterations)
            results = parse_output(output)
            store_results(db_path, M, N, K, iterations, results)
            update_status(db_path, exec_id, "completed")
        except ValueError as e:
            print(f"Skipping invalid parameters M={M}, N={N}, K={K}: {e}")
            update_status(db_path, exec_id, "unexecuted")
        except Exception as e:
            print(f"Error processing M={M}, N={N}, K={K}: {e}")
            update_status(db_path, exec_id, "unexecuted")

if __name__ == "__main__":
    main()
