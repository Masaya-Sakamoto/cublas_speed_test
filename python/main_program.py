import sys
from execution_module import execute_program
from parsing_module import parse_output
from database_module import initialize_database, store_results
from execution_plan_management import extract_unexecuted_rows, update_status
from schedule_database_module import initialize_schedule_database, check_schedule, insert_schedule, list_unregistered_parameters
from parameter_database_module import initialize_parameter_database, insert_parameter, initialize_parameters

def validate_parameters(M, N, K):
    if (M * K) % 32 != 0 or (K * N) % 32 != 0 or (M * N) % 32 != 0:
        raise ValueError("M*K, K*N, and M*N must all be multiples of 32")

def main():
    if len(sys.argv) != 10:
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
    initialize_parameter_database(db_path)
    initialize_parameters(db_path, range(M_min, M_max + 1, 32), range(N_min, N_max + 1, 32), range(K_min, K_max + 1, 32))
    
    schedule_exists = check_schedule(db_path)

    if not schedule_exists:
        unregistered_parameters = list_unregistered_parameters(db_path)
        for parameter_id in unregistered_parameters:
            insert_schedule(db_path, parameter_id)

    unexecuted_rows = extract_unexecuted_rows(db_path)
    for row in unexecuted_rows:
        exec_id, parameter_id = row
        try:
            update_status(db_path, exec_id, "in_progress")
            output = execute_program(program_path, parameter_id, iterations)
            results = parse_output(output)
            store_results(db_path, parameter_id, iterations, results)
            update_status(db_path, exec_id, "completed")
        except ValueError as e:
            print(f"Skipping invalid parameters parameter_id={parameter_id}: {e}")
            update_status(db_path, exec_id, "unexecuted")
        except Exception as e:
            print(f"Error processing parameter_id={parameter_id}: {e}")
            update_status(db_path, exec_id, "unexecuted")

if __name__ == "__main__":
    main()
