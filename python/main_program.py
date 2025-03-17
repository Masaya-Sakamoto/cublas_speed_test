import sys
from execution_module import execute_program
from parsing_module import parse_output
from database_module import initialize_database, store_results

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
    
    for M in range(M_min, M_max + 1, 32):
        for N in range(N_min, N_max + 1, 32):
            for K in range(K_min, K_max + 1, 32):
                try:
                    validate_parameters(M, N, K)
                    output = execute_program(program_path, M, N, K, iterations)
                    results = parse_output(output)
                    store_results(db_path, M, N, K, iterations, results)
                except ValueError as e:
                    print(f"Skipping invalid parameters M={M}, N={N}, K={K}: {e}")
                except Exception as e:
                    print(f"Error processing M={M}, N={N}, K={K}: {e}")

if __name__ == "__main__":
    main()
