import subprocess

def execute_program(program_path, M, N, K, iterations):
    try:
        command = [program_path, str(M), str(N), str(K), str(iterations)]
        result = subprocess.run(command, capture_output=True, text=True)
        return result.stdout
    except FileNotFoundError as e:
        print(f"Error: {e}. Program not found at {program_path}")
        return None
    except Exception as e:
        print(f"An error occurred while executing the program: {e}")
        return None
