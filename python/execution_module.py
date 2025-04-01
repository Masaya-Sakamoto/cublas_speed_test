import subprocess

def execute_program(program_path, M, N, K, divisions, iterations):
    try:
        command = [program_path, str(M), str(N), str(K), str(divisions), str(iterations)]
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        if result.stdout.strip() == "" and result.stderr.strip():
            print(f"Error: Program execution failed with error:\n{result.stderr}")
            return None
        
        return result.stdout
    except FileNotFoundError as e:
        print(f"Error: {e}. Program not found at {program_path}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error: Execution failed with return code {e.returncode}.\n{e.stderr}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
