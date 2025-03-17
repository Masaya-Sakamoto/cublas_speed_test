import subprocess

def execute_program(program_path, M, N, K, iterations):
    command = [program_path, str(M), str(N), str(K), str(iterations)]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout
