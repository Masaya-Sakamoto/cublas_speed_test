# Instructions for Using the Python Data Collection Program

## Purpose
The Python data collection program executes measurement programs, parses their output, and stores the results in an SQLite database.

## Program Structure
The program consists of the following modules:
1. `execution_module.py`: Executes the measurement programs and captures their output.
2. `parsing_module.py`: Parses the output and extracts the relevant data.
3. `database_module.py`: Stores the parsed data in an SQLite database.
4. `main_program.py`: Integrates the modules, handles user input, and coordinates the data collection process.

## Usage
To run the main program, use the following command:
```sh
python main_program.py <db_path> <program_path> <M_min> <M_max> <N_min> <N_max> <K_min> <K_max> <iterations>
```

### Parameters
- `<db_path>`: Path to the SQLite database file where the results will be stored.
- `<program_path>`: Path to the measurement program to be executed (e.g., `../lib/build/cblas_col_speed_test`).
- `<M_min>`: Minimum value for M (number of rows).
- `<M_max>`: Maximum value for M (number of rows).
- `<N_min>`: Minimum value for N (number of columns).
- `<N_max>`: Maximum value for N (number of columns).
- `<K_min>`: Minimum value for K (inner dimension).
- `<K_max>`: Maximum value for K (inner dimension).
- `<iterations>`: Number of iterations to run the measurement program.

### Example
```sh
python main_program.py results.db ../lib/build/cblas_col_speed_test 32 64 32 64 32 64 10
```

This command will execute the `cblas_col_speed_test` program with the specified parameters, parse the output, and store the results in the `results.db` SQLite database.

## Notes
- Ensure that `M*K`, `K*N`, and `M*N` are all multiples of 32.
- The program will skip invalid parameter combinations and print an error message.

## Quality Assurance
- Focus on improving quality through execution testing and validation.
- Ensure frequent commits to the git repository for version control and traceability.
