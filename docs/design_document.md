# Design Document for Data Collection Program

## Purpose
The purpose of this document is to outline the structure and functionality of the data collection program. The program will execute measurement programs, parse their output, and store the results in an SQLite database. It will also handle input parameters for matrix sizes and iterations, ensuring that `M*K`, `K*N`, and `M*N` are multiples of 32.

## Program Structure
The program will be created under the `python/` directory and will include the following modules:

1. **Execution Module**:
   - Executes the measurement programs (`cblas_col_speed_test`, `cblas_row_speed_test`, and `cublas_speed_test`) with the specified parameters.
   - Captures the output of the measurement programs.

2. **Parsing Module**:
   - Parses the output of the measurement programs.
   - Extracts the relevant data, including execution time, device copy time, and host copy time.

3. **Database Module**:
   - Stores the parsed data in an SQLite database.
   - Ensures data integrity and efficient storage.

4. **Main Program**:
   - Integrates the modules into a main program that handles user input and coordinates the data collection process.
   - Validates input parameters to ensure `M*K`, `K*N`, and `M*N` are multiples of 32.
   - Uses experimental design methods based on previous results to set M, N, and K for meaningful validation.

## Implementation Plan
The implementation will be carried out in the following stages:

1. **Execution Module**:
   - Develop functions to execute the measurement programs and capture their output.

2. **Parsing Module**:
   - Develop functions to parse the output and extract the relevant data.

3. **Database Module**:
   - Develop functions to store the parsed data in an SQLite database.

4. **Main Program**:
   - Integrate the modules into a main program.
   - Implement input validation and experimental design methods.

## Quality Assurance
- Focus on improving quality through execution testing and validation.
- Ensure frequent commits to the git repository for version control and traceability.

## Questions
- Are there any specific requirements for the format or structure of the SQLite database?
- Should the program include any additional features, such as data visualization or reporting?
