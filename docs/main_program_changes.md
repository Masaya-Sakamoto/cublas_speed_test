# Main Program Changes

## Purpose
The purpose of these changes is to update `main_program.py` to read from the configuration YAML file, set the arguments of `main()` to the program name and the number of iterations, match the program name with the configuration file, and retain the configuration contents other than `init`.

## Changes Made
1. **Read Configuration YAML File**:
   - Modified `main_program.py` to read the configuration from `run_config.yml`.

2. **Set Arguments of `main()`**:
   - Changed the arguments of `main()` to accept the program name and the number of iterations.

3. **Match Program Name**:
   - Checked if the program name provided as an argument matches the `name` in the configuration file.
   - If a match is found, used the corresponding configuration for the program.

4. **Retain Configuration Contents**:
   - Stored the configuration contents other than `init` in variables and used them in the program.

These changes ensure that `main_program.py` correctly reads from the configuration YAML file, sets the arguments of `main()` to the program name and the number of iterations, matches the program name with the configuration file, and retains the configuration contents other than `init`.
