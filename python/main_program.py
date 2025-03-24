import sys
import yaml
import logging
from execution_module import execute_program
from database_module import initialize_database, store_results, aggregate_results
from execution_plan_management import update_status
from schedule_database_module import check_schedule, register_parameters_to_schedule, get_parameters_by_id, initialize_schedule_database
from initialize_parameters import initialize_from_config
from raw_data_module import parse_and_insert_raw_data
from parameter_database_module import initialize_parameter_database, initialize_parameters

def validate_parameters(M, N, K):
    if (M * K) % 32 != 0 or (K * N) % 32 != 0 or (M * N) % 32 != 0:
        raise ValueError("M*K, K*N, and M*N must all be multiples of 32")

def main(program_name, iterations):
    with open('run_config.yml', 'r') as file:
        config = yaml.safe_load(file)

    program_config = None
    for program in config['programs'].values():
        if program['name'] == program_name:
            program_config = program
            break

    if not program_config:
        print(f"Program {program_name} not found in configuration.")
        sys.exit(1)
    
    db_path = program_config['db_path']
    program_path = program_config['path']
    
    initialize_from_config(db_path)
    initialize_database(db_path)
    initialize_schedule_database(db_path)
    
    unexecuted_rows = check_schedule(db_path)

    if not unexecuted_rows:
        register_parameters_to_schedule(db_path)
        unexecuted_rows = check_schedule(db_path)

    for row in unexecuted_rows:
        exec_id, parameter_id = row
        try:
            update_status(db_path, exec_id, "in_progress")
            param_dict = get_parameters_by_id(db_path, parameter_id)
            output = execute_program(program_path, param_dict["M"], param_dict["N"], param_dict["K"], iterations)
            parse_and_insert_raw_data(db_path, parameter_id, output)
            results = aggregate_results(db_path, parameter_id)
            store_results(db_path, parameter_id, results)
            update_status(db_path, exec_id, "completed")
        except ValueError as e:
            print(f"Skipping invalid parameters parameter_id={parameter_id}: {e}")
            update_status(db_path, exec_id, "unexecuted")
        except Exception as e:
            print(f"Error processing parameter_id={parameter_id}: {e}")
            update_status(db_path, exec_id, "unexecuted")

if __name__ == "__main__":
    try:
        with open('run_config.yml', 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logging.error("Configuration file 'run_config.yml' not found.")
        exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        exit(1)
    except Exception as e:
        logging.error(f"Unexpected error while reading config file: {e}")
        exit(1)

    # Read global parameters
    try:
        global_configs = config.get('global', {})
        global_iterations = global_configs.get('iterations', 1)  # Default to 1 if not specified
    except Exception as e:
        logging.error(f"Error retrieving global configurations: {e}")
        exit(1)

    # Read and process programs
    try:
        programs = config.get('programs', {})
        if not programs:
            logging.warning("No programs found in configuration.")
        
        for program in programs.values():
            try:
                program_name = program.get('name', 'UnnamedProgram')
                iterations = program.get('iterations', global_iterations)
                main(program_name=program_name, iterations=iterations)
            except Exception as e:
                logging.error(f"Error processing program '{program_name}': {e}")
    except Exception as e:
        logging.error(f"Error reading 'programs' section: {e}")
        exit(1)

    print("Done")