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

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

def main(program_name, iterations):
    try:
        with open('run_config.yml', 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logging.error("Configuration file 'run_config.yml' not found.")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while reading config file: {e}")
        raise

    try:
        programs = config.get('programs', {})
        program_config = next((p for p in programs.values() if p.get('name') == program_name), None)

        if not program_config:
            logging.error(f"Program {program_name} not found in configuration.")
            raise ValueError(f"Program {program_name} not found in configuration.")

        db_path = program_config.get('db_path')
        program_path = program_config.get('path')

        if not db_path or not program_path:
            logging.error("Missing 'db_path' or 'path' in program configuration.")
            raise ValueError("Missing 'db_path' or 'path' in program configuration.")

        # Initialize database and schedules
        try:
            initialize_from_config()
            initialize_database(db_path)
            initialize_schedule_database(db_path)
        except Exception as e:
            logging.error(f"Database initialization failed: {e}")
            raise

        # Check schedule
        try:
            unexecuted_rows = check_schedule(db_path)

            if not unexecuted_rows:
                register_parameters_to_schedule(db_path)
                unexecuted_rows = check_schedule(db_path)
        except Exception as e:
            logging.error(f"Error checking or registering schedule: {e}")
            raise

        for row in unexecuted_rows:
            exec_id, parameter_id = row
            try:
                update_status(db_path, exec_id, "in_progress")
                param_dict = get_parameters_by_id(db_path, parameter_id)

                if not all(k in param_dict for k in ["M", "N", "K"]):
                    raise ValueError("Missing required parameters M, N, or K")

                output = execute_program(f"{program_path}", param_dict["M"], param_dict["N"], param_dict["K"], iterations)

                if output is None:
                    raise RuntimeError(f"Execution failed for parameter_id={parameter_id}")

                parse_and_insert_raw_data(db_path, parameter_id, output)
                results = aggregate_results(db_path, parameter_id)
                store_results(db_path, parameter_id, results)
                update_status(db_path, exec_id, "completed")
            except ValueError as e:
                logging.warning(f"Skipping invalid parameters parameter_id={parameter_id}: {e}")
                update_status(db_path, exec_id, "unexecuted")
            except Exception as e:
                logging.error(f"Error processing parameter_id={parameter_id}: {e}")
                update_status(db_path, exec_id, "unexecuted")

    except Exception as e:
        logging.critical(f"Unexpected error in main execution: {e}")
        raise

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