import yaml
from parameter_database_module import initialize_parameter_database, initialize_parameters, initialize_settings

def initialize_from_config():
    with open('run_config.yml', 'r') as file:
        config = yaml.safe_load(file)

    for program in config['programs'].values():
        db_path = program['db_path']
        m_list = program['init']['init_m_list']
        n_list = program['init']['init_n_list']
        k_list = program['init']['init_k_list']

        initialize_settings(db_path, program)
        initialize_parameter_database(db_path)
        initialize_parameters(db_path, m_list, n_list, k_list)

if __name__ == "__main__":
    initialize_from_config()
