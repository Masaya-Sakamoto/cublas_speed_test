from parameter_database_module import initialize_parameter_database, initialize_parameters

def main():
    db_path = 'parameters.db'
    m_list = [32, 64, 96]
    n_list = [32, 64, 96]
    k_list = [32, 64, 96]

    initialize_parameter_database(db_path)
    initialize_parameters(db_path, m_list, n_list, k_list)

if __name__ == "__main__":
    main()
