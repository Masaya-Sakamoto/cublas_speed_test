# Program Specifications

## initialize_parameters.py
**Purpose**: Initializes the parameter database with lists of `M`, `N`, and `K` values.
**Inputs**: `m_list`, `n_list`, `k_list`
**Outputs**: None
**Requirements**: The parameter database must be initialized before running the main program.

## main_program.py
**Purpose**: Manages the execution plan, runs experiments, and stores results.
**Inputs**: `db_path`, `program_path`, `M_min`, `M_max`, `N_min`, `N_max`, `K_min`, `K_max`, `iterations`
**Outputs**: None
**Requirements**: The parameter database must be initialized before running the main program.

## execution_module.py
**Purpose**: Executes the measurement programs and captures their output.
**Inputs**: `program_path`, `M`, `N`, `K`, `iterations`
**Outputs**: Program output as a string
**Requirements**: None

## execution_plan_management.py
**Purpose**: Manages the execution plan by extracting unexecuted rows and updating their status.
**Inputs**: `db_path`
**Outputs**: Unexecuted rows, updated status
**Requirements**: None

## schedule_database_module.py
**Purpose**: Manages the schedule database, including initialization, checking, inserting schedules, and listing unregistered parameters.
**Inputs**: `db_path`, `parameter_id`
**Outputs**: None
**Requirements**: None

## parameter_database_module.py
**Purpose**: Manages the parameter database, including initialization, inserting parameters, and initializing parameters.
**Inputs**: `db_path`, `M`, `N`, `K`, `m_list`, `n_list`, `k_list`
**Outputs**: None
**Requirements**: None

## parsing_module.py
**Purpose**: Parses the output of the measurement programs and extracts relevant data.
**Inputs**: Program output as a string
**Outputs**: Parsed data as a dictionary
**Requirements**: None
