# Schedule Registration Changes

## Purpose
The purpose of these changes is to register the parameters from the parameter database to the schedule database in ascending order of the product `M * N * K` of `M`, `N`, and `K`. This ensures that the execution order is from the smallest to the largest `M * N * K`, allowing us to check the overall trend as quickly as possible.

## Changes Made
1. **Create a Function in `schedule_database_module.py`**:
   - Created a new function `register_parameters_to_schedule` that:
     - Retrieves unregistered parameters from the parameter database.
     - Sorts the parameters by the product `M * N * K`.
     - Inserts the sorted parameters into the schedule database with the appropriate priority.

2. **Update `main_program.py`**:
   - Modified the main program to call `register_parameters_to_schedule` instead of directly configuring the schedule.

These changes ensure that the main program correctly handles the scheduling of parameters in ascending order of `M * N * K` and avoids redundancy.
