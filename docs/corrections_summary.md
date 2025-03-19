# Corrections Summary

## Purpose
The purpose of these corrections is to unify the functionality of `check_schedule` and `extract_unexecuted_rows`, and update the main program to handle the scheduling and execution of experiments more efficiently.

## Changes Made
1. **Unify `check_schedule` and `extract_unexecuted_rows`**:
   - Combined the functionality of `check_schedule` and `extract_unexecuted_rows` into a single function named `check_schedule` in `schedule_database_module.py`.

2. **Update `main_program.py`**:
   - Modified the main program to:
     - Get the list of unexecuted parameter IDs with `check_schedule`.
     - If the list is empty, refer to the parameter database and add unregistered parameter IDs to the schedule.
     - Run `check_schedule` again if new parameter IDs were added.
     - Use the obtained parameter ID list to reference the parameter database and obtain parameters.
     - Experiment with the obtained parameters.
     - Organize the experiment results.
     - Update the schedule.

These changes ensure that the main program correctly handles the scheduling of parameters and avoids redundancy.
