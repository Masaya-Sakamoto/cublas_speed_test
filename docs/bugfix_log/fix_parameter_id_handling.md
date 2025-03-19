# Bug Fix Summary

## Purpose
The purpose of this bug fix is to ensure that the `parameter_id` is obtained from the parameter database and not from the `range` function. This change ensures that only parameters that are not registered in the schedule database are added to the schedule.

## Changes Made
1. **New Program File**:
   - Created `initialize_parameters.py` to initialize the parameter database before running the main program.

2. **Parameter Database Module**:
   - Added the `insert_parameter` function to enable parameters to be registered in the parameter database.

3. **Schedule Database Module**:
   - Updated the `insert_schedule` function to use `parameter_id`.
   - Added the `list_unregistered_parameters` function to list the differences between the parameter database and the schedule database.

4. **Main Program Modifications**:
   - Updated the main program to:
     - Extract `parameter_id` from the parameter database that does not exist in the schedule database.
     - Register the extracted `parameter_id` in the schedule database and set the `status`.
     - Check the schedule database again and extract a new `schedule_exists`.

These changes ensure that the main program correctly handles the scheduling of parameters and avoids redundancy.
