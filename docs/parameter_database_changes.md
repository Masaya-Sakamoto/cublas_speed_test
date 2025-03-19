# Parameter Database Changes

## Reasons for Change
The primary reason for this change is to improve the management and organization of parameters used in the experiments. By creating a separate parameter database, we can:
- Avoid redundancy by storing each unique set of parameters only once.
- Simplify the schedule and results databases by referencing parameters through their IDs.
- Enhance the flexibility and scalability of the system.

## Changes Made
1. **Parameter Database Module**:
   - Created `initialize_parameter_database` to create the parameter database.
   - Created `insert_parameter` to insert parameters into the database.

2. **Schedule Database Module**:
   - Updated the schedule database schema to include `parameter_id` instead of `M`, `N`, and `K`.

3. **Results Database Module**:
   - Updated the results database schema to include `parameter_id` instead of `M`, `N`, and `K`.

4. **Main Program Modifications**:
   - Updated the main program to:
     - Initialize the parameter database.
     - Insert parameters into the parameter database.
     - Link the schedule and results databases to the parameter database using `parameter_id`.

These changes improve the overall structure and efficiency of the system, making it easier to manage and extend in the future.
