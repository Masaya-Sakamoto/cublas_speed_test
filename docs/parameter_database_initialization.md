# Parameter Database Initialization

## Purpose
The purpose of this work is to initialize the parameter database with lists of `M`, `N`, and `K` values. This ensures that each unique set of parameters is stored only once, avoiding redundancy and simplifying the schedule and results databases.

## Changes Made
1. **Parameter Database Module**:
   - Added a new function `initialize_parameters` that receives `m_list`, `n_list`, and `k_list`.
   - The function uses a triple loop to iterate over `M`, `N`, and `K`, assigns an ID to each element, and inserts them into the database.

2. **Main Program Modifications**:
   - Updated the main program to:
     - Initialize the parameter database with the provided lists of `M`, `N`, and `K`.
     - End normally if there are no parameters in the parameter database that are not registered in the schedule.

These changes improve the overall structure and efficiency of the system, making it easier to manage and extend in the future.
