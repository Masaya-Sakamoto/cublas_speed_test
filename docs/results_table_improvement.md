# Results Table Improvement

## Purpose
The purpose of these changes is to improve the `results` table to make it easier to use by registering raw data in a separate table and then aggregating it with SQL and Python scripts.

## Changes Made
1. **Create a Raw Data Storage Table**:
   - Modified the `initialize_database` function in `database_module.py` to create a new table for storing raw data.

2. **Modify the Program to Insert Raw Data**:
   - Updated the main program to:
     - Read the three double values separated by commas from multiple lines of standard output.
     - Insert the three read values and the parameter ID into the raw data storage table.

3. **Aggregate Raw Data**:
   - Created a new function to:
     - Extract items with the same parameter ID from the raw data table.
     - Calculate the average and sample standard deviation of each value using numpy.
     - Insert or update the aggregated values in the `results` table using `ON CONFLICT(parameter_id) DO UPDATE`.

4. **Update Iterations**:
   - Updated the `store_results` function to include the total number of measurements in the `iterations` argument.

5. **Move Parsing Logic**:
   - Moved the parsing logic from `main_program.py` to `raw_data_module.py` for better maintainability.

These changes ensure that the main program correctly handles the registration and aggregation of raw data, making the `results` table easier to use.
