# Execution Plan Management

## Purpose
To manage the execution plan for experiments, we will enhance the existing SQLite database and Python program. This will allow us to track the execution status of each experiment and update the status as the experiment progresses.

## Database Schema
The SQLite database schema will include the following columns:
- `exec_id`: Unique identifier for each experiment.
- `M`: Number of rows.
- `N`: Number of columns.
- `K`: Inner dimension.
- `status`: Execution status (e.g., "unexecuted", "in_progress", "completed").

## Database Initialization
The `initialize_database` function will be updated to create the new schema with the `exec_id`, `M`, `N`, `K`, and `status` columns.

## Execution Plan Management Functions
We will create the following functions to manage the execution plan:
1. **Extract Unexecuted Rows**:
   - Extract rows with the status "unexecuted" from the database.
2. **Update Status to In Progress**:
   - Update the status to "in_progress" when an experiment starts.
3. **Update Status to Completed**:
   - Update the status to "completed" when an experiment ends.

## Main Program Modifications
The main program will be updated to:
1. Extract rows with the status "unexecuted".
2. Update the status to "in_progress" when an experiment starts.
3. Update the status to "completed" when an experiment ends.

## Implementation Plan
1. **Update Database Schema**:
   - Modify the `initialize_database` function to create the new schema with `exec_id`, `M`, `N`, `K`, and `status`.

2. **Execution Plan Management Functions**:
   - Create a function to extract rows with the status "unexecuted".
   - Create a function to update the status to "in_progress".
   - Create a function to update the status to "completed".

3. **Modify Main Program**:
   - Update the main program to:
     - Extract rows with the status "unexecuted".
     - Update the status to "in_progress" when an experiment starts.
     - Update the status to "completed" when an experiment ends.

## Quality Assurance
- Focus on improving quality through execution testing and validation.
- Ensure frequent commits to the git repository for version control and traceability.
