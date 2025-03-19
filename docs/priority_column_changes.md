# Priority Column Changes

## Purpose
The purpose of these changes is to add a priority column to the schedule database to manage the execution priority of experiments, similar to the `nice` value of a process in Linux.

## Changes Made
1. **Add Priority Column**:
   - Added a `priority` column to the schedule database.
   - Updated the `initialize_schedule_database` function to include the `priority` column.

2. **Update Functions**:
   - Updated the `insert_schedule` function to handle the `priority` column.
   - Updated the `check_schedule` function to consider the `priority` column when retrieving unexecuted rows.

3. **Update Main Program**:
   - Modified the main program to handle the `priority` column when inserting schedules and checking for unexecuted rows.

These changes ensure that the main program correctly handles the scheduling of parameters with priority and avoids redundancy.
