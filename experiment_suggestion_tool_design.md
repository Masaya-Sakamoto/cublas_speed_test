# Design Document: Execution Time Estimator and Experiment Suggestion Program

## 1. Overview

**Objective:**  
Develop a Python-based tool that:
- Reads an SQLite database containing historical execution times (with corresponding parameters, means, and standard deviations).
- Uses Gaussian estimation to predict execution times for given parameter sets.
- Identifies “uncertain points” where prediction uncertainty is high.
- Suggests next parameter sets for experimentation to improve model accuracy.

**Key Concepts:**
- **Gaussian Estimation:** Leverages the assumption that execution time data is normally distributed, using the stored mean and standard deviation.
- **Uncertainty Identification:** Evaluates each parameter set based on variance (or other criteria like sample count) to flag those with insufficient or highly variable data.
- **Iterative Improvement:** New experiments are performed on uncertain points to enrich the database and refine future predictions.

---

## 2. System Architecture

### High-Level Components

1. **Data Access Layer (DAL)**
   - **Responsibilities:**  
     - Connect to and query the SQLite database.
     - Retrieve historical execution data.
     - Update the database with new experiment results.
   - **Tools/Libraries:**  
     - Python’s built-in `sqlite3` module or SQLAlchemy for a higher-level ORM.

2. **Estimation Engine**
   - **Responsibilities:**  
     - Perform Gaussian estimation using the mean and standard deviation from the database.
     - Compute confidence intervals to quantify prediction uncertainty.
   - **Tools/Libraries:**  
     - `numpy` and `scipy.stats` (e.g., `norm` functions) for statistical calculations.

3. **Uncertainty Evaluator**
   - **Responsibilities:**  
     - Analyze the Gaussian model output to determine which parameter sets have high uncertainty.
     - Apply heuristic or threshold criteria (e.g., high standard deviation relative to mean, low sample count).
   - **Output:**  
     - A list of “uncertain points” that warrant further investigation.

4. **Parameter Suggestion Module**
   - **Responsibilities:**  
     - Based on the uncertainty analysis, suggest the next set(s) of parameters to experiment with.
     - Optionally, incorporate strategies from design of experiments (DoE) or Bayesian optimization.
   - **Output:**  
     - A prioritized list or ranked order of parameter sets for new experiments.

5. **Control Flow / Main Application**
   - **Responsibilities:**  
     - Orchestrate the overall process: data retrieval, estimation, uncertainty analysis, and suggestion.
     - Provide logging and error handling.
     - Optionally, present results via a CLI or GUI.

---

## 3. Detailed Module Descriptions

### 3.1 Data Access Layer (DAL)

- **Functions:**
  - `connect_db(db_path: str) -> Connection`: Establish connection to the SQLite database.
  - `fetch_execution_data() -> List[Dict]`: Query the database and return a list/dictionary of records.
  - `update_record(record: Dict)`: Insert or update experiment data after running new experiments.
- **Considerations:**
  - Ensure proper error handling when connecting to the database.
  - Use parameterized queries to avoid SQL injection.

### 3.2 Estimation Engine

- **Functions:**
  - `gaussian_estimate(mean: float, stddev: float, confidence: float = 0.95) -> Tuple[float, float]`: Calculate the confidence interval for a given parameter set.
  - `predict_execution_time(parameters: Dict) -> float`: Use a Gaussian model to predict the expected execution time.
- **Considerations:**
  - Validate that execution time data follows a normal distribution.
  - Consider transformations or alternative methods if the normality assumption is violated.

### 3.3 Uncertainty Evaluator

- **Functions:**
  - `evaluate_uncertainty(record: Dict) -> float`: Compute an uncertainty metric (e.g., relative standard deviation or margin of error).
  - `identify_uncertain_points(data: List[Dict], threshold: float) -> List[Dict]`: Return parameter sets where uncertainty exceeds a given threshold.
- **Considerations:**
  - Experiment with different thresholds or metrics.
  - Incorporate sample count into the uncertainty evaluation (i.e., fewer samples could imply higher uncertainty).

### 3.4 Parameter Suggestion Module

- **Functions:**
  - `suggest_next_parameters(uncertain_points: List[Dict]) -> List[Dict]`: From the list of uncertain points, suggest the next parameter set(s) to run experiments on.
  - Optionally, implement a mechanism for exploring parameter space (e.g., grid search, random search, or Bayesian optimization).
- **Considerations:**
  - Decide whether to suggest a single parameter set or multiple sets.
  - Prioritize points with the highest uncertainty or those that maximize learning potential.

### 3.5 Main Application Flow

- **Flow Steps:**
  1. **Initialization:**  
     - Establish database connection.
     - Load all historical execution data.
  2. **Estimation:**  
     - For each record, compute Gaussian estimation and associated confidence intervals.
  3. **Uncertainty Analysis:**  
     - Evaluate the uncertainty metric for each parameter set.
     - Identify “uncertain points” based on a predefined threshold.
  4. **Parameter Suggestion:**  
     - Use the Parameter Suggestion Module to select the next parameter sets for new experiments.
  5. **Output & Logging:**  
     - Present the results (e.g., via CLI output or logs).
     - Optionally, update the database with new experiment results once they are collected.

---

## 4. Data Flow Diagram

```
[SQLite Database]
       │
       ▼
[Data Access Layer] → [Estimation Engine]
       │                        │
       ▼                        ▼
[Historical Data]      [Gaussian Predictions & Confidence Intervals]
       │                        │
       └─────────────┬──────────┘
                     ▼
         [Uncertainty Evaluator]
                     │
                     ▼
      [Identified Uncertain Points]
                     │
                     ▼
       [Parameter Suggestion Module]
                     │
                     ▼
       [Output: Suggested Parameters for Experimentation]
```

---

## 5. Implementation Details

- **Programming Language:**  
  Python 3.x

- **Libraries and Tools:**
  - `sqlite3` or SQLAlchemy for database interaction.
  - `numpy` and `scipy.stats` for statistical computations.
  - `pandas` (optional) for data manipulation.
  - Logging via Python’s `logging` module.

- **Development Steps:**
  1. **Set up the environment:**  
     - Create a virtual environment and install required libraries.
  2. **Develop the DAL module:**  
     - Write functions for connecting to and querying the SQLite database.
  3. **Implement the Estimation Engine:**  
     - Develop functions to perform Gaussian-based estimation and compute confidence intervals.
  4. **Build the Uncertainty Evaluator:**  
     - Define the metric for uncertainty and code the logic to flag uncertain parameter sets.
  5. **Design the Parameter Suggestion Module:**  
     - Develop heuristics or use optimization techniques to suggest the next experiments.
  6. **Integrate modules into the Main Application:**  
     - Orchestrate data flow, logging, and error handling.
  7. **Testing:**  
     - Write unit tests for each module.
     - Test end-to-end functionality using sample data.

---

## 6. Testing and Validation

- **Unit Testing:**
  - Test each module (DAL, Estimation Engine, Uncertainty Evaluator, Parameter Suggestion) independently.
  - Validate correct database queries, accurate statistical computations, and proper identification of uncertain points.

- **Integration Testing:**
  - Simulate the entire workflow with a sample database to ensure proper data flow.
  - Verify that new parameter suggestions are reasonable based on the input data.

- **Performance Testing:**
  - Check that the program scales when more records are added to the database.
  - Profile performance-critical sections (e.g., database queries, statistical computations).

---

## 7. Future Enhancements

- **Model Refinement:**  
  - Incorporate machine learning models (e.g., regression models, Gaussian Processes) for more advanced predictions.
  
- **User Interface:**  
  - Develop a GUI or web-based dashboard for visualizing predictions, uncertainties, and suggested experiments.

- **Dynamic Experiment Scheduling:**  
  - Integrate with an automated experimental platform to run tests and automatically update the database.

- **Robustness to Distribution Changes:**  
  - Implement tests to verify the normality assumption; if invalid, consider data transformations or alternative statistical methods.

---

## 8. Conclusion

This design document outlines the development of a Python program that estimates execution times using Gaussian assumptions and identifies parameter sets that require further experimental validation. By following this design, the system will iteratively refine its predictions and drive more efficient experimentation, leading to a more accurate understanding of the program’s performance characteristics.
