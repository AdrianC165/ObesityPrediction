# ObesityPrediction
Exploring the Link Between Eating Habits and Obesity

## Project Overview
In this project, we’re digging into the relationship between eating habits and obesity using machine learning. The dataset comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). The main goal is to clean up the data, build a regression model, and see how well we can predict obesity levels based on various lifestyle factors.

## Dataset
We're using the "ObesityDataSet_raw_and_data_sinthetic.csv" from the UCI ML Repository. You can grab it from the link below and place it in the same folder as the code before running the project:
- [Download the Dataset](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)

## Required Libraries
You’ll need the following Python libraries to get everything running:
- `pandas` (for data handling)
- `numpy` (for numerical operations)
- `scikit-learn` (for the machine learning model)
- `matplotlib` (for plotting)

You can install everything in one go with:
```bash
pip install pandas numpy scikit-learn matplotlib
```
## Running the Code in VS Code

1. **Download the dataset** from the provided link and place it in the same directory as the main.py.

2. **Open VS Code** and navigate to the project directory.

3. **Set up a Python environment** (e.g., using a virtual environment) if you haven't already:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

4. **Install the required libraries** by running:
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```

5. **Open the main.py file in VS Code.

6. **Run the Python file** using the built-in terminal. This will:
    - Load and clean the dataset.
    - Convert categorical variables into numerical format.
    - Standardize the features.
    - Split the data into training and test sets.
    - Train a regression model using gradient descent.
    - Evaluate the model and generate plots.

7. **Check the output files** in the project directory:
    - `model_results.txt`: Contains evaluation metrics such as Mean Squared Error, model coefficients, and R2 Score.
    - `predicted_vs_actual.png`: A plot showing the predicted vs. actual values.
    - `tuning_log.txt`: A log file that records the different hyperparameter trials during the grid search, along with corresponding MSE and R^2 scores for each trial. 


## Running the Code in Terminal
Follow the same steps as in VS Code. You can run main.py directly in the terminal with:
```bash
    python main.py
```
    
## Notes
- Ensure that no hardcoded paths are used; the dataset should be in the same directory as main.py.
- The dataset file is not included in this repository but must be downloaded separately from the provided link.
