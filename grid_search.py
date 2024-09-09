from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import numpy as np

def perform_grid_search(X_train, y_train):
    # Define the model
    model = SGDRegressor(random_state=42)

    # Hyperparameter grid to search
    param_grid = {
        'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'eta0': [0.0001, 0.001, 0.01],
        'max_iter': [100, 500, 1000],
        'tol': [1e-5, 1e-4, 1e-3],
        'alpha': [0.0001, 0.001, 0.01]
    }

    # defining multiple scorers: MSE (for optimization) and R^2 (For monitoring)
    scorers = {
        'mse': make_scorer(mean_squared_error, greater_is_better=False),
        'r2': make_scorer(r2_score)
    }
        

    # Grid Search with Cross-Validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorers,
        refit='mse',
        cv=5,  # 5-fold cross-validation
        verbose=1
    )

    # Fit the search to the training data
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    cv_results = grid_search.cv_results_

    # Save/log the results
    with open('tuning_log.txt', 'w') as log_file:
        log_file.write("Learning_Rate, Max_Iterations, Tolerance, Alpha, MSE, R2\n")
        for params, mean_mse, mean_r2 in zip(
            grid_search.cv_results_['params'], 
            grid_search.cv_results_['mean_test_mse'], 
            grid_search.cv_results_['mean_test_r2']):
            log_file.write(f"{params['learning_rate']}, {params['max_iter']}, {params['tol']}, {params['alpha']}, {-mean_mse}, {mean_r2}\n")
        

    return best_model, best_params, cv_results
