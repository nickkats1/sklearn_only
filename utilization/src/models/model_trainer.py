# DataTransformation
from src.data.data_transformation import DataTransformation

# Config and logger
from helpers.config import load_config
from helpers.logger import logger

# sklearn
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    BaggingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV

from typing import Dict, List, Any, Tuple


class ModelTrainer:
    """Train and evaluate multiple regression models."""
    
    def __init__(self, config: dict, data: DataTransformation | None = None):
        """Initialize ModelTrainer class.
        
        Args:
            config: Configuration dictionary.
            data (DataTransformation): DataTransformation instance.
        """
        self.config = config or load_config()
        self.data = data or DataTransformation(self.config)
        
    def load_models_params(
        self,
    ) -> Tuple[
        Dict[str, Dict[str, List[Any]]],
        Dict[str, Tuple[Any, Dict[str, List[Any]]]],
    ]:
        """Load regression models and their hyperparameter grids.

        Returns:
            Tuple containing:
                - Dictionary of model names to parameter grids.
                - Dictionary of model names to (estimator, parameter grid).
        """
        try:
            params = {
                "LinearRegression": {
                    "copy_X": [True, False],
                    "fit_intercept": [True, False],
                    "positive": [True, False],
                },
                "Ridge": {
                    "alpha": [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100],
                },
                "Lasso": {
                    "alpha": [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]
                },
                "GradientBoostingRegressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 4, 5],
                    "min_samples_split": [2, 5, 10],
                },
                "BaggingRegressor": {
                    "n_estimators": [50, 100, 200],
                    "max_samples": [1.0, 0.8, 0.6],
                    "max_features": [1.0, 0.8, 0.6],
                },
                "RandomForestRegressor": {
                    "n_estimators": [10, 50, 100, 200],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
                "DecisionTreeRegressor": {
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            }

            models = {
                "LinearRegression": (LinearRegression(), params["LinearRegression"]),
                "Ridge": (Ridge(), params["Ridge"]),
                "Lasso": (Lasso(), params["Lasso"]),
                "GradientBoostingRegressor": (
                    GradientBoostingRegressor(),
                    params["GradientBoostingRegressor"],
                ),
                "BaggingRegressor": (BaggingRegressor(), params["BaggingRegressor"]),
                "RandomForestRegressor": (
                    RandomForestRegressor(),
                    params["RandomForestRegressor"],
                ),
                "DecisionTreeRegressor": (
                    DecisionTreeRegressor(),
                    params["DecisionTreeRegressor"],
                ),
            }

            return params, models
        except Exception as exc:
            logger.error(f"Error loading models and parameters: {exc}")
            return None, None

    def train_models(self) -> List[Dict[str, Any]]:
        """Train models using grid search and evaluate performance.

        Returns:
            List of dictionaries containing model evaluation results,
            best estimators, and best hyperparameters.
        """
        results = []

        try:
            # Split and transform features and targets
            X_train_scaled, X_test_scaled = self.data.split_transform_features()
            y_train, y_test = self.data.split_targets()

            # Load models and parameter grids
            params, models = self.load_models_params()

            # Perform grid search for each model
            for model_name, (model, params) in models.items():
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    cv=4,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                )
                grid_search.fit(X_train_scaled, y_train)

                # Predict on test data
                y_pred = grid_search.predict(X_test_scaled)

                # Evaluate model performance
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)

                results.append(
                    {
                        "Model": model_name,
                        "R2 Score": r2,
                        "Mean-Squared Error": mse,
                        "best estimator": grid_search.best_estimator_,
                        "best params": grid_search.best_params_,
                        "best score": grid_search.best_score_,
                    }
                )

            return results
        except Exception as exc:
            logger.info(f"Could not perform grid search: {exc}")
            return results