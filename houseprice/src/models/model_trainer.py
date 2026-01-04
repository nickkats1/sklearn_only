# data transformation
from src.data.data_transformation import DataTransformation

# logger and config
from helpers.logger import logger
from helpers.config import load_config

# sklearn
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import (
    RandomForestRegressor, 
    GradientBoostingRegressor, 
    BaggingRegressor
)
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error



from typing import Any, Dict, List


class ModelTrainer:
    """Class for training models with their respective parameters for hyper-parameter tuning via GridSearch."""
    
    def __init__(self, config: dict):
        """Initialize Model Trainer module.
        
        Args:
            config (dict): A configuration file containing url_links, paths, features, and targets.
        """
        self.config = config or load_config()
        self.results = []
        
    def load_params_and_models(self) -> Dict[str, Any]:
        """Parameters and models for hyper-parameter tuning.
        
        Returns:
            params (Dict[str, List[Any]]): A dictionary of the model name with the respective parameters.
            models (Dict[str, List[Any]]): models with their parameters,
        """
        try:
            params = {
                "LinearRegression_params": {
                    "copy_X": [True,False], 
                    "fit_intercept": [True,False], 
                    "n_jobs": [1000,5000,10000], 
                    "positive": [True,False]
                },
                "Lasso_params": {
                    "alpha": [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]
                },
                "Ridge_params": {
                    "alpha": [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]
                },
                "GradientBoostingRegressor_params": {
                    "n_estimators": [50,100,200],
                    "learning_rate": [0.01,0.1,0.2],
                    "max_depth": [3,4,5],
                    "min_samples_split": [2,5,10]
                },
                "RandomForestRegressor_params": {
                    "n_estimators": [10, 50, 100, 200],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "DecisionTreeRegressor_params": {
                    "max_depth": [None,10,15],
                    "min_samples_split": [2,5,10],
                    "min_samples_leaf": [1,2,5]
                
                },
                "BaggingRegressor_params": {
                    "n_estimators": [50, 100, 200],
                    "max_samples": [1.0, 0.8, 0.6],
                    "max_features": [1.0, 0.8, 0.6]
                },

                "XGBRegressor_params": {
                    "booster": ['gbtree', 'gblinear', 'dart'],
                    "verbosity": [0, 1, 2, 3]
                },
            }
            models = {
                "LinearRegression": (LinearRegression(), params["LinearRegression_params"]),
                "Lasso": (Lasso(), params["Lasso_params"]),
                "Ridge": (Ridge(), params["Ridge_params"]),
                "GradientBoostingRegressor": (GradientBoostingRegressor(), params["GradientBoostingRegressor_params"]),
                "RandomForestRegressor": (RandomForestRegressor(), params["RandomForestRegressor_params"]),
                "DecisionTreeRegressor": (DecisionTreeRegressor(), params["DecisionTreeRegressor_params"]),
                "BaggingRegressor": (BaggingRegressor(), params["BaggingRegressor_params"]),
                "XGBRegressor": (XGBRegressor(), params["XGBRegressor_params"])
                }
            return params, models
        except Exception as e:
            return f"Could not load models and params: {e}"
        
    
    def train_models(self) -> List[Dict[str, Any]]:
        """Perform GridSearchCV and log models and params and scores.
        
        Returns:
            results (Dict[List[str, Any]]): A dictionary containing results from all models by grisearch
        """
        try:
            # load in features and targets
            
            X_train_scaled, X_test_scaled = DataTransformation(self.config).split_transform_features()
            y_train, y_test = DataTransformation(self.config).split_targets()
            
            logger.info("X_train, X_test, y_train, and y_test were successfully implemented")
            
            # load in models and params
            
            params, models = self.load_params_and_models()
            
            for model_name, (models, params) in models.items():
                # grid search
                grid_search = GridSearchCV(models, params, cv=4, scoring="neg_mean_squared_error", n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
                
                y_pred = grid_search.predict(X_test_scaled)
                
                # r2 score
                r2 = r2_score(y_test, y_pred)
                print(f"Model: {model_name}")
                print(f"R2 Score: {r2*100:.2f}")
                
                # mean-squared error
                mse = mean_squared_error(y_test, y_pred)
                print(f"Model: {model_name}")
                print(f"Mean-Squared Error: {mse:.4f}")
                
                
                
                
                
                # best scores
                best_scores = grid_search.best_score_
                print(f"model: {model_name}")
                print(f"Best Scores: {best_scores}")
                
                # best params
                
                best_params = grid_search.best_params_
                print(f"Model Name: {model_name}")
                print(f"Best Params: {best_params}")
                
                # best estimator
                
                best_estimator = grid_search.best_estimator_
                print(f"Model Name: {model_name}")
                print(f"Best Estimator: {best_estimator}")
                
                
                
                self.results.append({
                    "Model": model_name,
                    "r2 score": r2,
                    "mean-squared error": mse,
                    "best_score": best_scores,
                    "best_params": best_params,
                    "best estimator": best_estimator
                    })
                
            return self.results
                
                
            
        except Exception as e:
            return f"Could Not Train Models: {e}"