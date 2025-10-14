import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import GradientBoostingRegressor,BaggingRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_squared_error,root_mean_squared_error,mean_absolute_percentage_error
from helpers.logger import logger
from sklearn.model_selection import GridSearchCV
from helpers.config import load_config

class ModelTrainer:
    def __init__(self,config):
        self.config = config
        
        
    def load_models_params(self):
        """ models and params for GridSearch through MlFlow """
        try:
            self.params = {
                "LinearRegression_params": {
                    "copy_X": [True,False],
                    "fit_intercept":[True,False],
                    "n_jobs": [1000,5000,10000],
                    "positive": [True,False]
                },
                "Ridge_params": {
                    "alpha": [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]
                },
                "Lasso_params": {
                    "alpha": [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]
                },
                "GradientBoostingRegressor_params": {
                    "n_estimators": [50,100,200],
                    "learning_rate": [0.01,0.1,0.2],
                    "max_depth": [3,4,5],
                    "min_samples_split": [2,5,10]
                },
                "BaggingRegressor_params": {
                    "n_estimators": [50,100,200],
                    "max_samples" : [1.0,0.8,0.6],
                    "max_features": [1.0,0.8,0.6]
                },
                "RandomForestRegressor_params": {
                    "n_estimators": [10, 50, 100, 200],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "DecisionTreeRegressor_params": {
                    "max_depth": [None, 5, 10],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
            }
            self.models = {
                "LinearRegression":(LinearRegression(),self.params["LinearRegression_params"]),
                "RandomForestRegressor":(RandomForestRegressor(),self.params["RandomForestRegressor_params"]),
                "GradientBoostingRegressor":(GradientBoostingRegressor(),self.params["GradientBoostingRegressor_params"]),
                "Lasso":(Lasso(),self.params["Lasso_params"]),
                "Ridge":(Ridge(),self.params["Ridge_params"]),
                "BaggingRegressor":(BaggingRegressor(),self.params["BaggingRegressor_params"]),
                "DecisionTreeRegressor":(DecisionTreeRegressor(),self.params["DecisionTreeRegressor_params"])
            }
            return self.params,self.models
        except Exception as e:
            logger.exception(f"Type error: {e}")
        raise
    
    
    def log_into_mlflow(self):
        """ Logs params and models into mlflow """
        try:
            # data
            X_train_scaled = pd.read_csv(self.config["train_processed"],delimiter=",")
            X_test_scaled = pd.read_csv(self.config['test_processed'],delimiter=",")
            y_train = pd.read_csv(self.config["train_target_raw"],delimiter=",")
            y_test = pd.read_csv(self.config["test_target_raw"],delimiter=",")
            mlflow.set_experiment("utilization-pipeline-1")
            for model_name, (model,params) in self.models.items():
                with mlflow.start_run(run_name=model_name):
                    grid_search = GridSearchCV(model,params,cv=4,scoring="neg_mean_squared_error",n_jobs=-1)
                    grid_search.fit(X_train_scaled,y_train.values.ravel())
                    pred = grid_search.predict(X_test_scaled)
                    
                    # mse, r2 score, rmse, mape
                    
                    r2 = r2_score(y_test,pred)
                    mse = mean_squared_error(y_test,pred)
                    rmse = root_mean_squared_error(y_test,pred)
                    mape = mean_absolute_percentage_error(y_test,pred)
                    
                    # logging metrics and params and best score into mlflow
                    
                    
                    mlflow.log_param("best_score", grid_search.best_score_)
                    mlflow.log_params(grid_search.best_params_)
                    mlflow.log_metric("r2_score", r2)
                    mlflow.log_metric("MAPE", mape)
                    mlflow.log_metric("RMSE", rmse)
                    mlflow.log_metric("mean_squared_error", mse)
                    
                    logger.info(f"Model Name: {model_name}; -- R2 Score: {r2:.3f}; mape: {mape:.4f}; RMSE: {rmse}; mse: {mse:.4f}")
            
        except Exception as e:
            raise e