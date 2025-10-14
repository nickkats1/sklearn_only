import mlflow
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_squared_error
from src.logger import logger
from src.config import load_config


class ModelTrainer:
    def __init__(self,config):
        self.config = config
        
        
    def select_models_params(self):
        """ Models and params for GridSearch """
        try:
            self.params = {
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
            }
            
            # models with parameters
            self.models = {
                "LinearRegression":(LinearRegression(),self.params["LinearRegression_params"]),
                "Lasso":(Lasso(),self.params["Lasso_params"]),
                "Ridge":(Ridge(),self.params["Ridge_params"]),
                "GradientBoostingRegressor":(GradientBoostingRegressor(),self.params["GradientBoostingRegressor_params"]),
                "RandomForestRegressor":(RandomForestRegressor(),self.params["RandomForestRegressor_params"]),
                "DecisionTreeRegressor":(DecisionTreeRegressor(),self.params["DecisionTreeRegressor_params"])
            }
            
            return self.params,self.models
        except ValueError as e:
            logger.exception(f"Value error: {e}")
        raise
    
    
    def log_into_mlflow(self):
        """ Log Hyper-Parameters into MLFlow"""
        try:
            # load in data for training and testing
            X_train_scaled = pd.read_csv("data/processed/train.csv",delimiter=",")
            X_test_scaled = pd.read_csv("data/processed/test.csv",delimiter=",")
            y_train = pd.read_csv("data/raw/y_train.csv",delimiter=",")
            y_test = pd.read_csv("data/raw/y_test.csv",delimiter=",")
            

            for model_name,(model,params) in self.models.items():
                with mlflow.start_run(run_name=model_name):
                    grid_search = GridSearchCV(model,param_grid=params,cv=4,scoring="neg_mean_squared_error",n_jobs=-1)
                    grid_search.fit(X_train_scaled,y_train)
                    pred = grid_search.predict(X_test_scaled)
                    r2 = r2_score(y_test,pred)
                    mse = mean_squared_error(y_test,pred)
                    
                    # log params, best score and metrics into mlflow
                    
                    
                    mlflow.log_param("best_score",grid_search.best_score_)
                    mlflow.log_params(grid_search.best_params_)
                    mlflow.log_metric("r2 score",r2)
                    mlflow.log_metric("mean squared error",mse)
                    logger.info(f"Model Name: {model_name}; -- R2 Score: {r2:.3f}; mse: {mse:.4f}")
                    
                    
        except Exception as e:
            raise e