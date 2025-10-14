from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor,BaggingRegressor
import pandas as pd
import mlflow
from helpers.logger import logger
from helpers.config import load_config


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        
    def models_params(self):
        """ Define models and params for Grid Search """
        try:
            self.params = {
                "LinearRegression_params": {
                    "copy_X": [True,False],
                    "fit_intercept": [True,False],
                    "positive":[True,False]
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
                "BaggingRegressor_params": {
                    "n_estimators": [50,100,200],
                    "max_samples" : [1.0,0.8,0.6],
                    "max_features": [1.0,0.8,0.6]
                },
            }
        
            self.models = {
                "LinearRegression":(LinearRegression(),self.params["LinearRegression_params"]),
                "Lasso": (Lasso(),self.params["Lasso_params"]),
                "Ridge": (Ridge(),self.params['Ridge_params']),
                "GradientBoostingRegressor": (GradientBoostingRegressor(),self.params["GradientBoostingRegressor_params"]),
                "BaggingRegressor": (BaggingRegressor(),self.params["BaggingRegressor_params"])
            }
            return self.params,self.models
        except Exception as e:
            raise
        
    
    def log_into_mlflow(self):
        """ log params and model metrics into mlflow """
        try:
            # models
            df_train_scaled = pd.read_csv(self.config['processed_train'],delimiter=",")
            df_test_scaled = pd.read_csv(self.config['processed_test'],delimiter=",")
            y_train = pd.read_csv(self.config['train_target_raw'],delimiter=",")
            y_test = pd.read_csv(self.config['test_target_raw'],delimiter=",")
            mlflow.set_experiment("house-price-pipeline=v4")
            for model_name,(model,params) in self.models.items():
                with mlflow.start_run(run_name=model_name):
                    grid_search = GridSearchCV(model,params,cv=4,scoring="neg_mean_squared_error",n_jobs=-1)
                    grid_search.fit(df_train_scaled,y_train.values.ravel())
                    pred = grid_search.predict(df_test_scaled)
                    r2 = r2_score(y_test,pred)
                    mse = mean_squared_error(y_test,pred)
                    mlflow.log_params(grid_search.best_params_)
                    mlflow.log_metric("best score",grid_search.best_score_)
                    mlflow.log_metric("mean squared error",mse)
                    mlflow.log_metric("R2 Score",r2)
                    mlflow.sklearn.log_model(grid_search.best_estimator_, "best_estimator")
                    logger.info(f"Model Name: {model_name}; R2 Score: {r2*100:.2f}; MSE: {mse:.4f}")
                    
                    
                    
        except Exception as e:
            raise e
