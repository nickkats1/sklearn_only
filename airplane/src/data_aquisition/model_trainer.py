import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso,Ridge,LinearRegression
from sklearn.ensemble import GradientBoostingRegressor,BaggingRegressor
import pandas as pd
from helpers.logger import logger
from sklearn.metrics import mean_squared_error,r2_score
from helpers.config import load_config


class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def load_params(self):
        """ Loads models and hyperparameters for GridSearchCV """
        try:
            self.params = {
                "lr_params": {
                    'copy_X': [True,False], 
                    'fit_intercept': [True,False], 
                    'n_jobs': [1000,5000,10000], 
                    'positive': [True,False]
                },
                "lasso_params": {
                    "alpha": [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]
                },
                "ridge_params": {
                    "alpha": [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]
                },
                "bagging_params": {
                    "n_estimators": [50,100,200],
                    "max_samples" : [1.0,0.8,0.6],
                    "max_features": [1.0,0.8,0.6]

                },
                "gradientboosting_params": {
                    "n_estimators": [50,100,200],
                    "learning_rate": [0.01,0.1,0.2],
                    "max_depth": [3,4,5],
                    "min_samples_split": [2,5,10]
                },
            }


            self.models = {
                "LinearRegression": (LinearRegression(), self.params['lr_params']),
                "GradientBoostingRegressor": (GradientBoostingRegressor(), self.params['gradientboosting_params']),
                "ridge":(Ridge(),self.params['ridge_params']),
                "lasso":(Lasso(),self.params['lasso_params']),
                "BaggingRegressor":(BaggingRegressor(),self.params['bagging_params'])
            }
            return self.params,self.models

        except Exception as e:
            logger.exception(f"Error loading models/params: {e}")
            raise

    def log_into_mlflow(self):
        """ Train models using GridSearchCV and log results to MLflow """
        try:
        
            mlflow.set_experiment("airplane-pipeline-v2")

            # Load datasets
            df_train_scaled = pd.read_csv(self.config['train_processed'],delimiter=",")
            df_test_scaled = pd.read_csv(self.config['test_processed'],delimiter=",")
            y_train = pd.read_csv(self.config['y_train_path'],delimiter=",")
            y_test = pd.read_csv(self.config['y_test_path'],delimiter=",")


            for model_name, (model,params) in self.models.items():
                with mlflow.start_run(run_name=model_name):
                    # grid search params
    
    

                    grid_search = GridSearchCV(model, params, cv=4, scoring="r2", n_jobs=-1)
                    grid_search.fit(df_train_scaled, y_train.values.ravel())

                    pred= grid_search.predict(df_test_scaled)
   
                    mse = mean_squared_error(y_test,pred)
                    r2 = r2_score(y_test,pred)

                    # log metrics mlflow
                    mlflow.log_param("best_score", grid_search.best_score_)
                    mlflow.log_params(grid_search.best_params_)
                    mlflow.log_metric("R2 Score", r2)
                    mlflow.log_metric("mean squared error",mse)
                    mlflow.sklearn.log_model(grid_search.best_estimator_)

                    logger.info(f"Model: {model_name} | R2 Score: {r2*100:.2f} | MSE: {mse:.4f}")

        except Exception as e:
            logger.exception(f"MLflow logging failed: {e}")
            raise