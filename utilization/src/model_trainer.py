import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import numpy as np
import logging
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_squared_error
from utils.config import load_config

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self,config):
        self.config = config
        
    def fetch_data(self):
        """Training and testing processed data"""
        try:
            # training and testing scaled path
            self.df_train_scaled = pd.read_csv(self.config['train_processed'],delimiter=",")
            self.df_test_scaled = pd.read_csv(self.config['test_processed'],delimiter=",")
            #target variable for y
            self.y_train_df = pd.read_csv(self.config['train_target_raw'],delimiter=",")
            self.y_test_df = pd.read_csv(self.config['test_target_raw'],delimiter=",")
            return self.df_train_scaled,self.df_test_scaled,self.y_train_df,self.y_test_df
        except Exception as e:
            logger.exception(f'Could not load data: {e}')
            raise
        
    def load_models(self):
        """models from training and test data"""
        try:
            self.models = {
                "Linear Regression":LinearRegression(),
                "Ridge Regressor":Ridge(),
                "Lasso Regressor": Lasso(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGB Regressor": XGBRegressor(),
                "SVM Regressor":SVR(),
                "Gradient Boosting Regressor":GradientBoostingRegressor()
            }
            return self.models
        except Exception as e:
            logger.exception(f"error loading models")
            raise
        

        
    def log_into_mlflow(self):
        """Log results into mlflow"""
        mlflow.set_experiment("utils pipeline")
        with mlflow.start_run():
            for model_name,model in self.models.items():
                model.fit(self.df_train_scaled,self.y_train_df)
                self.pred = model.predict(self.df_test_scaled)
                self.r2 = r2_score(self.y_test_df,self.pred)
                self.mse = mean_squared_error(self.y_test_df,self.pred)
                print(f'Model Name: {model_name}')
                print(f'R2 Score: {self.r2*100:.2f}')
                print(f'Mean Squared Error: {self.mse:.4f}')
                mlflow.log_metric("train score",model.score(self.df_train_scaled,self.y_train_df))
                mlflow.log_metric("test score",model.score(self.df_test_scaled,self.y_test_df))
                mlflow.log_metric("r2 score",self.r2)
                mlflow.log_metric("mean squared error score",self.mse)
                mlflow.sklearn.log_model("model name",model_name)
                
                

    
                
                
            
if __name__ == "__main__":
    config = load_config()
    model_trainer_config = ModelTrainer(config)
    model_trainer_config.fetch_data()
    model_trainer_config.load_models()
    model_trainer_config.log_into_mlflow()
                        