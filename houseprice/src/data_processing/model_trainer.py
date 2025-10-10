import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score,mean_squared_error
from src.config import load_config
from src.logger import logger


class ModelTrainer:
    def __init__(self,config):
        self.config = config
        
    def fetch_data(self):
        """Training and testing processed data"""
        try:
            # training and testing scaled path
            df_train_scaled = pd.read_csv(self.config['processed_train'],delimiter=",")
            df_test_scaled = pd.read_csv(self.config['processed_test'],delimiter=",")
            #target variable for y
            y_train_df = pd.read_csv(self.config['train_target_raw'],delimiter=",")
            y_test_df = pd.read_csv(self.config['test_target_raw'],delimiter=",")
            return df_train_scaled,df_test_scaled,y_train_df,y_test_df
        except Exception as e:
            logger.exception(f'Could not load data: {e}')
            raise
        
    def load_models(self):
        """models from training and test data"""
        try:
            models = {
                "Linear Regression":LinearRegression(),
                "Ridge Regressor":Ridge(),
                "Lasso Regressor": Lasso(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGB Regressor": XGBRegressor(),
                "SVM Regressor":SVR(),
                "Gradient Boosting Regressor":GradientBoostingRegressor()
            }
            return models
        except Exception as e:
            logger.exception(f"error loading models")
            raise
        

        
    def log_into_mlflow(self):
        """Log results into mlflow"""
        mlflow.set_experiment("utils pipeline")
        models = self.load_models()
        
        # load in data
        df_train_scaled = pd.read_csv(self.config['processed_train'],delimiter=",")
        df_test_scaled = pd.read_csv(self.config['processed_test'],delimiter=",")
        y_train_df = pd.read_csv(self.config['train_target_raw'],delimiter=",")
        y_test_df = pd.read_csv(self.config['test_target_raw'],delimiter=",")
        with mlflow.start_run():
            for model_name,model in models.items():
                model.fit(df_train_scaled,y_train_df)
                pred = model.predict(df_test_scaled)
                r2 = r2_score(y_test_df,pred)
                mse = mean_squared_error(y_test_df,pred)
                print(f'Model Name: {model_name}')
                print(f'R2 Score: {r2*100:.2f}')
                print(f'Mean Squared Error: {mse:.4f}')
                mlflow.log_metric("train score",model.score(df_train_scaled,y_train_df))
                mlflow.log_metric("test score",model.score(df_test_scaled,y_test_df))
                mlflow.log_metric("r2 score",r2)
                mlflow.log_metric("mean squared error score",mse)
                mlflow.sklearn.log_model("model name",model_name)
                
                

    
                
                
            
if __name__ == "__main__":
    config = load_config()
    model_trainer_config = ModelTrainer(config)
    model_trainer_config.fetch_data()
    model_trainer_config.load_models()
    model_trainer_config.log_into_mlflow()