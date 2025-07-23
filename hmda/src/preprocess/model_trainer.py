import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from src.logger import logger
from sklearn.linear_model import LogisticRegression
from src.config import load_config
from sklearn.preprocessing import StandardScaler

class ModelTrainer:
    def __init__(self,config):
        self.config = config

    def log_into_mlflow(self):
        data = pd.read_csv(self.config['used_raw_path'],delimiter=",")
        used_features = self.config['all_variables']
        data = data[used_features]
        data.drop_duplicates(inplace=True)
        

        features = data.drop(columns=[self.config['target']],axis=1) 
        target = data[self.config['target']]

   
        X_train,X_test,y_train,y_test = train_test_split(features, target, test_size=self.config['test_size'],random_state=self.config['random_state'])
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(**self.config['model_params']) 
        model.fit(X_train_scaled,y_train)

        mlflow.set_experiment("hmda pipeline")
        with mlflow.start_run():
            mlflow.log_params(self.config['model_params'])
            mlflow.log_metric("train_score", model.score(X_train_scaled, y_train)) 
            mlflow.log_metric("test_score", model.score(X_test_scaled, y_test)) 
            mlflow.sklearn.log_model(model, "logistic regression model") 

    
    
if __name__ == "__main__":
    config = load_config()
    model_trainer = ModelTrainer(config)
    model_trainer.log_into_mlflow()