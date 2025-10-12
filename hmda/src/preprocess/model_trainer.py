import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from helpers.config import load_config
from helpers.logger import logger


class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def load_params(self):
        """ Loads models and hyperparameters for GridSearchCV """
        try:
            self.params = {
                "lr_params": {
                    "C": [0.001, 0.01, 0.1,1,10],
                    "penalty": ['l1', 'l2'],
                    "solver": ["liblinear", "saga"], 
                    "max_iter": [1000, 5000, 10000]
                },
                'gradient_boosting_params': {
                    'n_estimators': [50,100,200],
                    'learning_rate': [1,0.5,0.25,0.1,0.05,0.01],
                    'max_depth': [3,4,5],
                    'min_samples_split': [2,5,10],
                },
                'svc_params': {
                    'C': [0.1,1,10,100,1000],
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                    'kernel': ['rbf']
                },
                'random_forest_params': {
                    'n_estimators': [50,100,200],
                    'max_depth': [None,10,20],
                    'min_samples_leaf':[1,2,4],
                    'max_features': ['sqrt','log2',None],
                    'criterion':['gini', 'entropy']
                },
                'bagging_classifier_params': {
                    'n_estimators': [50,100,200],
                    'max_samples' : [1.0,0.8,0.6],
                    'max_features': [1.0,0.8,0.6]
                },
                
                'knn_params': {
                    'n_neighbors' : [5,7,9,11,13,15],
                    'weights' : ['uniform','distance'],
                    'metric' : ['minkowski','euclidean','manhattan']
                },
                'xgb_params': {
                    'min_child_weight': [1,5,10],
                    'gamma': [0.5,1,1.5,2,5],
                    'subsample': [0.6,0.8,1.0],
                    'colsample_bytree': [0.6,0.8,1.0],
                    'max_depth': [3,4,5]
        },
            }
    


            self.models = {
                "LogisticRegression": (LogisticRegression(), self.params['lr_params']),
                "RandomForestClassifier": (RandomForestClassifier(), self.params['random_forest_params']),
                "GradientBoostingClassifier": (GradientBoostingClassifier(), self.params['gradient_boosting_params']),
                "SVC": (SVC(probability=True), self.params['svc_params']),
                "BaggingClassifier": (BaggingClassifier(), self.params['bagging_classifier_params']),
                "KnnearestNeighnors":(KNeighborsClassifier(),self.params['knn_params']),
                "xgboostingclassifier":(XGBClassifier(objective="binary:logistic"),self.params['xgb_params'])
}
            return self.params,self.models

        except Exception as e:
            logger.exception(f"Error loading models/params: {e}")
            raise

    def log_into_mlflow(self):
        """ Train models using GridSearchCV and log results to MLflow """
        try:
        
            mlflow.set_experiment("hmda-pipeline")

            # Load datasets
            df_train_scaled = pd.read_csv(self.config['train_scaled_path'],delimiter=",")
            df_test_scaled = pd.read_csv(self.config['test_scaled_path'],delimiter=",")
            y_train = pd.read_csv(self.config['y_train_raw'],delimiter=",")
            y_test = pd.read_csv(self.config['y_test_raw'],delimiter=",")


            for model_name, (model,params) in self.models.items():
                with mlflow.start_run(run_name=model_name):
                    # grid search params
    
    

                    grid_search = GridSearchCV(model, params, cv=4, scoring="roc_auc", n_jobs=-1)
                    grid_search.fit(df_train_scaled, y_train)

                    pred= grid_search.predict(df_test_scaled)
                    pred_prob = grid_search.predict_proba(df_test_scaled)[:,1]

                    acc = accuracy_score(y_test,pred)
                    roc = roc_auc_score(y_test,pred_prob)
    

                    # log metrics mlflow
                    mlflow.log_param("best_score", grid_search.best_score_)
                    mlflow.log_params(grid_search.best_params_)
                    mlflow.log_metric("accuracy score", acc)
                    mlflow.log_metric("roc/auc score",roc)
                    mlflow.sklearn.log_model(grid_search.best_estimator_)

                    logger.info(f"Model: {model_name} | Accuracy: {acc*100:.2f} | ROC/AUC: {roc*100:.2f}")

        except Exception as e:
            logger.exception(f"MLflow logging failed: {e}")
            raise

