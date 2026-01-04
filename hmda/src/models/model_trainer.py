# data transformation
from src.data.data_transformation import DataTransformation

# models for hyper-parameter tuning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    BaggingClassifier,
    RandomForestClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# metrics
from sklearn.metrics import roc_auc_score, accuracy_score

# GridSearchCV
from sklearn.model_selection import GridSearchCV

# helpers
from helpers.config import load_config
from helpers.logger import logger

# pandas, numpy and typing
from typing import Any, Dict, List


class ModelTrainer:
    """Module to train and perform grid-search for optimal hyper-parameters for best model."""
    
    def __init__(self, config: dict, data: DataTransformation | None = None):
        """Initialize ModelTrainer class.
        
        Args:
            config (dict): Configuration file consisting of features, targets, path, ect.
            data (DataTransformation): module with scaled training/testing features and targets.
        """
        self.config = config or load_config()
        self.data = data or DataTransformation(self.config)
    
        
    def load_models_and_params(self) -> List[Dict[str, Any]]:
        """Load in hyper-parameters from sklearn models and models with the parameters.
        
        Returns:
            params (List[Dict, Any]): model name with parameters for GridSearchCV.
            models (List[Dict, Any]): models with hyper parameters.
        """
        try:
            # load in models with parameters
            
            params = {
                "LogisticRegression_params": {
                    "C": [0.001, 0.01, 0.1,1,10],
                    "solver": ["liblinear", "saga"], 
                    "max_iter": [1000, 5000, 10000]
                },
                'GradientBoostingClassifier_params': {
                    'n_estimators': [50,100,200],
                    'learning_rate': [1,0.5,0.25,0.1,0.05,0.01],
                    'max_depth': [3,4,5],
                    'min_samples_split': [2,5,10],
                },
                'SVC_params': {
                    'C': [0.1,1,10,100,1000],
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                    'kernel': ['rbf']
                },
                'RandomForestClassifier_params': {
                    'n_estimators': [50,100,200],
                    'max_depth': [None,10,20],
                    'min_samples_leaf':[1,2,4],
                    'max_features': ['sqrt','log2',None],
                    'criterion':['gini', 'entropy']
                },
                'BaggingClassifier_params': {
                    'n_estimators': [50,100,200],
                    'max_samples' : [1.0,0.8,0.6],
                    'max_features': [1.0,0.8,0.6]
                },
                
                'KNeighborsClassifier_params': {
                    'n_neighbors' : [5,7,9,11,13,15],
                    'weights' : ['uniform','distance'],
                    'metric' : ['minkowski','euclidean','manhattan']
                },
                'XGBClassifier_params': {
                    'min_child_weight': [1,5,10],
                    'gamma': [0.5,1,1.5,2,5],
                    'subsample': [0.6,0.8,1.0],
                    'colsample_bytree': [0.6,0.8,1.0],
                    'max_depth': [3,4,5]
        },
            }
    


            models = {
                "LogisticRegression": (LogisticRegression(), params['LogisticRegression_params']),
                "RandomForestClassifier": (RandomForestClassifier(), params['RandomForestClassifier_params']),
                "GradientBoostingClassifier": (GradientBoostingClassifier(), params['GradientBoostingClassifier_params']),
                "SVC": (SVC(probability=True), params['SVC_params']),
                "BaggingClassifier": (BaggingClassifier(), params['BaggingClassifier_params']),
                "KNeighborsClassifier":(KNeighborsClassifier(),params['KNeighborsClassifier_params']),
                "XGBClassifier":(XGBClassifier(objective="binary:logistic"),params['XGBClassifier_params'])
}
            return params,models
        except Exception as e:
            logger.error(f"Error loading in models and parameters: {e}")
            return None, None
    
    def get_best_model(self) -> None:
        """Perform GridSearch on models"""
        try:
            # load in X_train_scaled, X_test_scaled, y_train, y_test
            
            X_train_scaled, X_test_scaled = self.data.split_and_scale_features()
            y_train, y_test = self.data.split_targets()
            
            # params and models
            params, models = self.load_models_and_params()
            
            
            # GridSearchCV and logging through MlFlow.
            for model_name, (model, param) in models.items():
      
                grid_search = GridSearchCV(model, param, cv=4, scoring="roc_auc", n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
                
                # predictions and predicted probability.
                
                y_pred = grid_search.predict(X_test_scaled)
                y_pred_prob = grid_search.predict_proba(X_test_scaled)[:,1]
                
                # model name
                print(f"Model: {model_name}")
                
                # accuracy
                
                acc = accuracy_score(y_test, y_pred)
                print(f"Accuracy Score: {acc*100:.2f}")
                
                # roc/auc score
                
                roc = roc_auc_score(y_test, y_pred_prob)
                print(f"Roc/ Auc Score: {roc*100:.2f}")
                
                # best score from grid-search
                best_score = grid_search.best_score_
                print(f"Best Score (roc/auc): {best_score*100:.2f}")
                
                # best params from grid_search
                
                best_params = grid_search.best_params_
                print(f"Best Params: {best_params}")
                
                # best estimator
                
                best_estimator = grid_search.best_estimator_
                print(f"Best Estimator: {best_estimator}")


                
                    
        except Exception as e:
            logger.error(f"Could not run models: {e}")
            return None