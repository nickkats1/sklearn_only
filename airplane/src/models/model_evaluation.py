# metrics
from sklearn.metrics import r2_score, mean_squared_error

# joblib to save model
import joblib

# logger and config
from helpers.config import load_config
from helpers.logger import logger

# data transformation
from src.data.data_transformation import DataTransformation

from typing import List, Any, Dict

# best model
from sklearn.ensemble import RandomForestRegressor

import pandas as pd

class ModelEvaluation:
    """Class to evaluate metrics of best model from model trainer with best params."""
    
    def __init__(self, config: dict):
        """Initialize ModelEvaluation class.
        
        Args:
            config (dict): Config file consisting of features, targets, file paths.
        """
        self.config = config or load_config()
        self.scores = []
        
    def eval_best_model(self, y_test: List[pd.Series], y_pred: List[pd.Series]) -> List[Dict[str, Any]]:
        """Evaluate metrics from best model from model trainer.
        
        Args:
            y_test List[pd.Series]: the actual value.
            y_pred List[pd.Series]: the predicted value.
        """
        try:
            # load in data
            X_train_scaled, X_test_scaled = DataTransformation(self.config).split_transform_features()
            
            y_train, y_test = DataTransformation(self.config).split_targets()
            

            
            model = RandomForestRegressor(
                max_depth=20,
                min_samples_leaf=2,
                min_samples_split=2,
                n_estimators=200
                )
            
            model = model.fit(X_train_scaled, y_train)
            
            # save model
            joblib.dump(model, "artifacts/best_model.pkl")
            
            y_pred = model.predict(X_test_scaled)
            
            # r2 score
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            self.scores.append({"Mean-Squared Error": mse, "R2 Score": r2})
            
            return self.scores
        except Exception as e:
            return f"could not return dictionary: {e}"
        