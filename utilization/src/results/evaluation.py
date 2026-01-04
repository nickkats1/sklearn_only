# metrics
from sklearn.metrics import r2_score, mean_squared_error

# joblib
import joblib

# Config and logger
from helpers.config import load_config
from helpers.logger import logger

# data transformation
from src.data.data_transformation import DataTransformation

# type-hinting
from typing import Dict, List

# pandas
import pandas as pd

# best performing model
from sklearn.ensemble import GradientBoostingRegressor


class Evaluation:
    """Class to evaluate metrics from best-performing model from model trainer."""
    
    def __init__(self, config: dict, data: DataTransformation = None):
        """Initialize Evaluation class.
        
        Args:
            config (dict): A configuration file.
            data (DataTransformation): Instance of DataIngestion.
        """
        
        self.config = config or load_config()
        self.data = data or DataTransformation(self.config)
        self.scores = []
        
    def evaluate_best_model(self, y_test: List[pd.Series], y_pred: List[pd.Series]) -> Dict[str, float]:
        """Evaluate metrics from best-scoring model from ModelTrainer.
        
        Args:
            y_test(List[pd.Series]): Actual values.
            y_pred(List[pd.Series]): Predicted values.
        """
        try:
            X_train_scaled, X_test_scaled = self.data.split_transform_features()
            y_train, y_test = self.data.split_targets()
            
            if self.data is None:
                raise ValueError("Could not get data from data transformation")
            
            model = GradientBoostingRegressor(
                max_depth=4,
                min_samples_split=5,
                n_estimators=200
                )
            
            model = model.fit(X_train_scaled, y_train)
            
            joblib.dump(model, self.config['model_path'])
            
            y_pred = model.predict(X_test_scaled)
            
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            logger.info(f"R2 Score | {r2:.2f} | Mean Squared Error | {mse:.4f}")
            
            self.scores.append({"R2 Score": r2, "MSE": mse})
        except Exception as e:
            logger.error(f"Could Not Get Scores: {e}")
            return None