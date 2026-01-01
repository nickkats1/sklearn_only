import joblib

from helpers.config import load_config
from src.data.data_transformation import DataTransformation
import pandas as pd

from typing import Dict, List, Tuple

from helpers.logger import logger

import numpy as np

class Predict:
    """Class to predict features from 'model_trainer.py' loaded from .pkl.
    """
    
    def __init__(self, config: dict):
        """
        Docstring for __init__
        
        :param self: Description
        :param config: Description
        :type config: dict
        :param model: Description
        :type model: joblib
        :return: Description
        :rtype: List[ndarray[_AnyShape, dtype[Any]]]
        """
        self.config = config or load_config()
        self.model = self.config['model_path']
        
    def predict(self, features) -> float:
        """Predicts 'best_model.pkl' through features.
        
        Args:
            features(List[pd.Series]): Input features for the 'best model' to predict target.
        """
        try:
            # model path
            model_path = self.config['model_path']
            model = joblib.load(model_path)
            
            y_pred = model.predict([features])[0]
            return round(y_pred, 2)
        except Exception as e:
            logger.error(f"{e}")
        return None