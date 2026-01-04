# Config
from helpers.config import load_config

# Typing
from typing import List
import pandas as pd

# Joblib for loading saved models
import joblib


class Predict:
    """
    Utility class to perform predictions using a pre-trained model.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the Predict class.

        Args:
            config (dict): Configuration dictionary containing the path to the trained model.
        """
        self.config = config or load_config()

    def predict_pipeline(self, features: List[pd.Series]) -> float:
        """
        Perform prediction using the trained model.

        Args:
            features (pd.DataFrame): Input features for making the prediction.

        Returns:
            float: Predicted value rounded to 2 decimal places.
        """
        try:
            model_path = self.config['model_path']
            model = joblib.load(model_path)
            prediction = model.predict([features])[0]
            return round(prediction, 2)

        except Exception as e:
            return f"did not enter enough values: {str(e)}"


