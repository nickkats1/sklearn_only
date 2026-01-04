# features
from src.data.feature_engineering import FeatureEngineering

# Pandas and Numpy
import pandas as pd
import numpy as np

# helpers
from helpers.config import load_config
from helpers.logger import logger

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from typing import Any, Dict, Tuple, Optional

class DataTransformation:
    """A utility class to scale training and testing features and split traing and testing features."""
    
    def __init__(self, config: dict, data: FeatureEngineering | None = None):
        """Initialize DataTransformation class.
        
        Args:
            config (dict): A configuration file consisting of files, target, unused features, features ect.
            data (FeatureEngineering): Feature Engineering module with cleaned data and selected features.
        """
        
        self.config = config or load_config()
        self.data = data or FeatureEngineering(self.config)
        self.scaler = StandardScaler()
        
    def split_and_scale_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """selected features scaled using StandardScaler and split 80/20.
        
        Returns:
        X_train_scaled (np.ndarray): A scaled dataframe consisting of 80% of features.
        X_test_scaled (np.ndarray): A scaled dataframe consisting of 20% of features.
        
        Raises:
            ValueError: If Splitting and scaling features fails.
        """
        try:
            data = self.data.select_features()
            if data is None:
                raise ValueError("Could not get Features from feature engineering")
            
            # features from config
            features = self.config['features']    
            
         
            X_train, X_test = train_test_split(data[features], test_size=0.20, random_state=42)
            logger.info("Features have been split")
            # scale X_train and X_test using Standard scaler.
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info(f"Shape of X_train_scaled: {X_train_scaled}")
            logger.info(f"Shape of X_test_scaled: {X_test_scaled.shape}")
            
            return X_train_scaled, X_test_scaled
        except Exception as exc:
            logger.error("Failed to split and scale features: %s", exc)
            raise ValueError(f"Could not split and scale data: {exc}") from exc
        
        
    def split_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Split targets into training and testing sets.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Training and testing target arrays.
            
        Raises:
            ValueError: If Splitting targets fails.
        """
        try:
            data = self.data.select_features()
            if data is None:
                raise ValueError("Could not split targets")
            
            # targets
            targets = self.config['target']
            
            y_train, y_test = train_test_split(data[targets], test_size=0.20, random_state=42)
            
            logger.info("y_train and y_test have been initialized")
            
            logger.info(f"Shape of y_train: {y_train.shape}")
            logger.info(f"Shape of y_test: {y_test.shape}")
            
            return y_train, y_test
        except Exception as exc:
            logger.error("Failed to split targets: %s", exc)
            raise ValueError(f"Could not split targets: {exc}") from exc

