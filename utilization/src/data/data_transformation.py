# feature engineering
from src.data.feature_engineering import FeatureEngineering

# Config logger
from helpers.config import load_config
from helpers.logger import logger

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Pandas numpy
import pandas as pd
import numpy as np

# type-hinting
from typing import Tuple


class DataTransformation:
    """Split Features and targets and scale features using StandardScaler."""
    
    def __init__(self, config: dict, data: FeatureEngineering | None = None):
        """Initialize DataTransformation.
        
        Args:
            config (dict): A configuration file consisting of features, targets, folder paths and url links.
            data (FeatureEngineering): FeatureEngineering instance.
        """
        self.config = config or load_config()
        self.data = data or FeatureEngineering(self.config).select_features()
        self.scaler = StandardScaler()
        
    def split_transform_features(self) -> tuple[np.ndarray, np.ndarray]:
        """Split and scale feature data.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]:
            - Scaled training set.
            - Scaled testing set.
        """
        try:
            # retrieve features from config.
            
            features = self.config['features']
            
            X_train, X_test = train_test_split(self.data[features], test_size=0.20, random_state=42)
            
            # scale features
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info(f"X_train and X_test have been scaled: {X_train_scaled.shape}")
            logger.info(f"X_test_scaled shape: {X_train_scaled.shape}")
            
            return X_train_scaled, X_test_scaled
        except Exception as e:
            logger.error("Could not split and scale features: %s",e)
            return ([]), ([])
        
    def split_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Split target data into training and testing sets.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - training targets
                - testing targets
        """
        try:
            target = self.config['target']
            
            y_train, y_test = train_test_split(self.data[target], test_size=0.20, random_state=42)
            
            logger.info("y_train and y_test have been loaded")
            logger.info(f"y_train Shape: {y_train.shape}")
            logger.info(f"y_test shape: {y_test.shape}")
            
            return y_train, y_test
        except Exception as exc:
            logger.info("Failed to split targets: %s", exc)
            return ([]), ([])
