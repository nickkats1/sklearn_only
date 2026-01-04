# Feature Engineering
from src.data.feature_engineering import FeatureEngineering

# pandas numppy
import pandas as pd
import numpy as np

# tools
from helpers.config import load_config
from helpers.logger import logger

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Typing
from typing import Tuple


class DataTransformation:
    """Utility class to split features and targets and scale split features"""
    
    def __init__(self, config: dict, data: FeatureEngineering | None = None):
        """Initialize data transformation class.
        
        Args:
            config (dict): A configuration dictionary.
            data (FeatureEngineering): Instance of FeatureEngineering class.
        """
        
        self.config = config or load_config()
        self.data = data or FeatureEngineering(self.config).select_features()
        self.scaler = StandardScaler()
        
    def split_transform_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Split Features and scale features using Standard Scaler.
        
        Returns:
            X_train_scaled, X_test_scaled(Tuple[np.ndarray, np.ndarray]):
            - X_train_scaled: Training features scaled.
            - X_test_scaled: testing features scaled.
        """
        try:
            
            X_train, X_test = train_test_split(
                self.data[self.config['features']],
                test_size=0.20,
                random_state=42
                )
            
            logger.info("X_train and X_test have been initialized!")
            
            # scale X_train and X_test
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info(f"Shape of X_train scaled: {X_train_scaled.shape}")
            logger.info(f"Shape of X_test_scaled: {X_test_scaled.shape}")
            
            return X_train_scaled, X_test_scaled
        except Exception as exc:
            logger.error("Failed Split and transform to features: %s", exc)
            return None, None
        
        
    def split_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Split targets into 80/20 split.
        
        Returns:
            y_train, y_test (np.ndarray, np.ndarray):
            - y_train: 80% of targets.
            - y_test: 20% of targets.
        """
        try:
            y_train, y_test = train_test_split(
                self.data[self.config['target']],
                test_size=0.20,
                random_state=42
                )
            logger.info(f"Shape of y_train: {y_train.shape}")
            logger.info(f"Shape of y_test: {y_test.shape}")
            return y_train, y_test
        except Exception as exc:
            logger.error("Failed to perform split on targets: %s", exc)
            return None,None
            