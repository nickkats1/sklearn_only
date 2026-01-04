# data ingestion for data
from src.data.data_ingestion import DataIngestion

# logger
from helpers.logger import logger
from helpers.config import load_config

# Standard scaler and train_test split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# numpy and pandas
import numpy as np
import pandas as pd

from typing import Tuple

class DataTransformation:
    """A utility class to split the data from data ingestion and scaled the training and testing features."""
    
    def __init__(self, config: dict):
        """__init__ data ingestion.
        
        Args:
            config (dict): A config file containing all of the features, targets, url links, paths needs.
        """
        
        self.config = config or load_config()
        self.scaler = StandardScaler()
        
    def split_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Split training and testing features and scale them using StandardScaler.
        
        Returns:
            X_train_scaled(np.ndarray): 80% of features with standard scaled applied.
            X_test_scaled(np.ndarray): 20% of features with standard scaler applied.
        """
        try:
            # load in data from data ingestion.
            data = DataIngestion(self.config).fetch_raw_data()
            
        except ImportError or FileNotFoundError as exc:
            logger.info(f"Could not import data ingestion or file was not found: {exc}")
        
        # features
        try:
            features = self.config['features']
        
            X_train, X_test = train_test_split(data[features], test_size=0.20, random_state=42)
        
            # scale training and testing features
        
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            logger.info(f"Shape of X_train: {X_train.shape}")
            logger.info(f"Shape of X_test: {X_test_scaled.shape}")
            
            return X_train_scaled, X_test_scaled
        except Exception as e:
            logger.info(f"Could not split and scale training and testing features: {e}")
        return [],[]
    
    
    def split_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Split targets.
        
        Returns:
            y_train (np.ndarray): training targets.
            y_test (np.ndarray): testing targets.
        """
        
        try:
            # targets
            data = DataIngestion(self.config).fetch_raw_data()
            targets = self.config['target']
            
            y_train, y_test = train_test_split(data[targets], test_size=0.20, random_state=42)
            logger.info(f"Shape of y_train: {y_train.shape}")
            logger.info(f"Shape of y_test: {y_test.shape}")
            return y_train, y_test
        except Exception as e:
            logger.info(f"Error splitting targets or error with import DataIngestion or something else: {e}")
        return None, None
            