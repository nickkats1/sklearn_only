# data ingestion
from src.data.data_ingestion import DataIngestion

# pandas and numpy
import numpy as np

# helpers
from helpers.config import load_config
from helpers.logger import logger

# typing
from typing import Tuple


# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataTransformation:
    """Split features and targets and scale features using standard scaler."""
    
    def __init__(self, config: dict, data: DataIngestion | None = None):
        """Initialize DataTransformation class.
        
        Args:
            config (dict): Configuration file.
            data (DataIngestion): DataIngestion class.
        """
        
        self.config = config or load_config()
        self.data = data or DataIngestion(self.config).get_data()
        self.scaler = StandardScaler()
        
    def split_transform_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Split features into training and testing array and then scale arrays using StandardScaler.
        
        Args:
            X_train_scaled, X_test_scaled (Tuple[np.ndarray, np.ndarray]):
             - X_train_scaled: training features scaled.
             - X_test_scaled: testing features scaled.
        """
        try:
            data = self.data
            if data is None:
                raise ValueError("DataIngestion could not retrieve raw data from url")
            

            # train test split
            X_train, X_test = train_test_split(
                self.data[self.config['features']],
                test_size=0.20,
                random_state=42
                )
            
            logger.info("training and testing features have been split!")
            
            # scale X_train and X_test
            
    
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info("X_train and X_test have been scaled!")
            logger.info(f"Shape of x_train_scaled: {X_train_scaled.shape}")
            logger.info(f"Shape of X_test Scaled: {X_test_scaled.shape}")
            
            return X_train_scaled, X_test_scaled
        except Exception as e:
            logger.info(f"split transform features failed: $s: {e}")



    def split_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Split targets.
        
        Returns:
            y_train, y_test (Tuple[np.ndarray, np.ndarray]):
            - y_train: training targets.
            - y_test: testing targets.
        """
        try:
            y_train, y_test = train_test_split(
                self.data[self.config['target']],
                test_size=0.20,
                random_state=42
                )
            logger.info("Targets have been split: y_train, y_test are available")
            logger.info(f"Shape of y_train: {y_train.shape}")
            logger.info(f"Shape of y_test: {y_test.shape}")
            return y_train, y_test
        except Exception as e:
            logger.info("Could Not Split Targets: %s", e)
            return None, None