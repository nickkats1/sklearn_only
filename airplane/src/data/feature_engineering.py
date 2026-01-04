# data ingestion
from src.data.data_ingestion import DataIngestion

# helpers logger
from helpers.config import load_config
from helpers.logger import logger


# Numpy
import numpy as np

# Pandas
import pandas as pd

class FeatureEngineering:
    """Feature engineering class to clean features and create features."""
    
    def __init__(self, config: dict, data: DataIngestion | None = None):
        """Initialize Feature Engineering class.
        
        Args:
            config (dict): Configuration file.
            data (DataIngestion): Instance of data ingestion class.
        """
        self.config = config or load_config()
        self.data = data or DataIngestion(self.config).fetch_raw_data()
        
    def select_features(self) -> pd.DataFrame:
        """select features and clean features.
        
        Returns:
            data (pd.DataFrame):
            - Clean DataFrame with optimal features.
        """
        try:
            
            data = self.data
            if data is None:
                raise KeyError("Data Ingestion failed")
            
            self.data['horse'] = np.log(self.data['horse']).dropna()
            self.data['price'] = np.log(self.data['price']).dropna()
            self.data['age'] = np.log(self.data['age']).dropna()
            self.data['fuel'] = np.log(self.data['fuel']).dropna()
            
            self.data.dropna(inplace=True)
            self.data.drop_duplicates(inplace=True)
            
            return data
        except Exception as exc:
            logger.error("Feature Engineering Failed: %s", exc)
            return None