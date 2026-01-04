# Data Ingestion
from src.data.data_ingestion import DataIngestion

# Pandas Numpy
import pandas as pd
import numpy as np

# Helpers
from helpers.config import load_config
from helpers.logger import logger



class FeatureEngineering:
    """Perform Feature Engineering and Feature Selection on ingested data."""
    
    def __init__(self, config: dict, data: DataIngestion | None = None):
        """Initialize Feature Engineering class.
        
        Args:
            config (dict): Configuration dictionary.
            data (DataIngestion): DataIngestion instance.
        """
        
        self.config = config or load_config()
        self.data = data or DataIngestion(self.config).get_data()
        
    def select_features(self) -> pd.DataFrame:
        """Select relevant features and clean the dataset.
        
        This method performs feature engineering, transforms categorical
        variables, removes intermediate features, and cleans the data
        by handling missing values and duplicates.
        
        Returns:
            pd.DataFrame: A cleaned, feature-engineered dataframe.
            
        Raises ValueError:
            If DataIngestion fails.
        """
        
        try:
            data = self.data
            if data is None:
                raise ValueError("Could Not get Data from DataIngestion")
            
            # create utility feature.
            
            data['utility'] = data['purchases'] / data['credit_limit']
            
            data['log_odds_utils'] = np.log(data['utility']) / (data['utility'] - 1)
            
            # Encode homeownership variable as binary.
            
            data['homeownership'] = [1 if X == "Rent" else 0 for X in data['homeownership']]
            
            # drop intermediate features
            data.drop('utility', inplace=True, axis=1)
            
            # Clean data
            data = pd.DataFrame(data)
            data.dropna(inplace=True)
            data.drop_duplicates(inplace=True)
            
            return data
        except Exception as exc:
            logger.info(f"could not create data or retrieve features: {exc}")
            return None