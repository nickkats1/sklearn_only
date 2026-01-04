# Helpers
from helpers.config import load_config
from helpers.logger import logger

# Pandas
import pandas as pd

class DataIngestion:
    """Retrieve raw data from url link and save as CSV file."""
    
    def __init__(self, config: dict):
        """Initialize data ingestion class.
        
        Args:
            config (dict): A configuration file consisting of url paths, model path, features, and target.
        """
        self.config = config or load_config()
        
    def get_data(self) -> pd.DataFrame:
        """Fetch data and combined from url links.
        
        Returns:
            data (pd.DataFrame): A dataframe fetched from url link.
        """
        try:
            # url links
            credit_url = self.config['credit_url']
            app_url = self.config['app_url']
            demo_url = self.config['demo_url']
            
            # Read CSV files into dataframe.
            credit_df = pd.read_csv(credit_url,delimiter=",")
            app_df = pd.read_csv(app_url, delimiter=",")
            demo_df = pd.read_csv(demo_url, delimiter=",")
            
            # concat data
            
            data = pd.concat([credit_df, app_df, demo_df], axis=1)
            
            data = data.loc[:, ~data.columns.duplicated()].copy()
            
            return data
        except Exception as exc:
            logger.error(f"Could not retrieve data from url sources: {exc}")
            return None
