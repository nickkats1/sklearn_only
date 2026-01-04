# helpers
from helpers.config import load_config
from helpers.logger import logger

# pandas
import pandas as pd


class DataIngestion:
    """Utility class for fetching data from url link and returning as CSV file."""
    
    
    def __init__(self, config: dict):
        """Initialize data ingestion class
        
        Args:
            config (dict): Configuration dictionary.
        """
        
        self.config = config or load_config()
        
    def get_data(self) -> pd.DataFrame:
        """Fetch raw data from source and save as pd.DataFrame.
        
        Returns:
            data (pd.DataFrame): Raw CSV file from source.
        """
        try:
            url_link = self.config['url_link']
            if 'url_link' not in self.config:
                raise ValueError("url link does not exist")

            
            data = pd.read_csv(url_link, delimiter=",")
            return data
        except Exception as exc:
            logger.error("Could Not retrieve file from link: %s",exc)
            return pd.DataFrame()
