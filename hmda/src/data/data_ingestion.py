# Helpers
from helpers.config import load_config
from helpers.logger import logger

# Pandas
import pandas as pd

class DataIngestion:
    """Utility class to fetch data from url link and return as pd.DataFrame."""
    
    
    def __init__(self, config: dict):
        """Initialize data ingestion class.
        
        Args:
            config (dict): A configuration file containing all features, targets and file paths.
        """
        
        self.config = config or load_config()
        
        
    def get_data(self) -> pd.DataFrame:
        """Fetch data from url link and return as pd.DataFrame
        
        Returns:
            data (pd.DataFrame):  A pd.DataFrame with data retrieved from url link.
        """
        
        try:
            url_link = self.config['url_link']
            data = pd.read_csv(url_link, delimiter=",")
            return data
        except Exception as exc:
            logger.error(f"failed to retrieve file from link: {exc}")
            return None