# import pandas
import pandas as pd

# import logger and config
from helpers.config import load_config
from helpers.logger import logger


class DataIngestion:
    """Get data from source and return as TXT file."""
    
    
    def __init__(self, config: dict):
        """Initialize DataIngestion class.
        
        Args:
            config (dict): A configuration file containing features, targets, file paths.
        """
        self.config = config or load_config()

        
    def get_data(self) -> pd.DataFrame:
        """Get data from url link and return as pd.DataFrame.
        
        Returns:
            data (pd.DataFrame): Data From Url Link.
        """
        try:
  
            data = pd.read_csv(self.config['url_link'], delimiter="\t")
            return data
        except Exception as e:
            logger.error(f"File does not exits or response error: {e}")
        return None