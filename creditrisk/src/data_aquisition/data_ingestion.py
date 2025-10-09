from helpers.config import load_config
from helpers.logger import logger
import pandas as pd
from urllib.request import urlretrieve

class DataIngestion:
    def __init__(self,config):
        self.config=  config
        
    def fetch_data(self):
        """" Fetch data from url """
        try:
            # path to url
            URL_PATH = self.config['url_link']
            # path to raw path
            RAW_PATH = self.config['raw_path']
            urlretrieve(url=URL_PATH,filename=RAW_PATH)
            
            
        except Exception as e:
            logger.exception(f"Could not find path: {e}")
            raise e
        


if __name__ == "__main__":
    config = load_config()
    data_ingestion_config = DataIngestion(config)
    data_ingestion_config.fetch_data()

