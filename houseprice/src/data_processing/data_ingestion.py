import pandas as pd
import numpy as np
from src.config import load_config
import logging
from urllib.request import urlretrieve
import numpy as np

logger = logging.getLogger(__name__)



class DataIngestion:
    def __init__(self,config):
        self.config = config
        
        
    
    def fetch_data(self):
        """Fetch data from url link"""
        try:
            self.URL_LINK = self.config['url_link']
            self.RAW_PATH = self.config['data_raw']
        
            urlretrieve(url=self.URL_LINK,filename=self.RAW_PATH)
            return self.URL_LINK,self.RAW_PATH
        except Exception as e:
            logger.error(f"Url could not be loaded to path: {e}")
            raise e
    
    def features_targets(self):
        """Define features and target variables"""
        try:
            self.df = pd.read_csv(self.RAW_PATH,delimiter=',')
            self.features = self.df.drop(self.config['target'],axis=1)
            self.target = self.df[self.config['target']]
            return [self.features,self.target]
        except Exception as e:
            logger.exception(f"Could not Split data into features and target: {e}")
            raise e

        
        

if __name__ == "__main__":
    config = load_config()
    data_ingestion_config=  DataIngestion(config)
    data_ingestion_config.fetch_data()
    data_ingestion_config.features_targets()
