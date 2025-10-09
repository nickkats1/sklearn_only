import pandas as pd
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
from helpers.config import load_config
from helpers.logger import logger

class DataIngestion:
    def __init__(self,config):
        self.config = config
        
        
    def fetch_data(self) -> pd.DataFrame:
        """ Fetch data from url """
        try:
            #url link
            self.url_link = self.config['url_link']
            # raw path for the data ingestion
            
            
            self.raw_path = self.config['data_raw']
            
            # data ingestion
            
            urlretrieve(url=self.url_link,filename=self.raw_path)
            
        except Exception as e:
            raise e
        
        

if __name__ == "__main__":
    config = load_config()
    data_ingestion_config = DataIngestion(config)
    data_ingestion_config.fetch_data()
    