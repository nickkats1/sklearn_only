import pandas as pd
import numpy as np
from helpers.config import load_config
from urllib.request import urlretrieve
import numpy as np
from helpers import logger
from sklearn.model_selection import train_test_split




class DataIngestion:
    def __init__(self,config):
        self.config = config
        
        
    
    def fetch_data(self):
        """Fetch data from url link"""
        try:
            URL_LINK = self.config['url_link']
            RAW_PATH = self.config['raw_path']
        
            urlretrieve(url=URL_LINK,filename=RAW_PATH)
            return URL_LINK,RAW_PATH
        except Exception as e:
            logger.error(f"Url could not be loaded to path: {e}")
            raise e
    
    def load_data(self) -> pd.DataFrame:
        """ converts urlretrive to dataframe """
        try:
            data = pd.read_csv(self.config['raw_path'],delimiter=",")
            data.drop_duplicates(inplace=True)
            data.to_csv(self.config['raw_path'],index=0)
            
            #features to be split
            features = data.drop('Default',axis=1)
            target = data['Default']
            
            self.df_train,self.df_test = train_test_split(features,test_size=.20,random_state=42)
            self.df_train.to_csv(self.config['train_raw'],index=0)
            self.df_test.to_csv(self.config['test_raw'],index=0)
            
            # target features split
            self.y_train_df,self.y_test_df = train_test_split(target,test_size=0.20,random_state=42)
            self.y_train_df.to_csv(self.config['train_target_raw'],index=0)
            self.y_test_df.to_csv(self.config['test_target_raw'],index=0)
            
            

        except FileExistsError as e:
            logger.exception(f"Could not find file: {e}")
            raise

        
        

if __name__ == "__main__":
    config = load_config()
    data_ingestion_config=  DataIngestion(config)
    data_ingestion_config.fetch_data()
    data_ingestion_config.load_data()