import pandas as pd
import numpy as np
from src.config import load_config
from urllib.request import urlretrieve
import numpy as np
from sklearn.model_selection import train_test_split
from src.logger import logger
from src.config import load_config



class DataIngestion:
    def __init__(self,config):
        self.config = config
        
        
    
    def fetch_data(self):
        """Fetch data from url link"""
        try:
            URL_LINK = self.config['url_link']
            RAW_PATH = self.config['data_raw']
        
            urlretrieve(url=URL_LINK,filename=RAW_PATH)
 
        except Exception as e:
            logger.error(f"Url could not be loaded to path: {e}")
            raise e
    
    def load_and_split_data(self) -> pd.DataFrame:
        """ converts urlretrive to dataframe and splits for training and testing """
        try:
            data = pd.read_csv(self.config['data_raw'],delimiter=",")
            data.drop_duplicates(inplace=True)
            data.to_csv(self.config['data_raw'],index=0)
            
            #features to be split
            features = data.drop('price',axis=1)
            target = data['price']
            
            df_train,df_test = train_test_split(features,test_size=.20,random_state=42)
            df_train.to_csv(self.config['raw_train'],index=0)
            df_test.to_csv(self.config['raw_test'],index=0)
            
            # target features split
            y_train_df,y_test_df = train_test_split(target,test_size=0.20,random_state=42)
            y_train_df.to_csv(self.config['train_target_raw'],index=0)
            y_test_df.to_csv(self.config['test_target_raw'],index=0)
            
            
  
        except Exception as e:
            logger.exception(f"Could not find file: {e}")
            raise e
        
        



        
        

if __name__ == "__main__":
    config = load_config()
    data_ingestion_config=  DataIngestion(config)
    data_ingestion_config.fetch_data()
    data_ingestion_config.load_and_split_data()
