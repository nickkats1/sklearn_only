import pandas as pd
import numpy as np
from src.config import load_config
from urllib.request import urlretrieve
import numpy as np
from src.logger import logger
from sklearn.model_selection import train_test_split




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
    
    def load_data(self) -> pd.DataFrame:
        """ converts urlretrive to dataframe """
        try:
            self.data = pd.read_csv(self.config['data_raw'],delimiter=",")
            self.data.drop_duplicates(inplace=True)
            self.data.to_csv(self.config['data_raw'],index=0)
            
            #features to be split
            self.features = self.data.drop('price',axis=1)
            self.target = self.data['price']
            
            self.df_train,self.df_test = train_test_split(self.features,test_size=.20,random_state=42)
            self.df_train.to_csv(self.config['raw_train'],index=0)
            self.df_test.to_csv(self.config['raw_test'],index=0)
            
            # target features split
            self.y_train_df,self.y_test_df = train_test_split(self.target,test_size=0.20,random_state=42)
            self.y_train_df.to_csv(self.config['train_target_raw'],index=0)
            self.y_test_df.to_csv(self.config['test_target_raw'],index=0)
            
            
            return self.df_train,self.df_test,self.y_train_df,self.y_test_df
        except FileExistsError as e:
            logger.exception(f"Could not find file: {e}")
            raise

        
        

if __name__ == "__main__":
    config = load_config()
    data_ingestion_config=  DataIngestion(config)
    data_ingestion_config.fetch_data()
<<<<<<< HEAD
<<<<<<< Updated upstream
    data_ingestion_config.load_data()
=======
    data_ingestion_config.features_targets()
>>>>>>> main
=======
    data_ingestion_config.features_targets()
=======
    data_ingestion_config.load_data()
>>>>>>> edit
>>>>>>> Stashed changes
