from helpers.config import load_config
from helpers.logger import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from urllib.request import urlretrieve

class DataIngestion:
    def __init__(self,config):
        self.config = config
        
    def fetch_data(self) -> pd.DataFrame:
        """ Retrieves data from url link """
        try:
            # path to url link
            URL_LINK = self.config['url_link']
            RAW_PATH = self.config['raw_path']
            urlretrieve(url=URL_LINK,filename=RAW_PATH)
            
        except Exception as e:
            raise e
    
    
    def split(self) -> pd.DataFrame:
        """ Split data loaded from first part of data ingestion """
        
        # load in data from url retrieve
        
        data = pd.read_csv(self.config['raw_path'],delimiter=",")
        data.drop_duplicates(inplace=True)
        
        # features and target for train/test split

        
        df_features = data.drop("price",axis=1)
        df_target = data['price']
        

        
        # X_train,X_test split
        df_train,df_test=  train_test_split(df_features,test_size=0.20,random_state=0)
        
        # save to .csv file
        df_train.to_csv(self.config['raw_train'],index=0)
        df_test.to_csv(self.config['raw_test'],index=0)
        
        # train/test split for targets
        
        y_train_df,y_test_df = train_test_split(df_target,test_size=0.20,random_state=0)
        
        # safe y_train,y_test to .csv file
        y_train_df.to_csv(self.config['train_target_raw'],index=0)
        y_test_df.to_csv(self.config['test_target_raw'],index=0)
        return data,df_train,df_test,y_train_df,y_test_df





if __name__ == "__main__":
    config = load_config()
    data_ingestion_config = DataIngestion(config)
    data_ingestion_config.fetch_data()
    data_ingestion_config.split()