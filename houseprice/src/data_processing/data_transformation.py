from helpers.config import load_config
from helpers.logger import logger
from sklearn.preprocessing import StandardScaler
import pandas as pd


class DataTransformation:
    def __init__(self,config):
        self.config = config
        
    def standardize_data(self) -> pd.DataFrame:
        """ Standardize X_train,X_test """
        try:
            df_train = pd.read_csv(self.config['raw_train'],delimiter=",")
            df_test = pd.read_csv(self.config['raw_test'],delimiter=",")
            
            # scaler X_train,X_test using standard scaler
            
            scaler = StandardScaler()
            df_train_scaled = scaler.fit_transform(df_train)
            df_test_scaled = scaler.transform(df_test)
            # convert to dataframe
            df_train_scaled = pd.DataFrame(df_train_scaled)
            df_test_scaled = pd.DataFrame(df_test_scaled)
            
            # save transformed data to processed path
            
            df_train_scaled.to_csv(self.config['processed_train'],index=0)
            df_test_scaled.to_csv(self.config['processed_test'],index=0)

        
        except Exception as e:
            raise e

