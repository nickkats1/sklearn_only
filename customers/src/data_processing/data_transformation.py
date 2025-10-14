from sklearn.preprocessing import StandardScaler
from src.config import load_config
from src.logger import logger
import pandas as pd


class DataTransformation:
    def __init__(self,config):
        self.config = config
        
        
    def standardize(self) -> pd.DataFrame:
        """ Standardize training and testing data """
        try:
            df_train = pd.read_csv(self.config['train_raw'],delimiter=",")
            df_test = pd.read_csv(self.config['test_raw'],delimiter=",")
            
            
            # load in scaler and then scale
            scaler = StandardScaler()
            df_train_scaled = scaler.fit_transform(df_train)
            df_test_scaled = scaler.transform(df_test)
            # convert to dataframe
            df_train_scaled = pd.DataFrame(df_train_scaled)
            df_test_scaled = pd.DataFrame(df_test_scaled)
            
            # convert to .csv file
            df_train_scaled.to_csv(self.config['train_processed'],index=0)
            df_test_scaled.to_csv(self.config['test_processed'],index=0)
            return df_train_scaled,df_train_scaled
        except Exception as e:
            logger.exception(f"Could not get data: {e}")
        raise