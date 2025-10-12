from helpers.config import load_config
from helpers.logger import logger
import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataTransformation:
    def __init__(self,config):
        self.config = config
        
    def standardize_data(self) -> pd.DataFrame:
        """ Use standard scaled to scaled training and testing data """
        try:
            df_train = pd.read_csv(self.config['train_raw'],delimiter=",")
            df_test = pd.read_csv(self.config['test_raw'],delimiter=",")
            
            # scaled df_train,df_test
            
            # load in scaler
            
            scaler = StandardScaler()
            
            df_train_scaled = scaler.fit_transform(df_train)
            df_test_scaled = scaler.transform(df_test)
            
            # convert df_train_scaled,df_test_scaled to dataframe
            
            df_train_scaled = pd.DataFrame(df_train_scaled)
            df_test_scaled = pd.DataFrame(df_test_scaled)
            
            df_train_scaled.to_csv(self.config['train_scaled_path'],index=0)
            df_test_scaled.to_csv(self.config['test_scaled_path'],index=0)
            logger.info(f"Shape of df_train_scaled : {df_train_scaled.shape}")
            logger.info(f"Shape of df_test_scaled: {df_test_scaled.shape}")
            return df_train_scaled,df_test_scaled
        
        except Exception as e:
            logger.exception(f"{e}")
        raise e


