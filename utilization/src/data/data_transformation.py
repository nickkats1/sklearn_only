from helpers.config import load_config
from helpers.logger import logger
from sklearn.preprocessing import StandardScaler
import pandas as pd


class DataTransformation:
    def __init__(self,config):
        self.config = config
        
    def transform_data(self):
        """ fetch data from DataIngestion and scaled the train test split"""
        try:
            df_train = pd.read_csv(self.config['train_raw'],delimiter=",")
            df_test = pd.read_csv(self.config['test_raw'],delimiter=",")
            
            # scaling data
            scaler = StandardScaler()
            df_train_scaled = scaler.fit_transform(df_train)
            df_test_scaled = scaler.transform(df_test)
            # turning arrays to dataframe
            df_train_scaled = pd.DataFrame(df_train_scaled)
            df_test_scaled = pd.DataFrame(df_test_scaled)
            df_train_scaled.to_csv(self.config['train_processed'],index=0)
            df_test_scaled.to_csv(self.config['test_processed'],index=0)

        except Exception as e:
            logger.exception(f"{e}")
            raise e



        


if __name__ == "__main__":
    config = load_config()
    data_transformation_config = DataTransformation(config)
    data_transformation_config.transform_data()
