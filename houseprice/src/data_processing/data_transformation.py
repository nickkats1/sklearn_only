from src.data_processing.data_ingestion import DataIngestion
from src.config import load_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging

logger = logging.getLogger(__name__)



class DataTransformation:
    def __init__(self,config):
        self.config = config
        
        
    def clean_data(self):
        """Clean features from data ingestion"""
        try:
            RAW_PATH = self.config['data_raw']
            self.data = pd.read_csv(RAW_PATH,delimiter=",")
            # drop duplicated values
            self.data.drop_duplicates(inplace=True)
            return self.data
        except FileNotFoundError as e:
            raise e
        
        
    def split(self):
        """Split data into training and testing dataframe"""
        try:
            self.features = self.data.drop('price',axis=1)
            self.target = self.data['price']
            # split y
            
            self.y_train_df,self.y_test_df = train_test_split(self.target,test_size=.20,random_state=42)
            self.df_train,self.df_test = train_test_split(self.features,test_size=.20,random_state=42)
            
            self.df_train.to_csv(self.config['raw_train'],index=0)
            self.df_test.to_csv(self.config['raw_test'],index=0)
            
            self.y_train_df.to_csv(self.config['train_target_raw'],index=0)
            self.y_test_df.to_csv(self.config['test_target_raw'],index=0)
            
            return self.df_train,self.df_test,self.y_train_df,self.y_test_df
        except ProcessLookupError as e:
            logger.exception(f"Could not process file: {e}")
            raise e
        
        
    def standardize_data(self):
        """Standardize training and testing data"""
        try:
            # features to be scaled for training and testing data 
            # processced path for saving scaled data
            self.df_train =  pd.read_csv(self.config['raw_train'],delimiter=",")
            self.df_test = pd.read_csv(self.config['raw_test'],delimiter=",")
            logger.info(f'Head of df_train and df_test: {self.df_train.head(10),self.df_test.head(10)}')
            scaler = StandardScaler()
            # scaler train and test data
            self.df_train_scaled = scaler.fit_transform(self.df_train)
            self.df_test_scaled = scaler.transform(self.df_test)
            
            # turn the scaled array into a array
            self.df_train_scaled = pd.DataFrame(self.df_train_scaled)
            self.df_test_scaled  = pd.DataFrame(self.df_test_scaled)
            
            # save scaled train/test split to path
            self.df_train_scaled.to_csv(self.config['processed_train'],index=0)
            self.df_test_scaled.to_csv(self.config['processed_test'],index=0)
            return self.df_train_scaled,self.df_test_scaled
        except Exception as e:
            logger.exception(f'Could not split data: {e}')
            raise e
        
        



if __name__ == "__main__":
    config = load_config()
    data_transformation_config = DataTransformation(config)
    data_transformation_config.clean_data()
    data_transformation_config.split()
    data_transformation_config.standardize_data()


