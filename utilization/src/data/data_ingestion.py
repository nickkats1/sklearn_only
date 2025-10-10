from helpers.logger import logger
from helpers.config import load_config
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class DataIngestion:
    def __init__(self,config):
        self.config = config
        
    def fetch_data(self):
        """ Fetch data from url links """
        # links to datasets
        try:
            applications = pd.read_csv(self.config['applications_url'],delimiter=",")
            demographic = pd.read_csv(self.config['demographic_url'],delimiter=",")
            credit = pd.read_csv(self.config['credit_url'],delimiter=",")
            combined = pd.concat([applications,credit,demographic],axis=1)
            # remove duplicated columns
            combined = combined.loc[:,~combined.columns.duplicated()].copy()
            # creating variable utility
            combined['Utility'] = combined['purchases'] / combined['credit_limit']
            
            # creating target variable log odds utility
            combined['log_odds_utils'] = np.log(combined['Utility']) / (combined['Utility'] - 1)
            
            # making homeownership binary
            combined['homeownership'] = [1 if X == "Rent" else 0 for X in combined['homeownership']] 
            # drop utility
            combined.drop("Utility",axis=1,inplace=True)
            combined = combined.dropna()
            combined.drop_duplicates(inplace=True)
            combined.to_csv(self.config['raw_path'],index=0)
            return combined
            
                
        except Exception as e:
            logger.exception(f"Could not get link: {e}")
            raise e
        
    def split(self):
        """ Split training and testing data """
        try:
            #features
            combined = self.fetch_data()
            features = combined.drop("log_odds_utils",axis=1)
            target = combined['log_odds_utils']
            #train test split
            
            df_train,df_test = train_test_split(features,test_size=0.20,random_state=42)
            df_train.to_csv(self.config['train_raw'],index=0)
            df_test.to_csv(self.config['test_raw'],index=0)
            logger.info(f"Shape of df_train: {df_train.shape}")
            logger.info(f"Shape of df_test: {df_test.shape}")            
            # split target values
            y_train_df,y_test_df = train_test_split(target,test_size=.20,random_state=42)
            y_train_df.to_csv(self.config['train_target_raw'],index=0)
            y_test_df.to_csv(self.config['test_target_raw'],index=0)
            logger.info(f'Shape of y_train_df: {y_train_df.shape}')
            logger.info(f'Shape of y_test_df: {y_test_df.shape}')

        except Exception as e:
            logger.exception(f'Could not find path: {e}')
            raise None
        

if __name__ == "__main__":
    config = load_config()
    data_ingestion_config = DataIngestion(config)
    data_ingestion_config.fetch_data()
    data_ingestion_config.split()