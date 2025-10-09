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
            self.applications = pd.read_csv(self.config['applications_url'],delimiter=",")
            self.demographic = pd.read_csv(self.config['demographic_url'],delimiter=",")
            self.credit = pd.read_csv(self.config['credit_url'],delimiter=",")
            self.combined = pd.concat([self.applications,self.credit,self.demographic],axis=1)
            # remove duplicated columns
            self.combined = self.combined.loc[:,~self.combined.columns.duplicated()].copy()
            # creating variable utility
            self.combined['Utility'] = self.combined['purchases'] / self.combined['credit_limit']
            
            # creating target variable log odds utility
            self.combined['log_odds_utils'] = np.log(self.combined['Utility']) / (self.combined['Utility'] - 1)
            
            # making homeownership binary
            self.combined['homeownership'] = [1 if X == "Rent" else 0 for X in self.combined['homeownership']] 
            # drop utility
            self.combined.drop("Utility",axis=1,inplace=True)
            self.combined = self.combined.dropna()
            self.combined.drop_duplicates(inplace=True)
            self.combined.to_csv(self.config['raw_path'],index=0)
            return self.combined
            
                
        except Exception as e:
            logger.exception(f"Could not get link: {e}")
            raise e
        
    def split(self):
        """ Split training and testing data """
        try:
            #features
            self.features = self.combined.drop("log_odds_utils",axis=1)
            self.target = self.combined['log_odds_utils']
            #train test split
            
            self.df_train,self.df_test = train_test_split(self.features,test_size=0.20,random_state=42)
            self.df_train.to_csv(self.config['train_raw'],index=0)
            self.df_test.to_csv(self.config['test_raw'],index=0)
            logger.info(f"Shape of df_train: {self.df_train.shape}")
            logger.info(f"Shape of df_test: {self.df_test.shape}")            
            # split target values
            self.y_train_df,self.y_test_df = train_test_split(self.target,test_size=.20,random_state=42)
            self.y_train_df.to_csv(self.config['train_target_raw'],index=0)
            self.y_test_df.to_csv(self.config['test_target_raw'],index=0)
            logger.info(f'Shape of y_train_df: {self.y_train_df.shape}')
            logger.info(f'Shape of y_test_df: {self.y_test_df.shape}')
            return self.df_train,self.df_test,self.y_train_df,self.y_train_df
        except Exception as e:
            logger.exception(f'Could not find path: {e}')
            raise None
        

if __name__ == "__main__":
    config = load_config()
    data_ingestion_config = DataIngestion(config)
    data_ingestion_config.fetch_data()
    data_ingestion_config.split()