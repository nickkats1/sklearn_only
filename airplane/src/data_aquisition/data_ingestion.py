from helpers.config import load_config
from helpers.logger import logger
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

class DataIngestion:
    def __init__(self,config):
        self.config = config
        
        
    def fetch_data(self) -> pd.DataFrame:
        """ Fetch Data from Url Links """
        try:
            # URL Links from config
            
            AIRPLANE_SALES_LINK = self.config['airplane_sales_link']
            AIRPLANE_SPECS_LINK = self.config['airplane_specs_link']
            AIRPLANE_PERF_LINK = self.config['airplane_perf_link']
            
            # load in links from url
            
            airplane_sales = pd.read_csv(AIRPLANE_SALES_LINK,delimiter=",")
            airplane_sales['age'] = airplane_sales[['age']]
            airplane_specs = pd.read_csv(AIRPLANE_SPECS_LINK,delimiter=",")
            airplane_perf = pd.read_csv(AIRPLANE_PERF_LINK,delimiter=",")
            airplane_sales_specs = pd.concat([airplane_specs,airplane_sales],axis=1)
            # drop duplicated columns
            
            airplane_sales_specs = airplane_sales_specs.loc[:,~airplane_sales_specs.columns.duplicated()].copy()

            # all of the dataframes combined
            data = pd.concat([airplane_sales,airplane_specs,airplane_perf],axis=1)
            data = data.loc[:,~data.columns.duplicated()].copy()
            data.rename(columns={"pass":"pas"},inplace=True)
            
            # features to log
            
            data['ceiling'] = data['ceiling'].apply(lambda x: np.log(x+1))
            data['fuel'] = data['fuel'].apply(lambda x: np.log(x+1))
            data['horse'] = data['horse'].apply(lambda x: np.log(x+1))
            data['cruise'] = data['cruise'].apply(lambda x: np.log(x+1))
            data.drop_duplicates(inplace=True)
            
            data.to_csv(self.config['raw_path'],index=0)

            
            

            
            
            return data
        except ValueError as e:
            logger.exception(f"Value error: {e}")
        raise
    
    
    def split(self) -> pd.DataFrame:
        """ dropping variables that are not needed and split dataframe """
        try:
            data = pd.read_csv(self.config['raw_path'],delimiter=",")
            
            # variables and target
            features = self.config['features']
            target = self.config['target']
            
            features_df = data[features]
            target_df  = data[target]
            target_df_log = np.log(target_df)
            # train test split for features
            df_train,df_test = train_test_split(features_df,test_size=.20,random_state=1)
            # train test split for features
            y_train,y_test = train_test_split(target_df_log,test_size=.20,random_state=1)
            



            # X_train,X_test to .csv file
            df_train.to_csv(self.config['train_raw'],index=0)
            df_test.to_csv(self.config['test_raw'],index=0)
            
            
            # y_train y_test to .csv file


            y_train.to_csv(self.config['y_train_path'],index=0)
            y_test.to_csv(self.config['y_test_path'],index=0)
            
            logger.info(f"Shape of X_train: {df_train.shape}")
            logger.info(f"Shape of X_test: {df_test.shape}")
            logger.info(f"Shape of y_train: {y_train.shape}")
            logger.info(f"Shape of y_test: {y_test.shape}")
            
            
            return df_train,df_test,y_train,y_test
        
        
        except Exception as e:
            raise e

