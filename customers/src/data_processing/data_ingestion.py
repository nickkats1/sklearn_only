import sqlite3 as sql
import pandas as pd
from src.logger import logger
from src.config import load_config
from sklearn.model_selection import train_test_split
import numpy as np


class DataIngestion:
    def __init__(self,config):
        self.config = config
        
    def wrangle_sqlite_data(self) -> pd.DataFrame:
        """ Select Correct featurs from .sqlite3 to convert to pandas dataframe """
        try:
            con = sql.connect("customers.db")
            
            # data from sqlite3 database that is needed
            
            df = pd.read_sql(
                """
                SELECT 
                a.*, 
                c.*, 
                d.*
                FROM Applications a
                JOIN CreditBureau c ON a.zip_code = c.zip_code
                JOIN Demographic d ON a.zip_code = d.zip_code;""",con)
            
            # close connection to .db
            con.close()
            
            # remove columns that are duplicated
            
            df = df.loc[:,~df.columns.duplicated()].copy()
            df.drop_duplicates(inplace=True)
            
            # make homeownership binary
            df['homeownership'] = df["homeownership"].map({"Rent":0,"Own":1})          
            
            # create variable named utils for the needed target variable
            df['utils'] = df['purchases'] / df['credit_limit']
            
            # create target log_odds_utils
            df['log_odds_utils'] = np.log(df['utils']) / (df['utils'] - 1)
            
            # drop utils variable because it is no longer needed
            df.drop("utils",inplace=True,axis=1)

            # convert df to .csv
            df.to_csv(self.config['raw_path'],index=0)
            return df
        except Exception as e:
            raise e
        
        
        
    def split(self) -> pd.DataFrame:
        """ Convert data ingested into a df_train,df_test dataframe """
        try:
            df = self.wrangle_sqlite_data()
            
            # features and targets
            
            features = self.config['features']
            target = self.config['target']
            
            train_features = df[features]
            train_target = df[target]
            
            df_train,df_test = train_test_split(train_features,test_size=.20,random_state=1)
            logger.info(f"shape of df_train: {df_train.shape}")
            logger.info(f"Shape of df_test: {df_test.shape}")
            # train/test split for target
            y_train,y_test = train_test_split(train_target,test_size=0.20,random_state=1)
            logger.info(f"Shape of y_train: {y_train.shape}")
            logger.info(f"Shape of y_test: {y_test.shape}")
            
            df_train.to_csv(self.config['train_raw'],index=0)
            df_test.to_csv(self.config['test_raw'],index=0)
            
            # y_train,y_test .csv
            y_train.to_csv(self.config['train_target_raw'],index=0)
            y_test.to_csv(self.config['test_target_raw'],index=0)
            
            
            return df_train,df_test,y_train,y_test
        except Exception as e:
            raise e
