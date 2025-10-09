from helpers.config import load_config
from helpers.logger import logger
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

class DataTransformation:
    def __init__(self,config):
        self.config = config
        
        
    def data_cleaning(self):
        """ Clean data """
        try:
            self.df = pd.read_csv(self.config['raw_path'],delimiter=",")
            # features
            self.features=  self.config['features']
            self.training_features = self.df.drop('Default',axis=1)
            
            
            # column transformer
            self.ct = make_column_transformer(
                (OneHotEncoder(),self.training_features.select_dtypes(include="object").columns),
                (StandardScaler(),self.training_features.select_dtypes(include="number").columns)
                ,remainder="passthrough")
            
            self.df_transformed = self.ct.fit_transform(self.training_features)
            self.df_transformed = pd.DataFrame(self.training_features)
            self.df_transformed.to_csv(self.config['processed_path'])
            return self.df_transformed
        
        except Exception as e:
            logger.exception(f"failed: {e}")
            raise e
        
    def split(self):
        """ Split features """
        try:
            # training and testing split
            self.df_train,self.df_test = train_test_split(self.features,test_size=0.10,random_state=1)
            logger.info(f'Length of df_train: {len(self.df_train)}')
            logger.info(f'Length of df_test: {len(self.df_test)}')
            # df_train and df_test to .csv
            self.df_train.to_csv(self.config['train_processed'])
            self.df_test.to_csv(self.config['test_processed'])
            
            # split target for training and testing data
            target = self.df['Default']
            
            self.y_train_df,self.y_test_df = train_test_split(target,test_size=.10,random_state=1)
            logger.info(f'Length of y_train_df: {len(self.y_train_df)}')
            logger.info(f'Length of y_test_df: {len(self.y_test_df)}')
            
            #y_train and y_test to dataframe
            self.y_train_df.to_csv(self.config['train_target_raw'],index=0)
            self.y_test_df.to_csv(self.config['test_target_raw'],index=0)
            return self.y_train_df,self.y_test_df
        except Exception as e:
            raise e



if __name__ == "__main__":
    config = load_config()
    data_transformation_config = DataTransformation(config)
    data_transformation_config.data_cleaning()
    data_transformation_config.split()