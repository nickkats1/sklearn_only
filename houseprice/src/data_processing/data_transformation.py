from src.config import load_config
from src.logger import logger
from sklearn.preprocessing import StandardScaler
import pandas as pd



class DataTransformation:
    def __init__(self,config):
        self.config = config
        
        
    def standardize_data(self):
        """ use standard scaler to standardize data """
        try:
            self.df_train = pd.read_csv(self.config['raw_train'],delimiter=",")
            self.df_test = pd.read_csv(self.config['raw_test'],delimiter=",")
            
            # use standard scaler to scale data y_train and y_test are not needed
            
            scaler = StandardScaler()
            
            self.df_train_scaled = scaler.fit_transform(self.df_train)
            self.df_test_scaled = scaler.transform(self.df_test)
            
            # turn each into a dataframe to the be turned into a .csv file
            
            self.df_train_scaled = pd.DataFrame(self.df_train_scaled)
            self.df_test_scaled = pd.DataFrame(self.df_test_scaled)
            
            # convert each dataframe to_csv
            self.df_train_scaled.to_csv(self.config['processed_train'],index=0)
            self.df_test_scaled.to_csv(self.config['processed_test'],index=0)
            return self.df_train_scaled,self.df_test_scaled
        
        
        except Exception as e:
            logger.exception(f'Could not load files: {e}')
        raise e


if __name__ == "__main__":
    config = load_config()
    data_transformation_config = DataTransformation(config)
    data_transformation_config.standardize_data()


