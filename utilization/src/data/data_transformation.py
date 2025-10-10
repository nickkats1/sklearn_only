<<<<<<< HEAD
<<<<<<< Updated upstream
from helpers.config import load_config
from helpers.logger import logger
=======
from utils.config import load_config
from utils.logger import logger
>>>>>>> main
=======
from utils.config import load_config
from utils.logger import logger
=======
from helpers.config import load_config
from helpers.logger import logger
>>>>>>> edit
>>>>>>> Stashed changes
from sklearn.preprocessing import StandardScaler
import pandas as pd


class DataTransformation:
    def __init__(self,config):
        self.config = config
        
    def transform_data(self):
        """ fetch data from DataIngestion and scaled the train test split"""
        try:
            self.df_train = pd.read_csv(self.config['train_raw'],delimiter=",")
            self.df_test = pd.read_csv(self.config['test_raw'],delimiter=",")
            
            # scaling data
            scaler = StandardScaler()
            self.df_train_scaled = scaler.fit_transform(self.df_train)
            self.df_test_scaled = scaler.transform(self.df_test)
            # turning arrays to dataframe
            self.df_train_scaled = pd.DataFrame(self.df_train_scaled)
            self.df_test_scaled = pd.DataFrame(self.df_test_scaled)
            self.df_train_scaled.to_csv(self.config['train_processed'],index=0)
            self.df_test_scaled.to_csv(self.config['test_processed'],index=0)
            return self.df_train_scaled,self.df_test_scaled
        except Exception as e:
            logger.exception(f"Could not transform data")
<<<<<<< HEAD
<<<<<<< Updated upstream
            raise e
=======
            raise None
>>>>>>> main
=======
            raise None
=======
            raise e
>>>>>>> edit
>>>>>>> Stashed changes
        


if __name__ == "__main__":
    config = load_config()
    data_transformation_config = DataTransformation(config)
    data_transformation_config.transform_data()
<<<<<<< HEAD
<<<<<<< Updated upstream
=======

>>>>>>> main
=======

=======
>>>>>>> edit
>>>>>>> Stashed changes
