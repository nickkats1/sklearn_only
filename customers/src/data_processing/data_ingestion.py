import pandas as pd
from urllib.request import urlretrieve
import os
from sklearn.model_selection import train_test_split
from src.logger import logger
from src.config import load_config

class DataIngestion:
    def __init__(self, config):
        self.config = config

    def fetch_data(self):
        """
        Fetches data from datalink and stores it locally.
        """
        try:
            urlretrieve(url=self.config['url_link'], filename=self.config['data_raw'])
            logger.info(f'Data Was Loaded From URL at: {self.config["data_raw"]}')
        except Exception as e:

            logger.error(f"File Could Not Be found or downloaded: {e}")
            raise 

    def load_data(self):
        """
        Loads raw data from the specified path and removes duplicate rows.
        """
        try:
            raw_data = pd.read_csv(self.config['data_raw'])
            raw_data.drop_duplicates(inplace=True)
            logger.info("Duplicates removed from raw data.")
            return raw_data
        except FileNotFoundError:
            logger.error(f"File not found at: {self.config['data_raw']}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Empty data or header encountered in {self.config['data_raw']}.")
            raise
        except Exception as e:
            logger.error(f'File Could Not Be found or loaded: {e}')
            raise e

    def split_data(self):
        """Splits the raw data into training and testing sets."""
        try:
            df = self.load_data()
            df.drop_duplicates(inplace=True)
            df_train, df_test = train_test_split(
                df,
                test_size=self.config['test_size'],
                random_state=self.config['random_state']
            )
            os.makedirs(os.path.dirname(self.config['train_raw']), exist_ok=True)
            os.makedirs(os.path.dirname(self.config['test_raw']), exist_ok=True)

            pd.DataFrame(df_train).to_csv(self.config['train_raw'], index=False)
            pd.DataFrame(df_test).to_csv(self.config['test_raw'], index=False)
            logger.info(f"Data split into training ({self.config['train_raw']}) "
                        f"and testing ({self.config['test_raw']}) sets.")
            logger.info(f'Shape of df_train: {df_train.shape}')
            logger.info(f'Shape of df_test: {df_test.shape}')
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise e



if __name__ == "__main__":
    config = load_config()
    data_ingestion = DataIngestion(config)
    data_ingestion.fetch_data()
    data_ingestion.load_data()
    data_ingestion.split_data()

