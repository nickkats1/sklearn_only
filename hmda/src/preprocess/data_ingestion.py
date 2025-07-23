from src.config import load_config
from src.logger import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from urllib.request import urlretrieve

class DataIngestion:
    def __init__(self, config):
        self.config = config

    def download_data(self):
        """Retrieves data from url link"""
        try:
            url_link = self.config['url_link']
            raw_path = self.config['data_raw']
            urlretrieve(url=url_link, filename=raw_path)
            logger.info(f"Data has been downloaded from {url_link} to {raw_path}")
        except Exception as e:
            logger.error(f"Data was not downloaded from URL: {e}")
            raise

    def process_data(self):
        """ Loads raw data from the specified path and splits it into train/test sets """
        try:
            raw_data = pd.read_csv(self.config['data_raw'])
            
  
            used_data = pd.read_csv(self.config['used_raw_path'],delimiter=",") # edited the datasets and did fe
            used_data.drop_duplicates(inplace=True)
            df_train, df_test = train_test_split(
                used_data,
                test_size=self.config['test_size'],
                random_state=self.config['random_state']
            )

            df_train.to_csv(self.config['train_raw'], index=False)
            df_test.to_csv(self.config['test_raw'], index=False)
            logger.info("Raw data loaded, and split into train/test sets.")
            return df_train, df_test, used_data
        except Exception as e:
            logger.error(f'File Could Not Be found or loaded during processing: {e}')
            raise e


if __name__ == "__main__":
    config = load_config()
    data_ingestion = DataIngestion(config)
    data_ingestion.download_data()
    train_df, test_df, full_data = data_ingestion.process_data()
    
    logger.info("Data ingestion process completed.")