# import pandas as pd
import pandas as pd

# config
from helpers.config import load_config
from helpers.logger import logger
# numpy
import numpy as np

class DataIngestion:
    """Utility class to fetch data from url links and save to raw paths"""
    
    def __init__(self, config: dict):
        """__init__ DataIngestion module.

        Args:
            config (dict): configuration file consisting of features, url links, file paths, and targets.
        """
        self.config = config or load_config()
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch data from url links and return as pd.DataFrame.
        
        Returns:
            data (pd.DataFrame): A cleaned pd.DataFrame consisting of the CSV from the url links.
        """
        
        try:
            # url links
            
            airplane_sales_link = self.config['airplane_sales_link']
            airplane_specs_link = self.config['airplane_specs_link']
            airplane_perf_link = self.config['airplane_perf_link']
            
            
            # load in from raw urls
            
            airplane_sales = pd.read_csv(airplane_sales_link, delimiter=",")
            airplane_specs = pd.read_csv(airplane_specs_link, delimiter=",")
            airplane_perf = pd.read_csv(airplane_perf_link, delimiter=",")
            
            logger.info(f"Shape of airplane sales: {airplane_sales.shape}")
            logger.info(f"Shape of airplane_specs: {airplane_specs.shape}")
            logger.info(f"Shape of airplane_perf: {airplane_perf.shape}")
            
            
            # airplanes sales specs dataframe.
            airplane_sales_specs = pd.concat([airplane_specs,airplane_sales],axis=1)
            
            # drop duplicated columns
            airplane_sales_specs = airplane_sales_specs.loc[:,~airplane_sales_specs.columns.duplicated()].copy()
            
            logger.info(f"Shape of airplane specs: {airplane_specs.shape}")
            
            # all data combined
            
            data = pd.concat([airplane_sales,airplane_specs,airplane_perf],axis=1)

            data = data.loc[:,~data.columns.duplicated()].copy()
            
            logger.info(f"Shape of dataframe: {data.shape}")
            
            # replace 'pass' with 'pas'.
            data['pas'] = data['pass']
            data.drop("pass", axis=1, inplace=True)
            
            # create features
            data['ceiling'] = data['ceiling'].apply(lambda x: np.log(x+1))
            data['fuel'] = data['fuel'].apply(lambda x: np.log(x+1))
            data['horse'] = data['horse'].apply(lambda x: np.log(x+1))
            data['cruise'] = data['cruise'].apply(lambda x: np.log(x+1))
        
            data.drop_duplicates(inplace=True)

            return data
        except FileNotFoundError:
            logger.info("File could not be found")
            return None
        

