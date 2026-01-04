# data ingestion
from src.data.data_ingestion import DataIngestion

# helpers
from helpers.config import load_config
from helpers.logger import logger

# pandas
import pandas as pd

class FeatureEngineering:
    """Class to select relevant features from data ingestion."""
    
    def __init__(self, config: dict, data: DataIngestion | None = None):
        """ Initialize features module.
        
        Args:
            config (dict): Configuration file containing relevant features, paths, target ect.
            data (DataIngestion): A module for getting data from source.
        """
        self.config = config or load_config()
        self.data = data or DataIngestion(self.config).get_data()
        
    def select_features(self) -> pd.DataFrame:
        """ Select Relevant features based on the paper or ones that explain the dependent variable.
        
        Returns:
            data (pd.DataFrame): Cleaned data from data ingestion with no missing values, duplicates or irrelevant features.
        """
        try:
            
            data = self.data
            if data is None:
                raise ValueError("Could not get data from data ingestion.")
            
            # select features based on description in paper.
            
            data.rename(
            columns={
                's5':'occupancy',
                's7':'approved',
                's13':'race',
                's15':'sex',
                's17':'income',
                's23a':'married',
                's27a':'self_employed',
                's33':'purchase_price',
                's34':'other_financing',
                's35':'liquid_assets',
                's40':'credit_history',
                's42':'chmp',
                's43':'chcp',
                's44':'chpr',
                's45':'debt_to_expense',
                's46':'di_ratio',
                's50':'appraisal',
                's53':'pmi_denied',
                's56':'unverifiable',
                's52':'pmi_sought'},inplace=True)
            
            
            # change features to numerical values manually
            
            # target
            data['approved'] = [1 if X == 3 else 0 for X in data['approved']]
            
            data['race'] = [0 if X == 3 else 1 for X in data['race']]
            data['married'] = [1 if X == 'M' else 0 for X in data['married']]
            data['sex'] = [1 if X == 1 else 0 for X in data['sex']]
            data['credit_history'] = [1 if X == 1 else 0 for X in data['credit_history']]
            
            # drop unused variables specified in config.yaml
            
            unused_variables = self.config['unused_variables']
            
            # drop unused variables
            data.drop(unused_variables, inplace=True, axis=1)
            # drop duplicates
            
            data.drop_duplicates(inplace=True)
            return data
        except Exception as e:
            return f"Invalid type or could not perform feature selection: {e}"
