from src.config import load_config,load_jobs
from pathlib import Path
import joblib
from src import logger
import pandas as pd
import numpy as np

class PredictPipeline:
    def __init__(self,config):
        self.config = config
        self.model = joblib.load(Path(self.config['model_path']))


    def predict(self,features):
        """
        predicted probabitlity from .joblib
        """
        try:
            if not isinstance(features,pd.DataFrame):
                features = pd.read_csv(Path(self.config['test_scaled_path']))

            pred_prob = self.model.predict_proba(features)
            return np.round(pred_prob,2)

        except Exception as e:
            logger.exception(e)
            raise e



if __name__ == "__main__":
    config = load_config()
    predict_pipeline = PredictPipeline(config)
            

            


