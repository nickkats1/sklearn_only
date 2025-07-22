from src.config import load_config,load_jobs
from src.data_processing.data_transformation import DataTransformation
import pandas as pd


class PredictPipeline:
    def __init__(self,config):
        self.config = config
        
    def predict_pipeline(self,features):
        """
        training the processed data
        """
        try:
            model_path = self.config['trained_model_path']
            model = load_jobs(model_path)
            pred = model.predict(features)[0]
            return round(pred,2)
        except Exception as e:
            raise e


if __name__ == "__main__":
    config = load_config()
    pred_pipeline = PredictPipeline(config)