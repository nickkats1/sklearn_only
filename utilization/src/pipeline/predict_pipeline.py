import pandas as pd
from helpers.config import load_config,load_file
from helpers.logger import logger



class PredictPipeline:
    def __init__(self,config):
        self.config = config
        
    def predict_pipeline(self,features):
        """ Predict model best model """
        try:
            # load model path
            model_path = self.config['trained_model_path']
            # load save .pkl scaler
            scaler_path = self.config['scaler_path']
            model = load_file(model_path)
            scaler = load_file(scaler_path)
            scaled_features = scaler.fit_transform(features)
            pred = model.predict(scaled_features)[0]
            return round(pred,2)
        except Exception as e:
            raise e