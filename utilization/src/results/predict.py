import joblib

from helpers.config import load_config
from helpers.logger import logger



class Predict:
    """Class to predict features from 'model_trainer.py' loaded from PKL file located at artifacts."""
    
    def __init__(self, config: dict):
        """Initialize Predict class.
        
        Returns:
            y_pred (float): The Predicted probability of being approved/denied a loan.
        """
        self.config = config or load_config()
        self.model = self.config['model_path']
        
    def predict(self, features) -> float:
        """Predicts 'best_model.pkl' through features.
        
        Args:
            features(List[pd.Series]): Input features for the 'best model' to predict target.
        """
        try:
            # model path
            model_path = self.config['model_path']
            model = joblib.load(model_path)
            
        
            
            y_pred = model.predict([features])[0]
            return round(y_pred, 2)
        except Exception as e:
            logger.error(f"Length of feature must equal 13{e}")
        return None