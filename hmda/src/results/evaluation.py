# metrics
from sklearn.metrics import roc_auc_score
# joblib to save model
import joblib

# logger and config
from helpers.config import load_config
from helpers.logger import logger

# data transformation
from src.data.data_transformation import DataTransformation

from typing import Dict

# best model
from sklearn.linear_model import LogisticRegression

class Evaluation:
    """Class to evaluate metrics of best model from model trainer with best params."""
    
    def __init__(self, config: dict, data: DataTransformation | None = None):
        """Initialize ModelEvaluation class.
        
        Args:
            config (dict): Config file consisting of features, targets, file paths.
        """
        self.config = config or load_config()
        self.data = data or DataTransformation(self.config)
        self.scores = []
        
    def eval_best_model(self, y_test: float, y_pred_prob: float) -> Dict[str, float]:
        """Evaluate metrics from best model from model trainer.
        
        Args:
            y_test (float): the actual value.
            y_pred (float): the predicted value.
        """
        try:
            # load in data
            X_train_scaled, X_test_scaled = self.data.split_and_scale_features()
            
            y_train, y_test = self.data.split_targets()
            

            
            model = LogisticRegression(
                C=1,
                max_iter= 1000,
                solver="liblinear"
                )
                
                
            
            model = model.fit(X_train_scaled, y_train)
            
            # save model
            joblib.dump(model, "artifacts/best_model.pkl")
            
   
            y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            # accuracy score and roc/auc score
            roc = roc_auc_score(y_test, y_pred_prob)
    
            
            self.scores.append({"roc/auc score": roc})
            
            return self.scores
        except Exception as e:
            logger.error(f"Could not get scores: {e}")
        return None
