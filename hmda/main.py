# data
from src.data.data_ingestion import DataIngestion
from src.data.feature_engineering import FeatureEngineering
from src.data.data_transformation import DataTransformation

# model trainer
from src.models.model_trainer import ModelTrainer

from src.results.evaluation import Evaluation
from src.results.predict import Predict

# config
from helpers.config import load_config


def main():
    """Main 'hmda' script"""
    # config
    config = load_config()
    
    data = DataIngestion(config).get_data()
    
    # feature selection
    data = FeatureEngineering(config).select_features()
    
    # data transformation
    X_train_scaled, X_test_scaled = DataTransformation(config).split_and_scale_features()
    y_train, y_test = DataTransformation(config).split_targets()
    
    # model trainer
    trainer = ModelTrainer(config)
    trainer.get_best_model()
    
    # Evaluate
    
    y_pred = [0,1,2,3,4,5,6,7]
    y_pred_prob = [0,1,0,1,1,1,0,1]
    y_test = [0,1,2,3,4,5,6,7]
    eval = Evaluation(config = load_config()).eval_best_model(y_test, y_pred, y_pred_prob)
    eval
    

if __name__ == "__main__":
    main()
    