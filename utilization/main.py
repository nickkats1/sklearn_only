# data
from src.data.data_ingestion import DataIngestion
from src.data.feature_engineering import FeatureEngineering
from src.data.data_transformation import DataTransformation

# model trainer
from src.models.model_trainer import ModelTrainer

# evaluate and predict
from src.results.evaluation import Evaluation
from src.results.predict import Predict

# config
from helpers.config import load_config


if __name__ == "__main__":
    config = load_config()
    
    # data/
    data = DataIngestion(config).get_data()
    print(data.head(10))
    
    # Feature Engineering
    data = FeatureEngineering(config).select_features()
    
    # DataTransformation
    X_train_scaled, X_test_scaled = DataTransformation(config).split_transform_features()
    y_train, y_test = DataTransformation(config).split_targets()
    
    # Model Trainer
    trainer = ModelTrainer(config).train_models()
    trainer
    
    # Evaluate
    evaluate = Evaluation(config).evaluate_best_model(
        y_test=[0,1,2,3,4,5,6,7,8,9,10,11,12],
        y_pred=[0,1,2,3,4,5,6,7,8,9,10,11,12]
    )
    evaluate
    
    # predict
    predict = Predict(config)
    pred = predict.predict(features=[1,2,3,4,5,6,7,8,9,10,11,12, 13])
    pred