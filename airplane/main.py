# data
from src.data.data_ingestion import DataIngestion
from src.data.feature_engineering import FeatureEngineering
from src.data.data_transformation import DataTransformation

# model trainer
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluation import ModelEvaluation
from src.models.predict import Predict


# helpers
from helpers.config import load_config



if __name__ == "__main__":
    config = load_config()
    
    # data ingestion
    data = DataIngestion(config).fetch_raw_data()
    
    # feature selection
    data = FeatureEngineering(config).select_features()
    
    # data transformation
    X_train_scaled, X_test_scaled = DataTransformation(config).split_transform_features()
    y_train, y_test = DataTransformation(config).split_targets()
    
    # model trainer
    ModelTrainer(config).train_models()
    
    # evaluate
    eval = ModelEvaluation(config)
    eval.eval_best_model(
        y_test = [1,2,3,4,5,6,7,8],
        y_pred = [1,2,3,4,5,6,7,8]
    )
    
    # predict
    predict = Predict(config)
    features = [10,12,12,13,14,15,15,63]
    pred = predict.predict(features)
    print(pred)
    