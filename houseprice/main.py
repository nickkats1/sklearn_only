# data
from src.data.data_ingestion import DataIngestion
from src.data.data_transformation import DataTransformation

# models
from src.models.model_trainer import ModelTrainer
from src.models.evaluate import Evaluate
from src.models.predict import Predict

# config
from helpers.config import load_config


if __name__ == "__main__":
    config = load_config()
    
    # data ingestion
    
    data = DataIngestion(config).get_data()
    
    # data transformation
    X_train_scaled, X_test_scaled = DataTransformation(config).split_transform_features()
    y_train, y_test = DataTransformation(config).split_targets()
    
    # model trainer
    mt = ModelTrainer(config)
    mt.train_models()
    
    # evaluate best model
    
    eval = Evaluate(config)
    y_test = [0,110,12,3,4,5,6,67,8,1,23]
    y_pred = [110, 23, 3, 4, 6, 7, 8.7, 8, 12, 11]
    
    results = eval.eval_best_model(y_test, y_pred)
    results
    
    # predict
    
    predict = Predict(config)
    
    features = [0,110,12,3,4,5,6,67,8,1,23]
    prediction = predict.predict_pipeline(features)
    print(f"Predicted price: {prediction}")