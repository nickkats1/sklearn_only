from src.data.data_ingestion import DataIngestion
from src.data.data_transformation import DataTransformation
from src.data.model_trainer import ModelTrainer
from helpers.config import load_config


if __name__ == "__main__":
    config = load_config()
    
    # data ingestion
    data_ingestion_config = DataIngestion(config)
    data_ingestion_config.fetch_data()
    data_ingestion_config.split()
    
    
    # data transformation
    
    data_transformation_config = DataTransformation(config)
    data_transformation_config.transform_data()
    
    # model trainer config
    model_trainer_config = ModelTrainer(config)
    model_trainer_config.load_models_params()
    model_trainer_config.log_into_mlflow()