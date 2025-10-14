from src.data_aquisition.data_ingestion import DataIngestion
from src.data_aquisition.model_trainer import ModelTrainer
from src.data_aquisition.data_transformation import DataTransformation
from helpers.config import load_config

if __name__ == "__main__":
    config = load_config()
    data_ingestion = DataIngestion(config)
    data_ingestion.fetch_data()
    data_ingestion.split()
    
    # data transformation
    data_transformation_config = DataTransformation(config)
    data_transformation_config.standardize_data()
    
    
    #model trainer
    model_trainer_config = ModelTrainer(config)
    model_trainer_config.load_params()
    model_trainer_config.log_into_mlflow()