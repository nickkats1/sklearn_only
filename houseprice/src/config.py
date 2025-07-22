import yaml
import joblib
from urllib.request import urlretrieve
import os
from pathlib import Path
from src.logger import logger
import requests
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import GradientBoostingRegressor,BaggingRegressor
from sklearn.metrics import r2_score,mean_squared_error








def load_config(config_path="config.yaml"):
    """Loads a YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.info(f"Error: Configuration file '{config_path}' not found.")
        return None
    except yaml.YAMLError as exc:
        logger.error(f"Error parsing YAML file '{config_path}':")
        if hasattr(exc, 'problem_mark'):
            logger.error(f"  {exc.problem_mark}")
            logger.error(f"  {exc.problem}")
        return None








def save_jobs(file_obj, obj):
    try:
        dir_path = os.path.dirname(file_obj)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_obj, "wb") as file_obj:
            joblib.dump(obj, file_obj)

    except Exception as e:
        raise e


            



def load_jobs(file_path):
    '''
    loading joblib files
    '''
    try:
        with open(file_path, 'rb') as file_obj:
            return joblib.load(file_obj)
        
    except Exception as e:
        raise e








    
    



