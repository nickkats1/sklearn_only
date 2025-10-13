import yaml
import pickle
from helpers.logger import logger 
from typing import Dict
import os





def load_config(config_path = "config.yaml") -> Dict:
    """Loads a YAML configuration file."""
    try:
        with open(config_path, 'r') as p:
            config = yaml.safe_load(p)
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








def save_file(file_obj, obj):
    try:
        dir_path = os.path.dirname(file_obj)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_obj, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise e


            



def load_file(file_path):
    '''
    loading joblib files
    '''
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise e


