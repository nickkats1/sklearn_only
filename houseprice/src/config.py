import yaml
import joblib
import os
from src.logger import logger
from typing import Dict






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


def load_params(params_path = "params.yaml") -> Dict:
    """ Load params.yaml"""
    try:
        with open(params_path,"r") as r:
            param_config = yaml.safe_load(r)
            return param_config
    except yaml.YAMLError as exe:
        logger.error(f'Could not parse Yaml file: {e}')
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



if __name__ == "__main__":
    config = load_config()
    params_config = load_params()







    
    



