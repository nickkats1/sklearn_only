import yaml
import joblib

def load_config():
    with open('config.yaml') as p:
        config = yaml.safe_load(p)
    return config

def dump_jobs(path,variable):
    with open(path, 'wb') as handle:
        joblib.dump(variable, handle)



def load_jobs(path):
    with open(path, 'rb') as handle:
        loaded = joblib.load(handle)
    return loaded