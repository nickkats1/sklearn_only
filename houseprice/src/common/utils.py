import yaml
import joblib




def load_config():
    with open("config.yaml") as p:
        load_config = yaml.safe_load(p)
        return load_config
    






def dump_jobs(path,variable):
    with open(path,"wb") as handle:
        joblib.dump(variable,handle)




def load_jobs(path):
    with open(path,"wb") as handle:
        loaded_jobs = joblib.load(handle)
        return loaded_jobs
    
