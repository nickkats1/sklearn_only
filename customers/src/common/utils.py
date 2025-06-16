import yaml
import joblib



def load_config():
    with open("config.yaml") as p:
        cfg_path = yaml.safe_load(p)
        return cfg_path





def dump_jobs(path,variables):
    with open(path,"wb") as handle:
        joblib.dump(variables,handle)


def load_jobs(path):
    with open(path,"rb") as handle:
        loaded_jobs = joblib.load(handle)
        return loaded_jobs