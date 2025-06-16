from src.model import model
from src.common.utils import load_config,dump_jobs
import src.data.datio as dataio



def train(config):
    filepath = config["dataprocessed"] + "train.csv"
    [X,y] = dataio.load(filepath)
    
    Model = model.Model()
    Model.train(X,y)
    
    dump_jobs(config["model_path"],Model)



if __name__ == "__main__":
    config = load_config()
    train(config)