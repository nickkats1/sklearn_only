from src.model import model
import src.common.utils as tools
import src.data.dataio as dataio



def train(config):
    filepath = config["data_preprocessed"] + "train.csv"
    [X,y] = dataio.load(filepath)
    
    Model = model.Model()
    Model.train(X,y)
    
    tools.dump_jobs(config["model_path"],Model)



if __name__ == "__main__":
    config = tools.load_config()
    train(config)