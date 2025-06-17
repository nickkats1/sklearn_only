from src.trained_models import model
import src.common.utils as tools
import src.data_processing.data_ingestion as dataio



def train(config):
    filepath = config["dataprocesseddirectory"] + "train.csv"
    [X,y] = dataio.load(filepath)
    
    Model = model.Model()
    Model.train(X,y)
    
    tools.dump_jubs(config["model_path"],Model)



if __name__ == "__main__":
    config = tools.load_config()
    train(config)
