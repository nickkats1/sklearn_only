import src.evaluate.evaluate as evaluate
import src.data.datio as dataio
from src.common.utils import load_config,load_jobs,dump_jobs


STAGE = "Predict"

def predict(config):
    filepath = config["dataprocessed"] + "test.csv"
    [X,y] = dataio.load(filepath)
    modelpath = config["model_path"]
    Model =load_jobs(modelpath)
    
    [yhat,classes] = Model.predict(X)
    Result = evaluate.Results(y,yhat,classes)
    resultspath = config["resultsrawpath"]
    dump_jobs(resultspath,Result)
    




if __name__ == "__main__":
    config = load_config()
    predict(config)