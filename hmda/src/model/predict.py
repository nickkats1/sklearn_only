from src.evaluate import evaluate
import src.common.utils as tools
import src.data.datio as dataio


def predict(config):
    filepath = config["preprocessed_path"] + "test.csv"
    [X,y] = dataio.load(filepath)
    
    modelpath = config["model_path"]
    Model = tools.load_jobs(modelpath)
    [yhat,classes] = Model.predict(X)
    
    Result = evaluate.Results(y,yhat,classes)
    Results_Path = config["raw_path"]
    tools.dump_jobs(Results_Path,Result)


if __name__ == "__main__":
    config = tools.load_config()
    predict(config)
    


