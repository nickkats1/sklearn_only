from src.evaluate import evaluate
import src.data.dataio as dataio
import src.common.utils as tools



def predict(config):
    filepath = config["data_preprocessed"] + "test.csv"
    [X,y] = dataio.load(filepath)
    modelpath = config["model_path"]
    Model = tools.load_jobs(modelpath)
    
    [yhat,classes] = Model.predict(X)
    Result = evaluate.Results(y,yhat,classes)
    resultspath = config["resultsrawpath"]
    tools.dump_jobs(resultspath,Result)
    




if __name__ == "__main__":
    config = tools.load_config()
    predict(config)