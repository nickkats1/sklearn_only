import src.evaluate.evaluate as evaluate
import src.data_processing.data_ingestion as dataio
import src.common.utils as tools




def predict(config):
    filepath = config["dataprocesseddirectory"] + "test.csv"
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
