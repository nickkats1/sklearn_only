from sklearn.metrics import mean_squared_error,r2_score
from src.common.utils import load_jobs,load_config,dump_jobs



class Results:
    def __init__(self,y_test,y_pred,classes) -> None:
        self.y_test = y_test
        self.y_pred = y_pred
        self.classes = classes
        self.metrics = {}
        
    def get_metrics(self):
        self.metrics["mean_squared_error"] = mean_squared_error(self.y_test,self.y_pred)
        self.metrics['r2_score'] = r2_score(self.y_test,self.y_pred)
        
    def print_metrics(self):
        for k in self.metrics:
            print(f"{k}=\n {self.metrics[k]}")
            


if __name__ == "__main__":
    config= load_config()
    
    resultspath = config["resultsrawpath"]
    Results = load_config(resultspath)
    Results.get_metrics()
    Results.print_metrics()

    validationpath = config["resultsevaluatedpath"]
    dump_jobs(validationpath,Results)