from sklearn.metrics import roc_auc_score,accuracy_score
import src.common.utils as tools
from src import logger



class Results:
    def __init__(self,y_test,y_pred,y_pred_prob,classes) -> None:
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_prob = y_pred_prob
        self.classes = classes
        self.metrics = {}
        
        
    def get_metrics(self):
        self.metrics["accuracy_score"] = accuracy_score(self.y_test,self.y_pred)
        self.metrics["roc_auc_score"] = roc_auc_score(self.y_test,self.y_pred_prob)
        
    def print_metrics(self):
        for key in self.metrics:
            print(f'{key} =\n {self.metrics[key]}')
            



if __name__ == "__main__":
    config = tools.load_config()
    logger.info(f'Config is : {config}')
    results_path = config["resultsrawpath"]
    Results = tools.load_jobs(results_path)
    Results.get_metrics()
    Results.print_metrics()
    validation_path = config["resultsevaluatedpath"]
    tools.dump_jobs(validation_path,Results)

    

