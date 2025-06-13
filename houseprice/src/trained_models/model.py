from sklearn.ensemble import GradientBoostingRegressor
import src.common.utils as tools


class Model:
    def __init__(self) -> None:
        self.model = []
        self.initialize()
        
    def initialize(self):
        self.model = GradientBoostingRegressor()
        
        
    def train(self,X,y):
        self.model.fit(X,y)
        
    def predict(self,X):
        pred = self.model.predict(X)
        classes = self.model.classes_
        return [pred,classes]
    


