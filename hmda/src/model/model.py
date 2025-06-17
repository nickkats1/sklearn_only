from sklearn.linear_model import LogisticRegression



class Model:
    def __init__(self) -> None:
        self.model = []
        self.initialize()
        
    def initialize(self):
        self.model = LogisticRegression()
        
    def train(self,X,y):
        self.model.fit(X,y)
        
    def predict(self,X):
        prediction = self.model.predict(X)
        classes = self.model.classes_
        return [prediction,classes]
    
    def predict_proba(self,X):
        prediction = self.model.predict_proba(X)
        classes = self.model.classes_
        return [prediction,classes]
    





