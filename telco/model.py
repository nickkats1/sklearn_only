import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
import warnings
import statsmodels.api as sm
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,confusion_matrix
import pickle
import logging


logging.basicConfig(filename='churn.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')



df = pd.read_csv("data/telecom_churn.csv",delimiter=',')

logging.info(f'number of null values in the dataframe: {df.isnull().sum()}')
logging.info(f'Duplicated Values in the DataFrame: {df.duplicated().sum()}')
logging.info(f'Unique Values in the DataFrame: {df.nunique()}')
logging.info(f'D-dtypes in the data frame: {df.dtypes}')


scaler = StandardScaler()

X = df.drop('Churn',axis=1)
y = df['Churn']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = GradientBoostingClassifier(learning_rate=0.1,max_depth=3,min_samples_split=5,n_estimators=50)
model.fit(X_train,y_train)
pred_prob = model.predict_proba(X_test)[:,1]
logging.info(f'Roc-Auc Score from GradientBoosting Classifier: {roc_auc_score(y_test,pred_prob)}')

with open("models/gbc.pkl","wb") as f:
    pickle.dump(model,f)



with open("models/scaler.pkl","wb") as f:
    pickle.dump(scaler,f)
    


model = pickle.load(open("models/gbc.pkl","rb"))
sc = pickle.load(open("models/scaler.pkl","rb"))

pred = model.predict_proba([[128,1,0,2.70,2,265.1,110,89.0,9.87,6.6]])[:,1]
logging.info(f'Probability of churn: {pred}')









