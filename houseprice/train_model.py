import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import StandardScaler
import warnings
import statsmodels.api as sm
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor,BaggingRegressor
import pickle
import logging

logging.basicConfig(filename="house.log",level=logging.INFO,format='%(asctime)s:%(levelname)s:%(message)s')


warnings.filterwarnings("ignore")

df = pd.read_csv("data/dataset.csv",delimiter=',')
logging.info(f'Null Values: {df.isnull().sum()}')
logging.info(f'Duplicated Values: {df.duplicated().sum()}')
logging.info(f'Unique Values: {df.nunique()}')
logging.info(f'Type of Data in DataFrame: {df.dtypes}')

df.drop_duplicates(inplace=True)


X = df.drop('price',axis=1)
y = df['price']

logging.info(X.dtypes)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.30,random_state=1)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = GradientBoostingRegressor(n_estimators=200,min_samples_split=10,max_depth=5,learning_rate=0.2)
model.fit(X_train_scaled,y_train)
pred = model.predict(X_test_scaled)
logging.info(f'R2 Score For Gradient Boosting Classifier: {r2_score(y_test,pred)}')

with open("models/gbr.pkl","wb") as f:
    pickle.dump(model,f)


with open("models/scaler.pkl","wb") as f:
    pickle.dump(scaler,f)














