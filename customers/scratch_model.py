import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error


df = pd.read_csv("data/raw/customer.csv",delimiter=",")
df.head()

scaler = StandardScaler()

X = df.drop("log_odds_utils",axis=1)
y = df["log_odds_utils"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestRegressor(max_depth=20,min_samples_leaf=1,min_samples_split=2,n_estimators=100)
model.fit(X_train_scaled,y_train)
pred = model.predict(X_test_scaled)

import joblib
from sklearn.pipeline import Pipeline

pipe = Pipeline(steps=[("scaler",scaler),("model",model)])
pipe.fit(X_train,y_train)
joblib.dump(pipe,"models/model.joblib")

model = joblib.load("models/model.joblib")
pred = model.predict(X_test)
print(r2_score(y_test,pred))