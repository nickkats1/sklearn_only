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
import joblib


warnings.filterwarnings("ignore")

df = pd.read_csv("../app/dataset.csv",delimiter=',')
print(f'Null Values: {df.isnull().sum()}')
print(f'Duplicated Values: {df.duplicated().sum()}')
print(f'Unique Values: {df.nunique()}')
print(f'Type of Data in DataFrame: {df.dtypes}')


df.drop_duplicates(inplace=True)


"""Desc Stats and heatmap"""


plt.figure(figsize=(20,6))
sns.heatmap(df.corr(),fmt="f",annot=True,cmap="Blues")
plt.title("Correlation of all of the features")
plt.show()

plt.figure(figsize=(16,8))
sns.scatterplot(data=df,x='x_coord',y='price')
plt.title("Price of House Based on X Coordinate")
plt.show()
 
sns.scatterplot(data=df,x='dist_lakes',y='price')
plt.title('Price of House Based on Distance from Lakes')
plt.show()

sns.scatterplot(data=df,x='parcel_size',y='price')
plt.title("Price of House Based on Parcel Size")
plt.show()

sns.scatterplot(data=df,x='home_size',y='price')
plt.title("Price of House Based on Home Size")
plt.show()

sns.lineplot(data=df,x='age',y='price')
plt.title("Price of House Based on Age")
plt.show()


sns.boxplot(data=df,x='year',y='price')
plt.title("Comparison of Year Vs Price of House")
plt.tight_layout()
plt.show()

 
sns.countplot(data=df,x='beds',hue='pool')
plt.title("Comparison of number of bedrooms and if the House has a Pool")
plt.tight_layout()
plt.show()


X = df.drop('price',axis=1)
y = df['price']

olsmodel = sm.OLS(exog=sm.add_constant(X),endog=y).fit()
print(f'anova: {olsmodel.summary()}')



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def evaluate(y_test,pred,model_name,r2,mse,cv_scores):


    result = {
        "Model": model_name,
        "R2": r2,
        "MSE": mse,
        "Cross-val Scores":cv_scores.mean()
    }

    return result


model_dict = []
models = {
    "LinearRegression":LinearRegression(),
    "lasso":Lasso(),
    "ridge":Ridge(),
    "GradientBoostingRegressor":GradientBoostingRegressor(),
    "BaggingRegressor":BaggingRegressor()
}



for model_name,model in models.items():
    model.fit(X_train_scaled,y_train)
    pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test,pred)
    r2 = r2_score(y_test,pred)
    cv_scores = cross_val_score(model, X_train_scaled,y_train,cv=10,scoring="neg_mean_squared_error")
    model_results = evaluate(y_test, pred, model_name,r2,mse,cv_scores)
    model_dict.append(model_results)



df_results = pd.DataFrame(model_dict)
print(df_results.head())



"""HyperParamter tuning to minimize the mean squared errors"""

linearregression_params = {
'copy_X': [True,False], 
'fit_intercept': [True,False], 
'n_jobs': [1000,5000,10000], 
'positive': [True,False]}



lasso_params = {
    'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]
    }


ridge_params = {
    'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]
    }



gradientboosting_params = {
    'n_estimators': [50,100,200],
    'learning_rate': [0.01,0.1,0.2],
    'max_depth': [3,4,5],
    'min_samples_split': [2,5,10]
    }

bagginregression_params = {
    'n_estimators': [50,100,200],
    'max_samples' : [1.0,0.8,0.6],
    'max_features': [1.0,0.8,0.6]
}



models = {
    "LinearRegression":(LinearRegression(),linearregression_params),
    "Lasso":(Lasso(),lasso_params),
    "ridge":(Ridge(),ridge_params),
    "GradientBoostingRegresser":(GradientBoostingRegressor(),gradientboosting_params),
    "BaggingREgressor":(BaggingRegressor(),bagginregression_params),
    }

best_scores = []

for model_name,(model,params) in models.items():
    grid_search = RandomizedSearchCV(model,params,scoring="neg_mean_squared_error",cv=4)
    grid_search.fit(X_train_scaled,y_train)
    print(f'Best Params for {model_name}: {grid_search.best_params_}')
    print(f'Best Score for {model_name}: {grid_search.best_score_}')
    best_scores.append({"Model":model_name,"Neg-Mean-Average-MSE":grid_search.best_score_})
    

best_scores_df = pd.DataFrame(best_scores)
print(best_scores_df.head())


"""Best Model with Optimized hyper-parameters"""


gradient_boosting_regressor = GradientBoostingRegressor(n_estimators=200,min_samples_split=10,max_depth=5,learning_rate=0.2)
gradient_boosting_regressor.fit(X_train_scaled,y_train)
y_pred = gradient_boosting_regressor.predict(X_test_scaled)
print('Best Model With Optimized Hypder Parameters\n')
print(r2_score(y_test, y_pred))






def predict(model,features):

    predictions = model.predict(features)
    return [[pred] for pred in predictions] 

if __name__ == "__main__":

    model = joblib.load("models/gbr.joblib")

    
    features = X_train
    scaler = joblib.load("models/scaler.joblib")
    features_scaled = scaler.fit_transform(features)
    predictions = predict(model,features_scaled)


    print("Predictions:")
    for i, pred in enumerate(predictions):
        if (i+1) % 100 == 0:
            print(f'Sample {i+1}: Prediction of House Price = (${round(pred[0],2)}')
