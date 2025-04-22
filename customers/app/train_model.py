import sqlite3 as sql
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
import logging
import pandas as pd
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
import pickle



pd.set_option('display.max_columns', 30)
con = sql.connect("../data/customers.db")


query = """
SELECT *
FROM sqlite_master
WHERE type='table';
"""
pd.read_sql_query(query, con)


""" query that will obtain data from customers table """


query1 = """
SELECT *
FROM Applications;"""
pd.read_sql_query(query1,con)

purchase_app = pd.read_sql("""SELECT * FROM Applications;""",con)
purchase_app.head()

print(f' Null Values from Purcase App df: {purchase_app.isnull().sum()}')
print(f'dtypes from Purchase APP df: {purchase_app.dtypes}')
print(f'nunique from Purchase App df: {purchase_app.nunique()}')
print(f'Duplicated Values from Purchase App: {purchase_app.duplicated().sum()}')



purchase_app['homeownership'] = purchase_app['homeownership'].map({"Rent":0,"Own":1})

""" Heatmap"""

plt.figure(figsize=(10,6))
sns.heatmap(purchase_app.corr(),annot=True,fmt="f",cmap="Blues")
plt.show()
purchase_app.dtypes

purchase_app.dtypes

fig,axs = plt.subplots(2,2,figsize=(10,6))
sns.scatterplot(x='income',y='purchases',hue="income",ax=axs[0,0],data=purchase_app)
axs[0,0].set_title("Income Compared to Purchase")
sns.boxplot(x='credit_limit',y='income',ax=axs[0,1],data=purchase_app)
axs[0,1].set_title("Credit Limit Commpared to Income")
sns.histplot(x='purchases',hue='homeownership',ax=axs[1,0],data=purchase_app)
axs[1,0].set_title("Histogram of Purchase")
sns.histplot(x='income',hue='homeownership',data=purchase_app)
plt.legend()
plt.show()


""" Ok, a OLS table"""

X0 = purchase_app.drop(['app_id','ssn','zip_code','purchases'],axis=1)
y0 = purchase_app['purchases']

model1 = sm.OLS(exog=sm.add_constant(X0), endog=y0).fit()
print(f'Results from Anova Table: {model1.summary()}')

""" Train/Test split later this is predictable"""

""" joining applications and credit """


purch_app_bureau = pd.read_sql(
    """
    SELECT a.*,c.*
    FROM Applications a
    JOIN CreditBureau c on a.zip_code = c.zip_code;""",con)


print(f'Null Values From purch_app_bureau: {purch_app_bureau.isnull().sum()}')
print(f'Duplicated Values From purch_app_bureau: {purch_app_bureau.duplicated().sum()}')
print(f'dtypes from purch_app_bureau: {purch_app_bureau.dtypes}')
print(f'Unique Values from purch_app_bureau: {purch_app_bureau.nunique()}')
print(f'Description of the Values in purch_app_bureau: {purch_app_bureau.describe()}')

purch_app_bureau['homeownership'] = purch_app_bureau['homeownership'].map({"Rent":0,"Own":1})
X1 = purch_app_bureau.drop(['ssn','app_id','purchases','zip_code'], axis=1)
y1 = purch_app_bureau['purchases']

model2 = sm.OLS(exog=sm.add_constant(X1), endog=y1).fit()
print(model2.summary())



""" Joining all tables"""

purchase_full = pd.read_sql("""
SELECT 
    a.*, 
    c.*, 
    d.*
FROM Applications a
JOIN CreditBureau c ON a.zip_code = c.zip_code
JOIN Demographic d ON a.zip_code = d.zip_code;
""", con)

purchase_full = purchase_full.loc[:,~purchase_full.columns.duplicated()].copy()
print(f'Null Values From Purchase Full: {purchase_full.isnull().sum()}')
print(f'Duplicated Values From Purchase Full: {purchase_full.duplicated().sum()}')
print(f'dtypes from Purchase full: {purchase_full.dtypes}')
print(f'Unique Values from Purchase Full: {purchase_full.nunique()}')
print(f'Description of the Values in Purchase Full: {purchase_full.describe()}')

con.close()

fig,axs = plt.subplots(2,2,figsize=(12,6))
sns.boxplot(data=purchase_full,ax=axs[0,0],x='num_bankruptcy',y='income')
sns.histplot(data=purchase_full,ax=axs[0,1],x='fico',hue='past_def')
sns.countplot(data=purchase_full,ax=axs[1,0],x='past_def',hue='homeownership')
sns.scatterplot(data=purchase_full,ax=axs[1,1],x='fico',y='avg_income')
plt.tight_layout()
plt.show()

purchase_full['homeownership'] = purchase_full['homeownership'].map({"Rent":0,"Own":1})

X2 = purchase_full.drop(['app_id','ssn','purchases','zip_code'],axis=1)
y2 = purchase_full['purchases']

reg_model_full = sm.OLS(exog=sm.add_constant(X2), endog=y2).fit()
print(reg_model_full.summary())


utilization = purchase_full['purchases'] / purchase_full['credit_limit']
y3 = utilization

log_odds_utils = np.log(utilization) / (utilization - 1)
X3 = purchase_full.drop(['app_id','ssn','zip_code'],axis=1)
y4= log_odds_utils

"""None of this is advanced. Build 1 Neural Netowork of any kind first
you cannot give advice to a person who can make GANS, CNNS, fine-tune BERT,
ect.. when you cannot use torch, this is not advanced"""


scaler = StandardScaler()


def evaluate(y_test,pred,r2,mse,cv_scores,model_name):
    results = {
        "Model":model_name,
        "R2 Score":r2,
        "Mean Squared Error":mse,
        "Cross-Validation Score":cv_scores.mean()
        }
    return results



models = {
    "linearregression":LinearRegression(),
    "lasso":Lasso(),
    "ridge":Ridge(),
    "gradientboosting":GradientBoostingRegressor(),
    "randomforest":RandomForestRegressor(),
    "decisiontree":DecisionTreeRegressor(),

    }



datasets = {
    "Applications": (X0, y0),
    "Purch App": (X1, y1),
    "Purchase Full": (X2, y2),
    "Utility": (X3, y3),
    "Log-Odds Utils":(X3,y4)
}



results = []

for dataset_name,(X,y) in datasets.items():
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        
        r2 = r2_score(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring='r2')
        results.append(evaluate(y_test, pred, r2, mse, cv_scores, f"{model_name} on {dataset_name}"))

df_results = pd.DataFrame(results)
print(df_results.sort_values(by="R2 Score",ascending=False))

plt.figure(figsize=(12,8))
sns.barplot(x=df_results['Cross-Validation Score'],y=df_results['Model'],label="Average R2 Score Using 10-fold Cross-Validation")
plt.title("Model and Average R2 Score Using 10-fold Cross-Validation")
plt.legend()
plt.show()


plt.figure(figsize=(10,6))
sns.barplot(x=df_results['R2 Score'],y=df_results['Model'],label="R2 Score the model",color="green")
plt.title("R2 Score For the Model and Dataset")
plt.legend()
plt.show()


"""Grid Search is better than optimizing through out-dated methods,
like all of this honestly"""


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

RandomForestRegressor().get_params(deep=True)
randomforest_params = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

decisiontreeregressor_params = {
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}




models = {
    "LinearRegression":(LinearRegression(),linearregression_params),
    "Lasso":(Lasso(),lasso_params),
    "ridge":(Ridge(),ridge_params),
    "randomforest":(RandomForestRegressor(),randomforest_params),
    "gradientboostingregressor":(GradientBoostingRegressor(),gradientboosting_params),
    "decisiontreeregreesor":(DecisionTreeRegressor(),decisiontreeregressor_params),

    }

best_scores = []

X_train,X_test,y_train,y_test = train_test_split(X3,y4,test_size=.20,random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
for model_name,(model,params) in models.items():
    model.fit(X_train_scaled,y_train)
    grid_search = GridSearchCV(model,params,scoring="neg_mean_squared_error",cv=4,n_jobs=-1)
    grid_search.fit(X_train_scaled,y_train)
    print(f'Best Params for {model_name}: {grid_search.best_params_}')
    print(f'Best Score for {model_name}: {grid_search.best_score_}')
    best_scores.append({"Model":model_name,"Best Score":grid_search.best_score_})
    
best_scores_df = pd.DataFrame(best_scores)
print(best_scores_df)


gradientboostingregressor = GradientBoostingRegressor(learning_rate=0.1,max_depth=5,min_samples_split=10,n_estimators=200)
gradientboostingregressor.fit(X_train_scaled,y_train)
pred = gradientboostingregressor.predict(X_test_scaled)
r2_bestmodel = r2_score(y_test, pred)
print('R2 Score For Best Model\n')
print(r2_score(y_test,pred))

with open("models/gbr.pkl","wb") as f:
    pickle.dump(gradientboostingregressor,f)
    

with open("models/scaler.pkl","wb") as f:
    pickle.dump(scaler,f)









