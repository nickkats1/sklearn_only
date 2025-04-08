import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import GradientBoostingRegressor,BaggingRegressor
import joblib
from sklearn.metrics import r2_score,mean_squared_error
import statsmodels.api as sm
import numpy as np

warnings.filterwarnings("ignore")

""" Airplane Sales"""

airplane_sales = pd.read_csv("https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/refs/heads/main/final_exam_2020/airplane_sales.csv")

""" descriptive stats"""


print(f'Airplane Sales null values\n: {airplane_sales.isnull().sum()}')
print(f'Airplane Sales duplicated values\n: {airplane_sales.duplicated().sum()}')
print(f'Airplane Sales Data Types\n: {airplane_sales.dtypes}')
print(f'Airplane Sales Described\n: {airplane_sales.describe()}')
print(f'Airplane Sales Unique Values\n: {airplane_sales.nunique()}')




""" heatmap to show the correlation of the variables"""

plt.figure(figsize=(10,6))
sns.heatmap(airplane_sales.corr(),fmt="f",annot=True,cmap="coolwarm")
plt.tight_layout()
plt.show()

""" Price as a function of age"""

X0 = airplane_sales[['age']]
y0 = airplane_sales['price']

OLSmodel = sm.OLS(exog=sm.add_constant(X0), endog=y0).fit()
print(f'Anova Results from Airplan Sales: {OLSmodel.summary()}')







"""loading in airplan specs dataset"""


airplane_specs = pd.read_csv("https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/refs/heads/main/final_exam_2020/airplane_specs.csv")


print(f'Airplane Specs null values\n: {airplane_specs.isnull().sum()}')
print(f'Airplane Specs duplicated values\n: {airplane_specs.duplicated().sum()}')
print(f'Airplane Specs Data Types\n: {airplane_specs.dtypes}')
print(f'Airplane Specs Described\n: {airplane_specs.describe()}')
print(f'Airplane Specs Unique Values\n: {airplane_specs.nunique()}')

airplane_specs.rename(columns={"pass":"pas"},inplace=True)

""" heatmap for the correlation of the airplane specs dataset"""

plt.figure(figsize=(10,6))
sns.heatmap(airplane_specs.corr(), fmt="f",annot=True,cmap="Blues")
plt.show()


fig,axs = plt.subplots(2,2,figsize=(10,6))
sns.barplot(data=airplane_specs,ax=axs[0,0],x='tdrag',y='wtop')
sns.lineplot(data=airplane_specs,ax=axs[0,1],x='pas',y='0Sale_ID')
sns.kdeplot(data=airplane_specs,ax=axs[1,0],x='0Sale_ID',hue="tdrag")
sns.countplot(data=airplane_specs,ax=axs[1,1],x='pas',hue='fixgear')
plt.tight_layout()
plt.show()

""" combining airplane specs and airplane sales together to form:
    'airplane_sales_specs'
"""

airplane_sales_specs = pd.concat([airplane_specs,airplane_sales],axis=1)
airplane_sales_specs = airplane_sales_specs.loc[:,~airplane_sales_specs.columns.duplicated()].copy()

print(f'Airplane Sales Specs null values\n: {airplane_sales_specs.isnull().sum()}')
print(f'Airplane Sales Specs duplicated values\n: {airplane_sales_specs.duplicated().sum()}')
print(f'Airplane Sales Specs Data Types\n: {airplane_sales_specs.dtypes}')
print(f'Airplane Sales Specs Described\n: {airplane_sales_specs.describe()}')
print(f'Airplane Sales Specs Unique Values\n: {airplane_sales_specs.nunique()}')


plt.figure(figsize=(10,6))
sns.heatmap(airplane_sales_specs.corr(), fmt="f",annot=True,cmap="coolwarm")
plt.show()

""" predicting price as a function of age,pass,wtop,fixgear and tdrag"""


X1 = airplane_sales_specs[['age','pas','wtop','fixgear','tdrag']]
y1 = airplane_sales_specs['price']



olsairplanesalespecs = sm.OLS(exog=sm.add_constant(X1),endog=y1).fit()
print('Results from airplane sales specs anova table\n')
print(olsairplanesalespecs.summary())








"""Loading in airplane perf"""

airplane_perf = pd.read_csv("https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/refs/heads/main/final_exam_2020/airplane_perf.csv")

print(f'Airplane Perf null values\n: {airplane_perf.isnull().sum()}')
print(f'Airplane Perf duplicated values\n: {airplane_perf.duplicated().sum()}')
print(f'Airplane Perf Data Types\n: {airplane_perf.dtypes}')
print(f'Airplane Perf Described\n: {airplane_perf.describe()}')
print(f'Airplane Perf Unique Values\n: {airplane_perf.nunique()}')


""" heatmap for airplane perf"""
plt.figure(figsize=(10,6))
sns.heatmap(airplane_perf.corr(),fmt="f",annot=True,cmap="Blues")
plt.show()



""" making a dataframe named 'airplane_full' by combining all datasets"""

airplane_full = pd.concat([airplane_sales,airplane_specs,airplane_perf],axis=1)
airplane_full = airplane_full.loc[:,~airplane_full.columns.duplicated()].copy()

print(f'Airplane Full null values\n: {airplane_full.isnull().sum()}')
print(f'Airplane Full duplicated values\n: {airplane_full.duplicated().sum()}')
print(f'Airplane Full Data Types\n: {airplane_full.dtypes}')
print(f'Airplane Full Described\n: {airplane_full.describe()}')
print(f'Airplane Full Unique Values\n: {airplane_full.nunique()}')


plt.figure(figsize=(10,6))
sns.heatmap(airplane_full.corr(),fmt="f",annot=True,cmap="coolwarm")
plt.title("Correlation of all features from all datasets")
plt.show()


""" making a regression model to evaluate:
    age,pass,wtop,fixgear,tdrag,horse,fuel,ceiling,cruise as a function of price"""

X2 = airplane_full[["age","pas","wtop","fixgear","tdrag","horse","fuel","ceiling","cruise"]]
y2 = airplane_full['price']


olsfullmodel = sm.OLS(exog=sm.add_constant(X2), endog=y2).fit()
print('Anova from all datasets\n')
print(olsfullmodel.summary())


"""creating log variables for price,age,horse,fuel,ceiling,cruise"""




log_price = np.log(airplane_full['price'])
log_age = np.log(airplane_full['age'])
log_horse = np.log(airplane_full['horse'])
log_fuel = np.log(airplane_full['fuel'])
log_ceiling = np.log(airplane_full['ceiling'])
log_cruise = np.log(airplane_full['cruise'])




X3 = pd.DataFrame({
    "log_age": log_age,
    "pass": airplane_full["pas"],
    "wtop": airplane_full["wtop"],
    "fixgear": airplane_full["fixgear"],
    "tdrag": airplane_full["tdrag"],
    "log_horse": log_horse,
    "log_fuel": log_fuel,
    "log_ceiling": log_ceiling
})

y3 = log_price





""" going to do all of the linear regression parts now for the sake of modularization"""


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
    "baggingregressor":BaggingRegressor(),
    "gradientboostingregressor":GradientBoostingRegressor()
    }






datasets = {
    "Airplane Sales": (X0, y0),
    "Airplane Sales Specs": (X1, y1),
    "Airplane Full": (X2, y2),
    "Log Transformed": (X3, y3)
}

results = []

for dataset_name,(X,y) in datasets.items():
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        r2 = r2_score(y_test, pred)
        mse = mean_squared_error(y_test, pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring='r2')
        results.append(evaluate(y_test, pred, r2, mse, cv_scores, f"{model_name} on {dataset_name}"))

df_results = pd.DataFrame(results)
pd.set_option('display.max_rows', 30)
print(df_results.sort_values(by="R2 Score",ascending=False))


plt.figure(figsize=(10,5))
sns.barplot(x=df_results['Cross-Validation Score'],y=df_results['Model'],label="Average R2 Score Using 10-fold Cross-Validation",color="red")
plt.title("Model and Average R2 Score Using 10-fold Cross-Validation")
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
sns.barplot(x=df_results['R2 Score'],y=df_results['Model'],label="R2 Score for the model",color="blue")
plt.title("R2 Score For the Model and Dataset")
plt.legend()
plt.show()



"""Just Use Grid Search"""


linearregression_params = {
'copy_X': [True,False], 
'fit_intercept': [True,False], 
'n_jobs': [1,5,10,15,None], 
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

X_train,X_test,y_train,y_test = train_test_split(X3,y3,test_size=.20,random_state=42)

for model_name,(model,params) in models.items():    
    model.fit(X_train,y_train)
    grid_search = GridSearchCV(model,params,scoring="neg_mean_squared_error",cv=4,n_jobs=-1)
    grid_search.fit(X_train,y_train)
    print(f'Best Params for {model_name}: {grid_search.best_params_}')
    print(f'Best Score for {model_name}: {grid_search.best_score_}')
    best_scores.append({"Model":model_name,"Best Score":grid_search.best_score_})
    


best_scores_df = pd.DataFrame(best_scores)
print(best_scores_df)


linear_regression_model = LinearRegression(copy_X=True,n_jobs=1,positive=False)
linear_regression_model.fit(X_train,y_train)
y_pred = linear_regression_model.predict(X_test)
print('R2 Score Best Linear Regression Model\n')
print(r2_score(y_test, y_pred))


joblib.dump(linear_regression_model,"models/lr_model.joblib")

ridge_model = Ridge(alpha=.001)
ridge_model.fit(X_train,y_train)
y_pred = ridge_model.predict(X_test)
print('Best Score for Ridge Model\n')
print(r2_score(y_test, y_pred))

lasso_model = Lasso(alpha=0.001).fit(X_train,y_train)
y_pred = lasso_model.predict(X_test)
print("Best Scores Lasso\n")
print(r2_score(y_test, y_pred))





features = X_train


joblib.dump(features,"models/features.joblib")


def predict(model,features):
    model = joblib.load("models/lr_model.joblib")
    features = joblib.load("models/features.joblib")
    predictions = model.predict(features)
    return [[pred] for pred in predictions] 

if __name__ == "__main__":
    predictions = predict(model,features)
    


    print("predictions")
    for i, pred in enumerate(predictions):
        if (i+1) % 1 == 0:
            print(f'Sample {i+1}: pred log price = ${(np.round(pred[0],2))}')
