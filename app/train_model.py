import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression,Lasso,Ridge
import warnings
import statsmodels.api as sm
import joblib

warnings.filterwarnings("ignore")




applications = pd.read_csv("https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/refs/heads/main/final_exam_2021/applications.csv")

print(applications.info())
print(applications.duplicated().sum())
print(applications.dtypes)
print(applications.isnull().sum())
print(applications.nunique())


""" turning homeownership into a dummy variable"""

applications['homeownership'] = [1 if X == "Own" else 0 for X in applications['homeownership']]



plt.figure(figsize=(10,6))
sns.heatmap(applications.corr(),fmt="f",annot=True,cmap="coolwarm")
plt.title("Correlation of application data")
plt.show()



fig,axs = plt.subplots(2,2,figsize=(10,6))
sns.scatterplot(data=applications,ax=axs[0,0],x='purchases',y='income')
axs[0,0].set_title("Purchases of Applicant Compared to Applicants Income")

sns.scatterplot(data=applications,ax=axs[0,1],x='credit_limit',y='income')
axs[0,1].set_title("Credit Limit of Applicant Vs Applicants Income")

sns.scatterplot(data=applications,ax=axs[1,0],x='credit_limit',y='purchases')
axs[1,0].set_title("Credit Limit of Applicant and their Purchases")


sns.lineplot(data=applications,ax=axs[1,1],x='homeownership',y='purchases')
axs[1,1].set_title("Homeownership Status of the Applicant and their income")
plt.tight_layout()
plt.show()


app_df = applications.drop(['app_id','ssn','zip_code'],axis=1)

X0 = app_df.drop('purchases',axis=1)
y0 = app_df['purchases']

OLSapp = sm.OLS(exog=sm.add_constant(X0),endog=y0).fit()
print(f'summary of Applicants dataset : {OLSapp.summary()}')



"""credit dataset"""

credit = pd.read_csv("https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/refs/heads/main/final_exam_2021/credit_bureau.csv")


print(credit.describe())
print(credit.isnull().sum())
print(credit.duplicated().sum())
print(credit.dtypes)
print(credit.nunique())


plt.figure(figsize=(10,6))
sns.heatmap(credit.corr(), annot=True,fmt="f",cmap="Blues")
plt.title("Correlation of Credit Data")
plt.show()





fig,axs = plt.subplots(2,2,figsize=(10,6))
sns.kdeplot(data=credit,ax=axs[0,0],x='num_late',hue='past_def')
axs[0,0].set_title('Histogram of the Number of Late Payments')

sns.lineplot(data=credit,ax=axs[0,1],x='num_late',y='num_bankruptcy')
axs[0,1].set_title("Number of Bankruptcy Vs Number of Late Payments")

sns.regplot(data=credit,ax=axs[1,0],x='past_def',y='fico')
axs[1,0].set_title("Past Default vs Fico Score")

sns.violinplot(data=credit,ax=axs[1,1],x='num_late',y='fico')
axs[1,1].set_title("Number of Late Payments and Fico Score")
plt.tight_layout()
plt.show()




""" combining credit and purchases datasets"""

purch_app_pred = pd.concat([credit,applications],axis=1)
purch_app_pred = purch_app_pred.loc[:,~purch_app_pred.columns.duplicated()].copy()
purch_app_pred.isnull().sum()
purch_app_pred.dropna(inplace=True)
purch_app_pred.duplicated().sum()

print(purch_app_pred.describe())
print(purch_app_pred.dtypes)
print(purch_app_pred.nunique())

plt.figure(figsize=(10,6))
sns.heatmap(purch_app_pred.corr(),fmt="f",annot=True,cmap="Blues")
plt.title("Correlation of features from the combined datasets")
plt.show()




fig,axs = plt.subplots(2,2,figsize=(10,6))
sns.kdeplot(data=purch_app_pred,ax=axs[0,0],x='purchases',hue='past_def')
sns.histplot(data=purch_app_pred,ax=axs[0,1],x='income',hue='num_late')
sns.lineplot(data=purch_app_pred,ax=axs[1,0],x='income',y='fico')
sns.lineplot(data=purch_app_pred,ax=axs[1,1],x='num_late',y='income')
plt.tight_layout()
plt.show()


"""Purcahses as the dependent variable without SSN,app_id, & zip_code"""


X1 = purch_app_pred.drop(['purchases','zip_code','ssn'],axis=1)
y1 = purch_app_pred['purchases']

""" anova table for purchase app pred """

OLSpurchapppred = sm.OLS(exog=sm.add_constant(X1),endog=y1).fit()
print(f'results from the applications dataset and applications dataset: {OLSpurchapppred.summary()}')



"""Loading in Demographic dataset"""

demographic = pd.read_csv("https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/refs/heads/main/final_exam_2021/demographic.csv")

print(demographic.isnull().sum())
print(demographic.duplicated().sum())
print(demographic.dtypes)
print(demographic.nunique())
print(demographic.describe())


plt.figure(figsize=(10,6))
sns.heatmap(demographic.corr(),fmt="f",annot=True,cmap="coolwarm")
plt.title("Correlations of Demographic Dataset")
plt.show()


plt.figure(figsize=(10,6))
sns.histplot(data=demographic,x='avg_income')
plt.title("Histogram of Average Income")
plt.show()




purchase_full = pd.concat([applications,credit,demographic],axis=1)
purchase_full = purchase_full.loc[:,~purchase_full.columns.duplicated()].copy()
print(purchase_full.dtypes)
purchase_full.dropna(inplace=True)
print(purchase_full.describe())
print(purchase_full.dtypes)
print(purchase_full.nunique())




plt.figure(figsize=(10,6))
sns.heatmap(purchase_full.corr(),fmt="f",annot=True,cmap="Blues")
plt.title("Correlation of Features")
plt.show()


""" Dropping SSN,zip code and app_id"""
X2 = purchase_full.drop(['app_id','ssn','zip_code'],axis=1)
y2 = purchase_full['purchases']




"""Creating Utilization Variable"""
utilization = purchase_full['purchases'] / purchase_full['credit_limit']



print(f'max utils: {np.max(utilization)}')
print(f'Minimum Utilization: {np.min(utilization)}')
print(f'Average Utilization: {np.mean(utilization)}')
print(f'Description of Utilization: {utilization}')



fig,axs = plt.subplots(2,2,figsize=(12,6))
sns.boxplot(data=purchase_full,ax=axs[0,0],x='num_bankruptcy',y='income')
sns.histplot(data=purchase_full,ax=axs[0,1],x='fico',hue='past_def')
sns.countplot(data=purchase_full,ax=axs[1,0],x='past_def',hue='homeownership')
sns.scatterplot(data=purchase_full,ax=axs[1,1],x='fico',y='avg_income')
plt.tight_layout()
plt.show()





X3 = purchase_full.drop(['ssn','app_id','zip_code'],axis=1)
y3= utilization




OLSutils = sm.OLS(exog=sm.add_constant(X3), endog=y3).fit()
print(f'ANOVA Table from utils: {OLSutils.summary()}')


fig,axs = plt.subplots(2,2,figsize=(12,6))
sns.boxplot(data=purchase_full,ax=axs[0,0],x='num_bankruptcy',y='income')
sns.histplot(data=purchase_full,ax=axs[0,1],x='fico',hue='past_def')
sns.countplot(data=purchase_full,ax=axs[1,0],x='past_def',hue='homeownership')
sns.scatterplot(data=purchase_full,ax=axs[1,1],x='fico',y='avg_income')
plt.tight_layout()
plt.show()



""" Utils as as function of other 'relevant' variables"""

X3 = purchase_full.drop(['ssn','app_id','zip_code'],axis=1)
y3= utilization




OLSutils = sm.OLS(exog=sm.add_constant(X3), endog=y3).fit()
print(f'ANOVA Table from utils: {OLSutils.summary()}')



'''new variable called 'log-odds utils'''

log_utilization = np.log(utilization) / (utilization - 1)


y4 = log_utilization



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

    }



datasets = {
    "Applications": (X0, y0),
    "Purch App Pred": (X1, y1),
    "Purchase Full": (X2, y2),
    "Utility": (X3, y3),
    "Log-Odds Utils":(X3,y4)
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



plt.figure(figsize=(10,6))
sns.barplot(x=df_results['Cross-Validation Score'],y=df_results['Model'],label="Average R2 Score Using 10-fold Cross-Validation")
plt.title("Model and Average R2 Score Using 10-fold Cross-Validation")
plt.legend()
plt.show()



plt.figure(figsize=(10,6))
sns.barplot(x=df_results['R2 Score'],y=df_results['Model'],label="R2 Score the model",color="green")
plt.title("R2 Score For the Model and Dataset")
plt.legend()
plt.show()


'''GridSearch for Optimal Parameters'''


linearregression_params ={
    'fit_intercept': [True, False],  
    'n_jobs': [None, -1],  
    'positive': [True, False]
}

lasso_params = {
    'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100],

}

ridge_params = {
    'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100],

}

models = {
    "LinearRegression":(LinearRegression(),linearregression_params),
    "Lasso":(Lasso(),lasso_params),
    "ridge":(Ridge(),ridge_params)
    }

best_scores = []

X_train,X_test,y_train,y_test = train_test_split(X3,y4,test_size=.20,random_state=42)

for model_name,(model,params) in models.items():
    model.fit(X_train,y_train)
    grid_search = GridSearchCV(model,params,scoring="r2",cv=5,n_jobs=-1)
    grid_search.fit(X_train,y_train)
    print(f'Best Params for {model_name}: {grid_search.best_params_}')
    print(f'Best Score for {model_name}: {grid_search.best_score_}')
    best_scores.append({"Model":model_name,"Best Score":grid_search.best_score_})
    
best_scores_df = pd.DataFrame(best_scores)
print(best_scores_df)



''' Using the hyper-parameters for the best models'''

linear_regression_model = LinearRegression(copy_X=True,positive=False).fit(X_train,y_train)
lr_pred = linear_regression_model.predict(X_test)
print('R2 Score of the optimized Linear Regression Model\n')
print(r2_score(y_test,pred))

lasso_model = Lasso(0.001).fit(X_train,y_train)
lasso_pred = lasso_model.predict(X_test)
print('Results From Optimal Lasso Parameters\n')
print(r2_score(y_test, lasso_pred))


'''ridge'''
ridge_model = Ridge(alpha=19.9).fit(X_train,y_train)
ridge_pred = ridge_model.predict(X_test)
print('Optimized Ridge Parameters\n')
print(r2_score(y_test, ridge_pred))


''' Using the hyper-parameters for the best models'''

linear_regression_model = LinearRegression(copy_X=True,positive=False).fit(X_train,y_train)
lr_pred = linear_regression_model.predict(X_test)
print('R2 Score of the optimized Linear Regression Model\n')
print(r2_score(y_test,pred))

lasso_model = Lasso(0.001).fit(X_train,y_train)
lasso_pred = lasso_model.predict(X_test)
print('Results From Optimal Lasso Parameters\n')
print(r2_score(y_test, lasso_pred))


'''ridge'''
ridge_model = Ridge(alpha=19.99).fit(X_train,y_train)
ridge_pred = ridge_model.predict(X_test)
print('Optimized Ridge Parameters\n')
print(r2_score(y_test, ridge_pred))






def predict(model,features):

    predictions = model.predict(features)

    return predictions.tolist()

if __name__ == "__main__":

    model = joblib.load("models/ridge.joblib")
    

    features = joblib.load("models/features.joblib")

    predictions = predict(model,features)
    
    print('Predictions:')
    for i, pred in enumerate(predictions,start=1): 
        if i % 10 == 0:
            print(f"Sample {i}: Predicted Utility = {np.round(pred,2)}")


