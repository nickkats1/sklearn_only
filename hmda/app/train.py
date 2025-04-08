import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import warnings
import statsmodels.api as sm
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")



df = pd.read_csv("hmda.txt",delimiter="\t")

print(df.isnull().sum())
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
print(df.nunique())
print(df.dtypes)


fig,axs = plt.subplots(2,2,figsize=(10,6))
sns.countplot(data=df,ax=axs[0,0],x='s23a')
axs[0,0].set_title("Applicants Maritial Status")

sns.countplot(data=df,ax=axs[0,1],x='s11')
axs[0,1].set_title("1 == Suffolk County ; 0 == Other County")

sns.histplot(data=df,ax=axs[1,0],x='s30a')
axs[1,0].set_title("Base Monthly Income of Applicant")

sns.boxplot(data=df,ax=axs[1,1],x='s45')
axs[1,1].set_title("debt-to-expense ratio")
plt.show()



fig,axs = plt.subplots(2,2,figsize=(10,6))

sns.boxplot(data=df,ax=axs[0,0],x='s31a',hue="s27a")
axs[0,0].set_title("Base Monthly Income of Applicant vs Employment Status")

sns.countplot(data=df,ax=axs[0,1],x='s52',hue='s56')
axs[0,1].set_title("Private Mortgage Insurance Sought vs Unverifiable Information")

sns.countplot(data=df,ax=axs[1,0],x='s42',hue='s53')
axs[1,0].set_title("Credit History Vs Private Mortgage Insurance Sougt")

sns.pointplot(data=df,ax=axs[1,1],x="s42",y="s44")
axs[1,1].set_title("Credit History Mortgage Payements vs Credit History Public Records")
plt.show()



df.rename(columns={'s5':'occupancy','s7':'approved','s11':'county','s13':'race',
                   's15':'sex','s17':'income','s23a':'married','s27a':'self_employed',
                   's33':'purchase_price','s34':'other_financing','s35':'liquid_assets',
                   's40':'credit_history','s42':'chmp','s43':'chcp','s44':'chpr',
                   's45':'debt_to_expense','s46':'di_ratio','s50':'appraisal',
                   's53':'pmi_denied','netw':'net_worth','uria':'unemployment',
                   'school':'education','s56':'unverifiable',
                   's52':'pmi_sought'},inplace=True)



df['approved'] = [1 if X == 3 else 0 for X in df['approved']]
df['race'] = [0 if X == 3 else 1 for X in df['race']]
df['married'] = [1 if X == 'M' else 0 for X in df['married']]
df['sex'] = [1 if X == 1 else 0 for X in df['sex']]
df['credit_history'] = [1 if X == 1 else 0 for X in df['credit_history']]
df['s4'].value_counts()


df.drop(['seq','s3','s9','s14','s16','s18','s19a','s19b','s19c','s19d','s20','s48','s49','s54','dprop'],inplace=True,axis=1)








X_anova = df.drop('approved',axis=1)
y_anova = df['approved']

olsmodel = sm.OLS(exog=sm.add_constant(X_anova),endog=y_anova).fit()
print(olsmodel.summary())


"""Most relevant features from the anova table"""

selected_features = ['occupancy','race','sex','income','married','credit_history','di_ratio',
            'pmi_denied','unverifiable','pmi_sought',"vr"]



""" using X and Y based on the most relevant features from the GLM anova table"""


X = df[selected_features]
y = df['approved']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)



def evaluate(y_test,pred,pred_prob,model_name,cv_scores):


    result = {
        "Model": model_name,
        "Accuracy": acc,
        "RocAucScore": roc,
        "CVscores": cv_scores.mean()
    }

    return result
    


dict_list = []

models = {
    "LogisticRegregression":LogisticRegression(),
    "BaggingClassifier":BaggingClassifier(),
    "RandomForestClassifier":RandomForestClassifier(),
    "GradientBoostingClassifier":GradientBoostingClassifier(),
    "KNN":KNeighborsClassifier(),
    "DecisionTreeClassifier":DecisionTreeClassifier(),
    "svc":SVC(probability=True),
    "xgb":XGBClassifier(objective="binary:logistic")
    }




    
    


for model_name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    pred_prob = model.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, pred)
    roc = roc_auc_score(y_test, pred_prob)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    evaluation_result = evaluate(y_test, pred, pred_prob, model_name, cv_scores)
    dict_list.append(evaluation_result)



df_results = pd.DataFrame(dict_list)


print(df_results)



def plot_roc_curve(models,X_test,y_test):
    plt.figure(figsize=(12,6))
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        pred_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, pred_prob)
        plt.plot(fpr, tpr, label=model_name)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()



plot_roc_curve(models, X_test, y_test)



lr_params = {
    'C': [0.001, 0.01, 0.1,1,10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'], 
    'max_iter': [1000, 5000, 10000]
}



gradient_boosting_params = {
    'n_estimators': [50,100,200],
    'learning_rate': [1,0.5,0.25,0.1,0.05,0.01],
    'max_depth': [3,4,5],
    'min_samples_split': [2,5,10],
}

random_forest_params = {
    'n_estimators': [50,100,200],
    'max_depth': [None,10,20],
    'min_samples_leaf':[1,2,4],
    'max_features': ['sqrt','log2',None],
    "criterion":["gini", "entropy"]
}



svc_params = {
    'C': [0.1,1,10,100,1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf']
}

bagging_classifier_params = {
    'n_estimators': [50,100,200],
    'max_samples' : [1.0,0.8,0.6],
    'max_features': [1.0,0.8,0.6]
}


knn_params = {
    'n_neighbors' : [5,7,9,11,13,15],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']
               }



xgb_params = {
        'min_child_weight': [1,5,10],
        'gamma': [0.5,1,1.5,2,5],
        'subsample': [0.6,0.8,1.0],
        'colsample_bytree': [0.6,0.8,1.0],
        'max_depth': [3,4,5]
        }
    



models = {
    "LogisticRegression": (LogisticRegression(), lr_params),
    "RandomForestClassifier": (RandomForestClassifier(), random_forest_params),
    "GradientBoostingClassifier": (GradientBoostingClassifier(), gradient_boosting_params),
    "SVC": (SVC(probability=True), svc_params),
    "BaggingClassifier": (BaggingClassifier(), bagging_classifier_params),
    "KnnearestNeighnors":(KNeighborsClassifier(),knn_params),
    "xgboostingclassifier":(XGBClassifier(objective="binary:logistic"),xgb_params)
}


best_score = []


for model_name,(model,params) in models.items():
    print(f'GridSearch For {model_name}:')
    clf = GridSearchCV(model, params,cv=4,scoring="roc_auc",n_jobs=-1).fit(X_train,y_train)
    print(f'Best Parameters for {model_name} : {clf.best_params_}')
    print(f'Best Average RocAuc Scores for: {model_name} : {clf.best_score_ * 100:.2f}%')
    best_score.append({"Model":model_name,"RocScore":clf.best_score_})

best_scores_df = pd.DataFrame(best_score)

print(best_scores_df.head())



gradient_boosting= GradientBoostingClassifier(learning_rate=0.05,max_depth=3,min_samples_split=5,n_estimators=100)
gradient_boosting.fit(X_train,y_train)
gradient_boosting_pred_prob = gradient_boosting.predict_proba(X_test)[:,1]
print('Results From Optimal Logistic Regression model\n')
print(round(roc_auc_score(y_test, gradient_boosting_pred_prob)*100,2))



features = X_train




joblib.dump(features,"models/features.joblib")


joblib.dump(gradient_boosting,"models/gbr_model.joblib")


features = joblib.load("models/features.joblib")
model = joblib.load("models/gbr_model.joblib")

def predict(model,features):

    predictions = model.predict(features)
    pred_probabilities = model.predict_proba(features)[:,1]
    
    

    data = []
    for pred,pred_prob in zip(predictions,pred_probabilities):
        data.append([pred,pred_prob])
    
    return data,pred_probabilities





if __name__ == "__main__":
    
    model = joblib.load("models/gbr_model.joblib")
    features = joblib.load("models/features.joblib")
    predictions, pred_probabilities = predict(model, features)
    ''' 1 == Approved; 0 == "denied'''
    print('Predictions:')
    for i, (pred, prob) in enumerate(predictions):
        if (i+1) % 10== 0:
            print(f"Sample {i+1}: Prediction = {pred}, Probability of being approved/denied = {prob:.2f}")

    print("Predicted Probabilities:")
    print(pred_probabilities)



