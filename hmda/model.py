from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import logging

logging.basicConfig(filename='hmda.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')


df = pd.read_csv("data/hmda.txt",delimiter="\t")

logging.info(f'number of null values in the dataframe: {df.isnull().sum()}')
logging.info(f'Duplicated Values in the DataFrame: {df.duplicated().sum()}')
logging.info(f'Unique Values in the DataFrame: {df.nunique()}')
logging.info(f'D-dtypes in the data frame: {df.dtypes}')


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



df.drop(['seq','s3','s9','s14','s16','s18','s19a','s19b','s19c','s19d','s20','s48','s49','s54','dprop'],inplace=True,axis=1)

selected_features = ['occupancy','race','sex','income','married','credit_history','di_ratio',
            'pmi_denied','unverifiable','pmi_sought',"vr"]

X = df[selected_features]
y = df['approved']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.20,random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




model = LogisticRegression(C=0.1,max_iter=1000,penalty="l2",solver="liblinear").fit(X_train_scaled,y_train)

pred = model.predict_proba(X_test_scaled)[:,1]
logging.info(f'ROC/AUC Score from best model: {roc_auc_score(y_test,pred)}')


joblib.dump(model,"models/lr_model.joblib")
