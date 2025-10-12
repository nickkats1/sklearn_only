import pandas as pd
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
from helpers.config import load_config
from helpers.logger import logger

class DataIngestion:
    def __init__(self,config):
        self.config = config
        
        
    def fetch_data(self) -> pd.DataFrame:
        """ Fetch data from url """
        try:
            #url link
            self.url_link = self.config['url_link']
            # raw path for the data ingestion
            
            
            self.raw_path = self.config['data_raw']
            
            # data ingestion
            
            urlretrieve(url=self.url_link,filename=self.raw_path)
            URL_LINK = self.config['url_link']
            # raw path for the data ingestion
            
            
            RAW_PATH = self.config['data_raw']
            
            # data ingestion
            
            urlretrieve(url=URL_LINK,filename=RAW_PATH)

            
        except Exception as e:
            raise e
        
        



    def cleaning(self) -> pd.DataFrame:
        """ Clean features and select relevant variables """
        try:
            data = pd.read_csv(self.config['data_raw'],delimiter=",")
            

            
            # clean features based on variable description
            
            data.rename(
            columns={
                's5':'occupancy',
                's7':'approved',
                's13':'race',
                's15':'sex',
                's17':'income',
                's23a':'married',
                's27a':'self_employed',
                's33':'purchase_price',
                's34':'other_financing',
                's35':'liquid_assets',
                's40':'credit_history',
                's42':'chmp',
                's43':'chcp',
                's44':'chpr',
                's45':'debt_to_expense',
                's46':'di_ratio',
                's50':'appraisal',
                's53':'pmi_denied',
                's56':'unverifiable',
                's52':'pmi_sought'},inplace=True)

            
            
            
            # change data to variable meaning
            
            data['approved'] = [1 if X == 3 else 0 for X in data['approved']]
            data['race'] = [0 if X == 3 else 1 for X in data['race']]
            data['married'] = [1 if X == 'M' else 0 for X in data['married']]
            data['sex'] = [1 if X == 1 else 0 for X in data['sex']]
            data['credit_history'] = [1 if X == 1 else 0 for X in data['credit_history']]
            # drop all unused features
            unused_variables = self.config['unused_variables']
            
            data.drop(unused_variables,inplace=True,axis=1)

            data.drop_duplicates(inplace=True)
            
            return data
        except Exception as e:
            raise e
        
        
    def split(self) -> pd.DataFrame:
        """ split features and target into train/test split """
        try:
            data = self.cleaning()
            # changing variables to actual meaning from paper description
            
            features = self.config['features']
            target = self.config['target']
            features = data[features]
            target = data[target]

            
            df_train,df_test = train_test_split(features,test_size=.20,random_state=42)
            
            # y_train,y_test
            
            
            y_train,y_test = train_test_split(target,test_size=0.20,random_state=42)
            
            # convert train/test split DataFrame to .csv file
            df_train.to_csv(self.config['train_raw'],index=0)
            df_test.to_csv(self.config['test_raw'],index=0)
            print(f"Shape of df_train: {df_train.shape}")
            print(f"Shape of df_test: {df_test.shape}")
            
            y_train.to_csv(self.config['y_train_raw'],index=0)
            y_test.to_csv(self.config['y_test_raw'],index=0)
            
            print(f"Shape of y_train: {y_train.shape}")
            print(f"Shape of y_test: {y_test.shape}")
            
            return df_train,df_test,y_train,y_test
        
        except Exception as e:
            raise e
        

