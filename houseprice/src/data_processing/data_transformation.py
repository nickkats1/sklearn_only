import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from sklearn.model_selection import train_test_split
from src.config import load_config,load_jobs
from pathlib import Path

class DataTransformation:
    def __init__(self, config):
        self.config = config

    def transform_data(self):
        
        data_raw_path = self.config['data_raw']
        data = pd.read_csv(data_raw_path,delimiter=",")
        data.drop_duplicates(inplace=True)
        scaler = load_jobs(Path("models/scaler.joblib"))
 
        df_train,df_test = train_test_split(data,test_size=self.config['test_size'],random_state=self.config['random_state'])




        df_train_scaled_array = scaler.fit_transform(df_train)
        df_test_scaled_array = scaler.transform(df_test)


        df_train_scaled = pd.DataFrame(df_train_scaled_array)
        df_test_scaled = pd.DataFrame(df_test_scaled_array)


        df_train_scaled.to_csv(self.config['processed_train'], index=False)
        df_test_scaled.to_csv(self.config['processed_test'], index=False)

        return df_train_scaled, df_test_scaled


if __name__ == "__main__":
    config = load_config()
    data_transformation = DataTransformation(config)
    data_transformation.transform_data()




