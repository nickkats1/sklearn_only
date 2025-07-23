import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from sklearn.model_selection import train_test_split
from src.config import load_config
from pathlib import Path

class DataTransformation:
    def __init__(self, config):
        self.config = config

    def transform_data(self):
        data_raw_path = self.config['data_raw']
        target_column_name = self.config['target']
        feature_column_names = self.config['features']

        data = pd.read_csv(data_raw_path, delimiter=",")
        data.drop_duplicates(inplace=True)


        X = data[feature_column_names]
        y = data[target_column_name]


        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        ) 


        scaler = StandardScaler() 
        X_train_scaled_array = scaler.fit_transform(X_train) 
        X_test_scaled_array = scaler.transform(X_test) 


        X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=feature_column_names)
        X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=feature_column_names)

        X_train_scaled.to_csv(self.config['processed_train'], index=False)
        y_train.to_csv(self.config['processed_test'], index=False)
        X_test_scaled.to_csv(self.config['processed_test'], index=False)
        y_test.to_csv(self.config['target'], index=False)


        return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    config = load_config()
    data_transformation = DataTransformation(config)