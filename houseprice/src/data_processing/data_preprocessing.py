import src.common.utils as tools
import src.data_processing.data_ingestion as dataio
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.common.utils import load_config

def standardize_fit(X):
    scaler = StandardScaler()
    return scaler.fit(X)


def standardize_transform(X,scaler):
    return scaler.transform(X)


def split(X,y,test_fraction):
    X_train,X_test,y_train,y_test=  train_test_split(X,y,test_size=test_fraction)
    return [X_train,X_test,y_train,y_test]


def preprocess(config):
    rawdatapath = config["raw_data_file"] + config["package_name"] + ".csv"
    [X,y] = dataio.load(rawdatapath)
    test_fraction = 0.3
    [X_train,X_test,y_train,y_test] = split(X,y,test_fraction)
    savepath = config["datainterim"]
    dataio.save(X_train,y_train,savepath + "train.csv")
    dataio.save(X_test,y_test,savepath + "test.csv")
    
    scaler = standardize_fit(X_train)
    X_train_scaled = standardize_transform(X_train,scaler)
    X_test_scaled = standardize_transform(X_test,scaler)
    
    savepath = config["dataprocesseddirectory"]
    dataio.save(X_train_scaled,y_train,savepath + "train.csv")
    dataio.save(X_test_scaled,y_test,savepath + "test.csv")


if __name__ == "__main__":
    config = tools.load_config()
    preprocess(config)

