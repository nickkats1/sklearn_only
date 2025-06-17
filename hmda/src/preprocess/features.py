import src.common.utils as tools
import src.data.datio as dataio
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def standardize_fit(X):
    scaler = StandardScaler()
    return scaler.fit(X)


def standardize_transform(X,scaler):
    return scaler.transform(X)


def split(X,y,test_fraction):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_fraction)
    return [X_train,X_test,y_train,y_test]


def preprocess(config):
    rawdatapath = config["raw_path"] + config["data_name"] + ".csv"
    [X,y] = dataio.load(rawdatapath)
    test_fraction = 0.3
    [X_train,X_test,y_train,y_test] = split(X,y,test_fraction)
    savepath = config["interim_path"]
    dataio.save(X_train,y_train,savepath + "train.csv")
    dataio.save(X_test,y_test,savepath + "test.csv")
    
    scaler = standardize_fit(X_train)
    X_train_scaled = standardize_transform(X_train,scaler)
    X_test_scaled = standardize_transform(X_test,scaler)
    
    savepath = config["preprocessed_path"]
    dataio.save(X_train_scaled,y_train,savepath + "train.csv")
    dataio.save(X_test_scaled,y_test,savepath + "test.csv")


if __name__ == "__main__":
    config = tools.load_config()
    preprocess(config)




