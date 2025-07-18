from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np


def separate_xy(dataframe):
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, 12].values
    return [X, y]

def combine_xy(X,y):
    return np.concatenate((X,y [:, np.newaxis]),axis=1)

def load(datapath):
    dataset = pd.read_csv(datapath, header=0) 
    [X, y] = separate_xy(dataset)
    return [X, y]

def save(X,y,savepath):
    combined = combine_xy(X,y)
    df = pd.DataFrame(combined)
    df.to_csv(savepath, header=False, index=False)