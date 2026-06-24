import numpy as np
import pandas as pd
from sklearn.metrics import *

def dot(vector1, vector2):
    return np.dot(vector1, vector2)

def transfrom_pandas(group, result, target=1):
    true = (group == target).astype(int)
    top5_idx = result.nlargest(5).index
    pred = pd.Series(0, index=result.index)
    pred.loc[top5_idx] = 1
    return true.values, result.values, pred.values

def accuracy(true, pred):
    return accuracy_score(true, pred)

def precision(true, pred):
    return precision_score(true, pred)

def recall(true, pred):
    return recall_score(true, pred)

def f1(true, pred):
    return f1_score(true, pred)