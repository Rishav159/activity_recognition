from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle

dataFrame = pd.read_csv('validation.csv')
Y = dataFrame['activity'].values.reshape(-1,1)
X = dataFrame.drop('activity',axis=1).values


clf = pickle.load( open( "model.p", "rb" ))
predictions = clf.predict(X)
print(accuracy_score(Y,predictions))
