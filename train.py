from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle

dataFrame = pd.read_csv('train.csv')
Y = dataFrame['activity'].values.reshape(-1,1)
X = dataFrame.drop('activity',axis=1).values


clf = MLPClassifier(max_iter=20000,hidden_layer_sizes=[100,],solver='adam',alpha=0.01, beta_1 = 0.8)
clf.fit(X,Y)
predictions = clf.predict(X)
print("Training Accuracy")
print(accuracy_score(Y,predictions))


pickle.dump( clf, open( "model.p", "wb" ))
dataFrame = pd.read_csv('validation.csv')
Y = dataFrame['activity'].values.reshape(-1,1)
X = dataFrame.drop('activity',axis=1).values
predictions = clf.predict(X)
print("Testing Accuracy")
print(accuracy_score(Y,predictions))
