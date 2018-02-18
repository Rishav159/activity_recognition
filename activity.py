from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

dataFrame = pd.read_csv('phishing.csv')
Y = dataFrame['Result'].values.reshape(-1,1)
X = dataFrame.drop('Result',axis=1).values
Y[Y==-1] = 0
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
# X_train, X_test, Y_train, Y_test = X_train.T, X_test.T, Y_train.T, Y_test.T

from l_layer_nn import LLayerNeuralNetwork
clf = LLayerNeuralNetwork(iterations=20000,hidden_layer_sizes=[25],learning_rate=0.1)
clf.fit(X.T,Y.T)
predictions = clf.predict(X.T)*1
print Y.T
print accuracy_score(Y.T[0],predictions[0])
