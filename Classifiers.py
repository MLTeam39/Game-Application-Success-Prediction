import pandas as pd
from math import sqrt
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import utils
##################################Loading Data##################################
X_Train = pd.read_csv('PreprocessedTrain.csv')
Y_Train = pd.read_csv('TrainData.csv')
Y_Train = Y_Train['Rate']
X_Test = pd.read_csv('PreprocessedTest.csv')
Y_Test = pd.read_csv('TestData.csv')
Y_Test = Y_Test['Rate']
print(Y_Train)
print(X_Train)


d = {'High': 2, 'Intermediate': 1, 'Low': 0}
Y_Train = Y_Train.map(d)




#feature selection using analysis of variance f_test
# configure to select all features
fs = SelectKBest(score_func=f_classif, k='all')
# learn relationship from training data
fs.fit(X_Train,Y_Train)

droppedIndices=[]
for i in range(len(fs.scores_)):
 print('Feature %d: %f' % (i, fs.scores_[i]))
 if(fs.scores_[i]<1):
  droppedIndices.append(i)
# plot the scores
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()

print(droppedIndices)
X_Train = X_Train.drop(X_Train.columns[[droppedIndices]],axis = 1)
X_Test= X_Test.drop(X_Test.columns[[droppedIndices]],axis = 1)

print(X_Train.shape)


##decision trees

dtree = DecisionTreeClassifier(max_leaf_nodes=4)
dtree = dtree.fit(X_Train, Y_Train)
Y_Test = Y_Test.map(d)
res_pred = dtree.predict(X_Test)
score = accuracy_score(Y_Test, res_pred)
print(score)

"""KNN"""
knn_model = KNeighborsRegressor(n_neighbors=2)
knn_model.fit(X_Train, Y_Train)
train_preds = knn_model.predict(X_Train)
mse = mean_squared_error(Y_Train, train_preds)
rmse = sqrt(mse)
print(rmse)
"""SVM"""
svm_clf = svm.SVC(kernel='linear', degree=1)
svm_clf.fit(X_Train, Y_Train)
accuracy_score = svm_clf.score(X_Test, Y_Test)
print(accuracy_score)

