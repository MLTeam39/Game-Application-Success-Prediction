import pandas as pd
from math import sqrt
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

##################################Loading Data##################################
X_Train = pd.read_csv('PreprocessedTrain.csv')
Y_Train = pd.read_csv('TrainData.csv')
Y_Train = Y_Train['Rate']
X_Test = pd.read_csv('PreprocessedTest.csv')
Y_Test = pd.read_csv('TestData.csv')
Y_Test = Y_Test['Rate']
print(Y_Train)
print(X_Train)
"""Decision Tree"""
d = {'High': 2, 'Intermediate': 1, 'Low': 0}
Y_Train = Y_Train.map(d)
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

