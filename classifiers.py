import joblib
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
##################################Loading Data##################################
X_Train = pd.read_csv('PreprocessedTrain.csv')
Y_Train = pd.read_csv('TrainData.csv')
Y_Train = Y_Train['Rate']
X_Test = pd.read_csv('PreprocessedTest.csv')
Y_Test = pd.read_csv('TestData.csv')
Y_Test = Y_Test['Rate']

# print(Y_Train)
# print(X_Train)

##################################Preprocessing Y##################################
d = {'High': 2, 'Intermediate': 1, 'Low': 0}
Y_Train = Y_Train.map(d)

##################################Feature Selection##################################
# feature selection using analysis of variance f_test
# configure to select all features
fs = SelectKBest(score_func=f_classif, k='all')
# learn relationship from training data
fs.fit(X_Train, Y_Train)

droppedIndices = []
print('Features Scores:')
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
    if fs.scores_[i] < 1:
        droppedIndices.append(i)
print()

# plot the scores
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()

# print(droppedIndices)
droppedCols = X_Train.columns[droppedIndices]
# print(droppedCols)

# TODO: Save Classification top features to use later for testing
features_file = open('Clf_Dropped_Features.txt', 'w')
for col in droppedCols:
    features_file.write('%s\n' % col)
features_file.close()

X_Train = X_Train.drop(droppedCols, axis=1)
X_Test = X_Test.drop(droppedCols, axis=1)

# print(X_Train.shape)

##################################Classification##################################
print('Classification Models:')

"""Decision Tree"""
dtree = DecisionTreeClassifier(max_leaf_nodes=4)
dtree = dtree.fit(X_Train, Y_Train)
Y_Test = Y_Test.map(d)
res_pred = dtree.predict(X_Test)
score = accuracy_score(Y_Test, res_pred)
print('-Decision Tree:')
print('Accuracy', "%.4f" % score, '\n')

# TODO : Save model to use later for testing
joblib.dump(dtree, 'Decision_Clf_Model')


"""KNN"""
knn_model = KNeighborsRegressor(n_neighbors=2)
knn_model.fit(X_Train, Y_Train)
train_preds = knn_model.predict(X_Train)
mse = mean_squared_error(Y_Train, train_preds)
root_mse = sqrt(mse)
print('-KNN:')
print('Mean Square Error', mse)
print('Root Mean Square Error', root_mse, '\n')

# TODO : Save model to use later for testing
joblib.dump(knn_model, 'KNN_Clf_Model')


"""SVM"""
svm_clf = svm.SVC(kernel='linear', degree=1)
svm_clf.fit(X_Train, Y_Train)
accuracy_score = svm_clf.score(X_Test, Y_Test)
print('-SVM:')
print('Accuracy', "%.4f" % accuracy_score, '\n')

# TODO : Save model to use later for testing
joblib.dump(svm_clf, 'SVM_Clf_Model')

##################################Bagging Idea##################################
print('Bagging Idea:')

Bagging = BaggingClassifier(base_estimator=dtree, n_estimators=10, random_state=50)
Bagging.fit(X_Train, Y_Train)
score_Bagging = Bagging.score(X_Test, Y_Test)

print(f"Accuracy Bagging Idea: {score_Bagging}",'\n')

##################################Boosting Idea##################################
print('Boosting Idea:')

Boosting = AdaBoostClassifier(base_estimator=dtree, n_estimators=100, learning_rate=0.1, random_state=42)
Boosting.fit(X_Train, Y_Train)
score_Boosting = Boosting.score(X_Test, Y_Test)

print(f"Accuracy Boosting Idea: {score_Boosting}",'\n')

##################################Random Forest##################################
print('Random Forest:')

RandomForest = RandomForestClassifier(n_estimators=100, random_state=42)
RandomForest.fit(X_Train, Y_Train)
score_RandomForest = RandomForest.score(X_Test, Y_Test)

print(f"Accuracy Random Forest: {score_RandomForest}",'\n')
