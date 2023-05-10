import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn import utils

import warnings
warnings.filterwarnings('ignore')

#recursive feature selection with logistic reg
data=pd.read_csv('PreprocessedTrain.csv')
df=pd.DataFrame(data)
ogFeatures=data.columns
X=df.iloc[:,:]
lab_enc = preprocessing.LabelEncoder()
data2=pd.read_csv('games-regression-dataset.csv')
df2=pd.DataFrame(data2)


print(df.shape)
# Select Last Column
Y = df2.iloc[0:4171,-1]
Y=pd.array(Y)

print("Last Column Of Dataframe : ")
print(Y)
print('Type: ', type(Y))



#feature selection using analysis of variance f_test
# configure to select all features
fs = SelectKBest(score_func=f_classif, k='all')
# learn relationship from training data
fs.fit(X,Y)

droppedIndices=[]
for i in range(len(fs.scores_)):
 print('Feature %d: %f' % (i, fs.scores_[i]))
 if(fs.scores_[i]<1):
  droppedIndices.append(i)
# plot the scores
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()

print(droppedIndices)
df = df.drop(df.columns[[droppedIndices]],axis = 1)


print(df.shape)