# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np # for numerical operations
import pandas as pd # for handling input data
import matplotlib.pyplot as plt # for data visualization 
"""import seaborn as sns # for data visualization """
import timeit
import sklearn 
import random
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
"""from Pre_processing import *"""

from sklearn.preprocessing import LabelEncoder
import numpy as np

data = pd.read_csv("C:/Users/Tech/Downloads/Lab3/games-regression-dataset.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
"""count duplicates in Names"""
print(data)
len(data['Name'])-len(data['Name'].drop_duplicates())
print(data)



len(data['ID'])-len(data['Name'].drop_duplicates())
print(data)

data = data.drop(data.columns[[0,1,2,4,11]], axis=1) #droped columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(data)
"""Age Rating"""
data['NewAgeRate'] = data['Age Rating']
data.loc[data['NewAgeRate'].isin(['17+']), 'NewAgeRate'] = '3'
data.loc[data['NewAgeRate'].isin(['12+']), 'NewAgeRate'] = '2'
data.loc[data['NewAgeRate'].isin(['9+']), 'NewAgeRate'] = '1'
data.loc[data['NewAgeRate'].isin(['4+']), 'NewAgeRate'] = '0'
"""counts nulls"""
counts = data.isna().sum()




