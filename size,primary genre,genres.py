#!/usr/bin/env python
# coding: utf-8

# In[109]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import datetime as dt
from sklearn.linear_model import LinearRegression
import itertools 
import functools
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler

#Load data
data = pd.read_csv("games-regression-dataset.csv")


##########################################
#Drop the rows that contain missing values
data.dropna(how='any',inplace=True)
data.drop_duplicates()

##########################################
_data=data.iloc[:,:]
X=data.iloc[:,0:16] #Features
Y=data['Average User Rating'] #Label

##########################################
#Split the data to training and testing sets                          
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20,shuffle=True)

##########################################
##feature scaling
##'Size'
Size_new=pd.read_csv("games-regression-dataset.csv",usecols=['Size'])

#print(data['Size'].isna().sum())

tempFeat = np.array(Size_new['Size']).reshape(-1,1)
scaler = MaxAbsScaler()
scaler.fit(tempFeat)
scaledFeat = scaler.transform(tempFeat)
Size_new['Size'] = scaledFeat.reshape(1,-1)[0]
data_cpycpy=data.drop(['Size'],axis=1)
data= pd.DataFrame(data_cpycpy)
data['Size'] = Size_new['Size']

##########################################
##'Primary Genre'
##'Genres'
PrimaryGenre =pd.read_csv("games-regression-dataset.csv",usecols=['Primary Genre'])
Genres =pd.read_csv("games-regression-dataset.csv",usecols=['Genres'])

count =0

for i  in range(len(PrimaryGenre.index)):
    if(PrimaryGenre.iloc[i].to_string().split(' ')[5].__eq__((Genres.iloc[i].to_string()).split(', ')[0].split(' ')[4])):
            count=count+1
  
        
flag="%.2f"%(count/len(PrimaryGenre)*100)    

if (float(flag)>=99):
    data = data.drop(['Primary Genre'],axis=1)
    
########################################## 
##'Genres'

output = data['Genres'].str.get_dummies(sep=', ')

for i in output.columns.values.tolist():
    data[i]=output[i]


