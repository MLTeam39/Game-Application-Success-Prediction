# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from scipy.stats import f_oneway
# from sklearn.preprocessing import LabelEncoder
#
# # Loading data
# data = pd.read_csv("C:/Users/Dell/Downloads/games-regression-dataset.csv")
# X = data.drop(["Average User Rating"], axis=1)  # features
# Y = data["Average User Rating"]  # label
# # print(X)
#
# # Split the data to training and testing sets
# Y = np.array(Y)
# Y = Y.reshape((X.shape[0], 1))
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
#
# # pre-processing
#
# # Missing Values
# print(X_train['Original Release Date'].isnull().sum())  # there is no missing value
# print(X_train['Current Version Release Date'].isnull().sum())  # there is no missing value
# print(X_train['Languages'].isnull().sum())  # there is 11 missing value
#
# X_train['Languages'].value_counts()  # most_frequent is'EN'=2728
# most_frequent = 'EN'
# X_train['Languages'].fillna(most_frequent, inplace=True)
# X_test['Languages'].fillna(most_frequent, inplace=True)
# print('null',X_test['Languages'].isnull().sum())  # there is 11 missing value
#
# # Reformate Date
# X_train['Original Release Date'] = pd.to_datetime(X_train['Original Release Date'], dayfirst=True)
# X_train['Original Release Year'] = X_train['Original Release Date'].dt.year
# X_train['Original Release Month'] = X_train['Original Release Date'].dt.month
# X_train['Original Release Day'] = X_train['Original Release Date'].dt.day
# X_train['Original Release Year'] = X_train['Original Release Year'].astype(float)
# X_train['Original Release Month'] = X_train['Original Release Month'].astype(float)
# X_train['Original Release Day'] = X_train['Original Release Day'].astype(float)
# # print(data['Original Release Date'].corr(data['Average User Rating']))
#
# X_train['Current Version Release Date'] = pd.to_datetime(X_train['Current Version Release Date'], dayfirst=True)
# X_train['Current Version Release Year'] = X_train['Current Version Release Date'].dt.year
# X_train['Current Version Release Month'] = X_train['Current Version Release Date'].dt.month
# X_train['Current Version Release Day'] = X_train['Current Version Release Date'].dt.day
# X_train['Current Version Release Year'] = X_train['Current Version Release Year'].astype(float)
# X_train['Current Version Release Month'] = X_train['Current Version Release Month'].astype(float)
# X_train['Current Version Release Day'] = X_train['Current Version Release Day'].astype(float)
# # print(data['Original Release Date'].corr(data['Average User Rating']))
#
# # Languages
# # print(data.at[0, 'Languages'])
# i = 0
# for row in X_train['Languages']:
#     # print(len(row.split()))
#     print(row,type(row))
#     X_train.at[i, 'Languages'] = len(row.split())
#     i += 1
# print(data['Languages'])
# # data['Num_Languages'] = data['Languages']
# # print(data['Num_Languages'])
# # print(data['Num_Languages'].corr(data['Average User Rating']))
#
# # output = pd.DataFrame
# # output = data['Languages'].str.get_dummies(sep=', ')
# # print(output)
# # print(output.columns.values.tolist())
# # print(languages_data_updated.value_counts())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Loading data
data = pd.read_csv("C:/Users/Dell/Downloads/games-regression-dataset.csv")
X = data.drop(["Average User Rating"], axis=1)  # features
Y = data["Average User Rating"]  # label
# print(X)
most_frequent = 'EN'
# data['Languages'].fillna(most_frequent, inplace=True)
data = data.dropna()
# Split the data to training and testing sets
Y = np.array(Y)
Y = Y.reshape((X.shape[0], 1))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
X_train = X_train.dropna()

print('null',X_train['Languages'].isnull().sum())  # there is 11 missing value

# pre-processing
# print(X_train['Languages'])
# Missing Values
print(X_train['Original Release Date'].isnull().sum())  # there is no missing value
print(X_train['Current Version Release Date'].isnull().sum())  # there is no missing value

# X_train['Languages'].value_counts()  # most_frequent is'EN'=2728


# Reformate Date
X_train['Original Release Date'] = pd.to_datetime(X_train['Original Release Date'], dayfirst=True)
X_train['Original Release Year'] = X_train['Original Release Date'].dt.year
X_train['Original Release Month'] = X_train['Original Release Date'].dt.month
X_train['Original Release Day'] = X_train['Original Release Date'].dt.day
X_train['Original Release Year'] = X_train['Original Release Year'].astype(float)
X_train['Original Release Month'] = X_train['Original Release Month'].astype(float)
X_train['Original Release Day'] = X_train['Original Release Day'].astype(float)
# print(data['Original Release Date'].corr(data['Average User Rating']))

X_train['Current Version Release Date'] = pd.to_datetime(X_train['Current Version Release Date'], dayfirst=True)
X_train['Current Version Release Year'] = X_train['Current Version Release Date'].dt.year
X_train['Current Version Release Month'] = X_train['Current Version Release Date'].dt.month
X_train['Current Version Release Day'] = X_train['Current Version Release Date'].dt.day
X_train['Current Version Release Year'] = X_train['Current Version Release Year'].astype(float)
X_train['Current Version Release Month'] = X_train['Current Version Release Month'].astype(float)
X_train['Current Version Release Day'] = X_train['Current Version Release Day'].astype(float)
# print(data['Original Release Date'].corr(data['Average User Rating']))

# Languages
i = 0
e=[]
for row in X_train['Languages']:
    print(row,type(row))
    print(i)
    # X_train.at[i, 'Num_Languages'] = len(row.split())
    e.append(len(row.split()))
    i += 1
X_train['Num_Languages'] =e
print(X_train['Num_Languages'])
# print(data['Num_Languages'].corr(data['Average User Rating']))

# output = pd.DataFrame
# output = data['Languages'].str.get_dummies(sep=', ')
# print(output)
# print(output.columns.values.tolist())
# print(languages_data_updated.value_counts())