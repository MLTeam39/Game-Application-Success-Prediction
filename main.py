##################################All Packages##################################
import numpy

import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import preprocessingFunctions as preFun
import testPreprocessing as testPre
from scipy import stats
from scipy.stats import norm
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

##################################Loading Data##################################
data = pd.read_csv('datasets\\games-regression-dataset.csv')
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

##################################Split Data##################################
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

trainData = X_train
trainData['Average User Rating'] = Y_train
testData = X_test
testData['Average User Rating'] = Y_test

trainData.to_csv('datasets\\TrainData.csv', index=False)
testData.to_csv('datasets\\TestData.csv', index=False)

trainData = pd.read_csv('datasets\\TrainData.csv')
testData = pd.read_csv('datasets\\TestData.csv')
X_train = trainData.iloc[:, :-1]
Y_train = trainData.iloc[:, -1]
X_test = testData.iloc[:, :-1]
Y_test = testData.iloc[:, -1]

##################################Preprocessing##################################
# Check All features Nulls and Unique Percentage
X_train = preFun.feature_selection(X_train)
print('X_train Columns after dropping : ', X_train.columns.values, '\n')

##################################User Rating Count##################################
# print(X_train['User Rating Count'])
# null means below 5 (1+2+3+4/5 = 2)
X_train['User Rating Count'].fillna(2, inplace=True)
# TODO: Scaling from 0 -> 1 or -1 -> 1
X_train = preFun.feature_scaling(X_train, 'User Rating Count')
# print(X_train['User Rating Count'])

##################################Price##################################
# print(X_train['Price'])
X_train = preFun.feature_scaling(X_train, 'Price')
# print(X_train['Price'])

##################################In-app Purchases##################################
# Fill nulls
# print(X_train['In-app Purchases'])
X_train['In-app Purchases'].fillna(X_train['In-app Purchases'].mode()[0], inplace=True)
# print(X_train['In-app Purchases'])
# TODO: Test
X_test['In-app Purchases'].fillna(X_train['In-app Purchases'].mode()[0], inplace=True)

splittedRow = []
newCol = []
for row in X_train['In-app Purchases']:
    splittedRow = [float(value) for value in row.split(', ')]
    newCol.append(statistics.mean(splittedRow))

X_train['In-app Purchases'] = newCol
# print(X_train['In-app Purchases'])

# TODO: Test
testRow = []
testCol = []
for test in X_test['In-app Purchases']:
    testRow = [float(v) for v in test.split(', ')]
    testCol.append(statistics.mean(testRow))

X_test['In-app Purchases'] = testCol

# for i in X_train['In-app Purchases']:
#     splittedRow = i.split(', ')
#     print(splittedRow)
#
#     for item in splittedRow:
#         newRow.append(float(item))
#
#     newCol.append(statistics.mean(newRow))

##################################Developer##################################
# Feature Encoding
# print(X_train['Developer'])
developerEnc = LabelEncoder()
X_train['Developer'] = developerEnc.fit_transform(X_train['Developer'])
np.save('preprocessingData\\Developer Encoder.npy', developerEnc.classes_, allow_pickle=True)

# TODO: Test
encoder = LabelEncoder()
encoder.classes_ = np.load('preprocessingData\\Developer Encoder.npy', allow_pickle=True)
X_test['Developer'] = X_test['Developer'].map(lambda s: 'unknown' if s not in encoder.classes_ else s)
encoder.classes_ = np.append(encoder.classes_, 'unknown')
X_test['Developer'] = encoder.transform(X_test['Developer'])


X_train = preFun.feature_scaling(X_train, 'Developer')
# print(X_train['Developer'])

##################################Age Rating##################################
# TODO: approximate the input to the nearest value ex 15+ is near to 17+
X_train.loc[X_train['Age Rating'].isin(['17+']), 'Age Rating'] = 4
X_train.loc[X_train['Age Rating'].isin(['12+']), 'Age Rating'] = 3
X_train.loc[X_train['Age Rating'].isin(['9+']), 'Age Rating'] = 2
X_train.loc[X_train['Age Rating'].isin(['4+']), 'Age Rating'] = 1

X_train['Age Rating'] = X_train['Age Rating'].astype(np.int64)

# TODO: Test
notInRate = ['4+', '9+', '12+', '17+']
col = X_test['Age Rating']
for i in range(len(col.index)):
    if col.iloc[i] not in notInRate:
        col.iloc[i] = 0.0

X_test['Age Rating'] = col
X_test.loc[X_test['Age Rating'].isin(['17+']), 'Age Rating'] = 4
X_test.loc[X_test['Age Rating'].isin(['12+']), 'Age Rating'] = 3
X_test.loc[X_test['Age Rating'].isin(['9+']), 'Age Rating'] = 2
X_test.loc[X_test['Age Rating'].isin(['4+']), 'Age Rating'] = 1

X_test['Age Rating'] = X_train['Age Rating'].astype(np.int64)

##################################Languages##################################
# print(X_train['Languages'])
valFreq = X_train['Languages'].value_counts()  # most_frequent is'EN'=2728
most_frequent = str(valFreq)
X_train['Languages'].fillna(most_frequent, inplace=True)

languages = []
for row in X_train['Languages']:
    # print(len(row.split()))
    languages.append(len(row.split(', ')))

X_train['Languages'] = languages
# print(X_train['Languages'])

# TODO: Test
X_test['Languages'].fillna(most_frequent, inplace=True)
languages = []
for row in X_test['Languages']:
    # print(len(row.split()))
    languages.append(len(row.split(', ')))

X_test['Languages'] = languages

##################################Size##################################
# print(X_train['Size'])
X_train = preFun.feature_scaling(X_train, 'Size')
# print(X_train['Size'])

##################################Primary Genre and Genres##################################
# Primary Genre vs Genres
# PrimaryGenre = X_train['Primary Genre']
# Genres = X_train['Genres']
PrimaryGenre = pd.read_csv("datasets\\games-regression-dataset.csv", usecols=['Primary Genre'])
Genres = pd.read_csv("datasets\\games-regression-dataset.csv", usecols=['Genres'])

count = 0

for i in range(len(PrimaryGenre.index)):
    if (
            PrimaryGenre.iloc[i].to_string().split(' ')[5].__eq__(
                (Genres.iloc[i].to_string()).split(', ')[0].split(' ')[4])):
        count = count + 1

flag = "%.2f" % (count / len(PrimaryGenre) * 100)

if float(flag) >= 99:
    X_train = X_train.drop(['Primary Genre'], axis=1)
    # TODO: Test
    X_test = X_test.drop(['Primary Genre'], axis=1)

# Genres
output_train = X_train['Genres'].str.get_dummies(sep=', ')
for i in output_train.columns.values.tolist():
    X_train[i] = output_train[i]

# TODO: Test
unseenData = []
# print(X_test['Genres'])
output_test = X_test['Genres'].str.get_dummies(sep=', ')
X_test['others'] = 0
for i in output_test.columns.values.tolist():
    if i in output_train.columns.values.tolist():
        X_test[i] = output_test[i]
    else:
        unseenData.append(i)
        X_test['others'] += output_test[i]


for i in output_train.columns.values.tolist():
    if i not in X_test.columns.values.tolist():
        X_test[i] = '0'

X_train['others'] = 0
X_train = X_train.drop(['Genres'], axis=1)
# TODO: Test
print('unseenData : ', unseenData)
print('row 191 genres: ', X_test['Genres'][191])
print('row 191 others: ', X_test['others'][191])
print('row 696 genres: ', X_test['Genres'][696])
print('row 696 others: ', X_test['others'][696])
X_test = X_test.drop(['Genres'], axis=1)

##################################Dates##################################
# Reformat Dates
# print(X_train['Original Release Date'])
X_train['Original Release Date'] = pd.to_datetime(X_train['Original Release Date'], dayfirst=True)
X_train['Original Release Year'] = X_train['Original Release Date'].dt.year
X_train['Original Release Month'] = X_train['Original Release Date'].dt.month
X_train['Original Release Day'] = X_train['Original Release Date'].dt.day
# X_train['Original Release Year'] = X_train['Original Release Year'].astype(float)
# X_train['Original Release Month'] = X_train['Original Release Month'].astype(float)
# X_train['Original Release Day'] = X_train['Original Release Day'].astype(float)
# print(data['Original Release Date'].corr(data['Average User Rating']))
# print(X_train['Original Release Year'])
# print(X_train['Original Release Month'])
# print(X_train['Original Release Day'])
X_train = X_train.drop(['Original Release Date'], axis=1)


X_train['Current Version Release Date'] = pd.to_datetime(X_train['Current Version Release Date'], dayfirst=True)
X_train['Current Version Release Year'] = X_train['Current Version Release Date'].dt.year
X_train['Current Version Release Month'] = X_train['Current Version Release Date'].dt.month
X_train['Current Version Release Day'] = X_train['Current Version Release Date'].dt.day
# X_train['Current Version Release Year'] = X_train['Current Version Release Year'].astype(float)
# X_train['Current Version Release Month'] = X_train['Current Version Release Month'].astype(float)
# X_train['Current Version Release Day'] = X_train['Current Version Release Day'].astype(float)
# print(data['Original Release Date'].corr(data['Average User Rating']))
X_train = X_train.drop(['Current Version Release Date'], axis=1)

# TODO: Test
# print(X_test['Original Release Date'])
X_test['Original Release Date'] = pd.to_datetime(X_test['Original Release Date'], dayfirst=True)
X_test['Original Release Year'] = X_test['Original Release Date'].dt.year
X_test['Original Release Month'] = X_test['Original Release Date'].dt.month
X_test['Original Release Day'] = X_test['Original Release Date'].dt.day
# print(X_test['Original Release Year'])
# print(X_test['Original Release Month'])
# print(X_test['Original Release Day'])
X_test = X_test.drop(['Original Release Date'], axis=1)

X_test['Current Version Release Date'] = pd.to_datetime(X_test['Current Version Release Date'], dayfirst=True)
X_test['Current Version Release Year'] = X_test['Current Version Release Date'].dt.year
X_test['Current Version Release Month'] = X_test['Current Version Release Date'].dt.month
X_test['Current Version Release Day'] = X_test['Current Version Release Date'].dt.day
X_test = X_test.drop(['Current Version Release Date'], axis=1)

########################################################################################################################################

# print(X_test.columns)
X_test = testPre.drop_test(X_test)
# print(X_test.columns)

X_test = testPre.scaler(X_test, 'Developer')
X_test = testPre.scaler(X_test, 'Price')
X_test = testPre.scaler(X_test, 'Size')
X_test = testPre.scaler(X_test, 'User Rating Count')

X_train.to_csv('datasets\\PreprocessedTrain.csv', index=False)
X_test.to_csv('datasets\\PreprocessedTest.csv', index=False)

##################################Correlation##################################
for i in X_train:
    print(i, type(X_train[i][0]))
    print(X_train[i].corr(Y_train))

##################################Regression##################################
# print(X_train.columns)
# print(X_test.columns)

X_test = X_test.reindex(columns=X_train.columns)

"""Linear reg """
model = LinearRegression()
X = np.expand_dims(X_train['Current Version Release Year'], axis=1)
Y = np.expand_dims(Y_train, axis=1)
X_test_year = np.expand_dims(X_test['Current Version Release Year'], axis=1)
model.fit(X, Y)
y_pred = model.predict(X_test_year)
print('Mean Square Error', metrics.mean_squared_error(Y_test, y_pred))
print('Accuracy', "%.2f" % (metrics.r2_score(Y_test, y_pred)*100), '%')
# print(f"coefficient of determination: {r_sq}")

"""multiple reg"""

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
predicted = regr.predict(X_test)
y_pred = regr.predict(X_test)
print('Mean Square Error', metrics.mean_squared_error(np.asarray(Y_test), y_pred))
print('Accuracy', "%.2f" % (metrics.r2_score(Y_test, y_pred)*100), '%')
# compare between expected and trained answers
