##################################All Packages##################################
import numpy as np
import pandas as pd
import statistics
import re
import matplotlib.pyplot as plt
import preprocessingFunctions as preFun
import testPreprocessing as testPre
from sklearn import linear_model
from sklearn import metrics
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

filledColumns = {}
nltk.download('wordnet')
DesColList = []
DesColtest = []
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

##################################Loading Data##################################
data = pd.read_csv('games-regression-dataset.csv')
data2 = pd.read_csv('games-classification.csv')
X = data.drop(["Average User Rating"], axis=1)  # features
X['Rate'] = data2['Rate']
Y = data["Average User Rating"]  # label

##################################Split Data##################################
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

trainData = X_train
trainData['Average User Rating'] = Y_train
testData = X_test
testData['Average User Rating'] = Y_test

trainData.to_csv('TrainData.csv', index=False)
testData.to_csv('TestData.csv', index=False)

trainData = pd.read_csv('TrainData.csv')
testData = pd.read_csv('TestData.csv')
trainData = trainData.drop('Rate', axis=1)
testData = testData.drop('Rate', axis=1)
X_train = trainData.iloc[:, :-1]
Y_train = trainData.iloc[:, -1]
X_test = testData.iloc[:, :-1]
Y_test = testData.iloc[:, -1]

##################################Preprocessing##################################
# Check All features Nulls and Unique Percentage
X_train = preFun.feature_selection(X_train)
# print('X_train Columns after dropping : ', X_train.columns.values, '\n')

##################################User Rating Count##################################
# print(X_train['User Rating Count'])

# null means below 5 (1+2+3+4/5 = 2)
X_train['User Rating Count'].fillna(2, inplace=True)
filledColumns['User Rating Count'] = 2

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
filledColumns['In-app Purchases'] = X_train['In-app Purchases'].mode()[0]
# print(X_train['In-app Purchases'])

##################################Description##################################
for row in X_train['Description']:
    row = word_tokenize(row)
    row = [preFun.remove_NewLine(i) for i in row]
    row = [i for i in row if i == re.sub(r'//', '', i)]
    row = [i for i in row if i == re.sub(r'https', '', i)]
    row = [re.sub(r'[^a-zA-Z0-9\s]+', '', preFun.remove_punc(i)) for i in row]
    row = [preFun.remove_numbers(i) for i in row]
    row = [word for word in row if word not in stop_words]
    row = [i for i in row if i != '']
    row = [stemmer.stem(word) for word in row]
    row = [lemmatizer.lemmatize(word, pos='v') for word in row]
    DesColList.append(len(set(row)))
X_train['Description']= DesColList
X_train = preFun.feature_scaling(X_train, 'Description')

splittedRow = []
newCol = []
for row in X_train['In-app Purchases']:
    splittedRow = [float(value) for value in row.split(', ')]
    newCol.append(statistics.mean(splittedRow))

X_train['In-app Purchases'] = newCol
# print(X_train['In-app Purchases'])

##################################Developer##################################
# Feature Encoding
# print(X_train['Developer'])
developerEnc = LabelEncoder()
X_train['Developer'] = developerEnc.fit_transform(X_train['Developer'])
np.save('Developer Encoder.npy', developerEnc.classes_, allow_pickle=True)

X_train = preFun.feature_scaling(X_train, 'Developer')
# print(X_train['Developer'])

##################################Age Rating##################################
X_train.loc[X_train['Age Rating'].isin(['17+']), 'Age Rating'] = 4
X_train.loc[X_train['Age Rating'].isin(['12+']), 'Age Rating'] = 3
X_train.loc[X_train['Age Rating'].isin(['9+']), 'Age Rating'] = 2
X_train.loc[X_train['Age Rating'].isin(['4+']), 'Age Rating'] = 1

X_train['Age Rating'] = X_train['Age Rating'].astype(np.int64)

##################################Languages##################################
# print(X_train['Languages'])
valFreq = X_train['Languages'].value_counts()  # most_frequent is'EN'=2728
most_frequent = str(valFreq)
X_train['Languages'].fillna(most_frequent, inplace=True)
filledColumns['Languages'] = most_frequent

languages = []
for row in X_train['Languages']:
    # print(len(row.split()))
    languages.append(len(row.split(', ')))

X_train['Languages'] = languages
# print(X_train['Languages'])

##################################Size##################################
# print(X_train['Size'])
X_train = preFun.feature_scaling(X_train, 'Size')
# print(X_train['Size'])

##################################Primary Genre and Genres##################################
# Primary Genre vs Genres
# PrimaryGenre = X_train['Primary Genre']
# Genres = X_train['Genres']
PrimaryGenre = pd.read_csv("games-regression-dataset.csv", usecols=['Primary Genre'])
Genres = pd.read_csv("games-regression-dataset.csv", usecols=['Genres'])

count = 0

for i in range(len(PrimaryGenre.index)):
    if (
            PrimaryGenre.iloc[i].to_string().split(' ')[5].__eq__(
                (Genres.iloc[i].to_string()).split(', ')[0].split(' ')[4])):
        count = count + 1

flag = "%.2f" % (count / len(PrimaryGenre) * 100)

if float(flag) >= 99:
    X_train = X_train.drop(['Primary Genre'], axis=1)

# # Genres
# output_train = X_train['Genres'].str.get_dummies(sep=', ')
# for i in output_train.columns.values.tolist():
#     X_train[i] = output_train[i]

### Combine all genres in one column
genres = []
genresRows = [[]]
for row in X_train['Genres']:
    # if type(genres) != types.NoneType:
    #     genreRow = [gen for gen in row.split(', ') if gen not in genres]
    #     genres.extend(genreRow)
    # else:
    #     genres = row.split(', ')
    genreRow = [gen for gen in row.split(', ')]
    genresRows.append(genreRow)
    genres.extend(genreRow)

# genresFreq = X_train['Genres'].str.count('Games').sum()
# print(genresRows)

genres = pd.DataFrame(genres)
genresFreq = genres.value_counts()

for genre in genresFreq.keys():
    # print(type(genre))
    genresFreq[genre] /= len(X_train['Genres'])

# print('genres Frequency : \n', genresFreq)

eachRow = []
i = 1
for row in X_train['Genres']:
    eachRow.append(genresFreq[genresRows[i]].sum())
    i += 1

X_train['Genres'] = eachRow
# print(X_train['Genres'])

# Test
# unseenData = []
# # print(X_test['Genres'])
# output_test = X_test['Genres'].str.get_dummies(sep=', ')
# X_test['others'] = 0
# for i in output_test.columns.values.tolist():
#     if i in output_train.columns.values.tolist():
#         X_test[i] = output_test[i]
#     else:
#         unseenData.append(i)
#         X_test['others'] += output_test[i]
#
# for i in output_train.columns.values.tolist():
#     if i not in X_test.columns.values.tolist():
#         X_test[i] = '0'
#
# X_train['others'] = 0
# X_train = X_train.drop(['Genres'], axis=1)
#
# # TODO: Test
# print('unseenData : ', unseenData)
# print('row 191 genres: ', X_test['Genres'][191])
# print('row 191 others: ', X_test['others'][191])
# print('row 696 genres: ', X_test['Genres'][696])
# print('row 696 others: ', X_test['others'][696])
# X_test = X_test.drop(['Genres'], axis=1)

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

##################################Average User Rating##################################

# # Y_train = preFun.feature_scaling(Y_train, 'Average User Rating')
# print('Y_train before reshaping : \n', Y_train, '\n')
# reshaped_train_col = np.array(Y_train).reshape(-1, 1)
# print('Y_train after reshaping : \n', reshaped_train_col, '\n')
#
# # TODO: Check scaling range
# scaler = MaxAbsScaler()
# scaled_train_col = scaler.fit_transform(reshaped_train_col)
# scaler_path = 'Average User Rating' + ' Scaler.gz'
# joblib.dump(scaler, scaler_path)
# # TODO: Test
# # my_scaler = joblib.load('scaler.gz')
# # scaled_test_col = my_scaler.transform(reshaped_test_col)
#
# # print('X_train before reshaping : \n', scaled_train_col, '\n')
# Y_train = scaled_train_col.reshape(1, -1)[0]
# print(type(Y_train))
# print('Y_train after reshaping : \n', Y_train, '\n')
# Y_train = pd.Series(Y_train)

##################################Test##################################

testPre.filledColumns = filledColumns
# print(X_test.columns)
X_test = testPre.drop_test(X_test)
# print(X_test.columns)

# print(X_test['Languages'].isnull().sum())
X_test = testPre.fill_nulls(X_test)
# print(X_test['Languages'].isnull().sum())

X_test = testPre.inApp_test(X_test)
X_test = testPre.developer_test(X_test)
X_test = testPre.avgRating_test(X_test)
X_test = testPre.languages_test(X_test)
X_test = testPre.primary_test(X_test)
X_test = testPre.genres_test(X_test, genresFreq, genres)
X_test = testPre.dates_test(X_test)

X_test = testPre.scaler(X_test, 'Developer')
X_test = testPre.scaler(X_test, 'Price')
X_test = testPre.scaler(X_test, 'Size')
X_test = testPre.scaler(X_test, 'User Rating Count')

for row in X_test['Description']:
    row = word_tokenize(row)
    row = [preFun.remove_NewLine(i) for i in row]
    row = [i for i in row if i == re.sub(r'//', '', i)]
    row = [i for i in row if i == re.sub(r'https', '', i)]
    row = [re.sub(r'[^a-zA-Z0-9\s]+', '', preFun.remove_punc(i)) for i in row]
    row = [preFun.remove_numbers(i) for i in row]
    row = [word for word in row if word not in stop_words]
    row = [i for i in row if i != '']
    row = [stemmer.stem(word) for word in row]
    row = [lemmatizer.lemmatize(word, pos='v') for word in row]
    DesColtest.append(len(set(row)))
    
X_test['Description']= DesColtest
X_test = testPre.scaler(X_test,'Description')

# # # Test
# # Y_test = testPre.scaler(Y_test, 'Average User Rating')
# reshaped_test_col = np.array(Y_test).reshape(-1, 1)
# scaler_path = 'Average User Rating' + ' Scaler.gz'
# scaler = joblib.load(scaler_path)
# Y_test = scaler.transform(reshaped_test_col)

X_train.to_csv('PreprocessedTrain.csv', index=False)
X_test.to_csv('PreprocessedTest.csv', index=False)

##################################Correlation##################################
# Feature Selection
print('Correlation Results: ')
for i in X_train:
    print(i)
    print(X_train[i].corr(Y_train))

# Get the correlation between the features
corr_data = X_train
corr_data['Average User Rating'] = Y_train
corr = corr_data.corr()

top_features = corr.index[abs(corr['Average User Rating']) >= 0.02]

top_features = top_features.delete(-1)
print('\nTop Features :\n', top_features.values, '\n')

X_train = X_train[top_features]
X_test = X_test[top_features]

##################################Regression##################################
# print(X_train.columns)
# print(X_test.columns)

X_test = X_test.reindex(columns=X_train.columns)

"""Linear Reg"""

for col in X_train:
    model = LinearRegression()
    X = np.expand_dims(X_train[col], axis=1)
    Y = np.expand_dims(Y_train, axis=1)
    X_test_col = np.expand_dims(X_test[col], axis=1)
    model.fit(X, Y)
    y_pred = model.predict(X_test_col)

    print('-Linear Regression Using ', col)
    print('Mean Square Error', metrics.mean_squared_error(Y_test, y_pred))
    print('Accuracy', "%.4f" % (metrics.r2_score(Y_test, y_pred)), '\n')

    plt.figure(figsize=(4, 3))
    ax = plt.axes()
    ax.scatter(X, Y)
    ax.plot(X_test_col, y_pred, color='red')
    plt.title('Linear Regression')
    ax.set_xlabel(col)
    ax.set_ylabel('Average User Rating')
    ax.axis('tight')
    # plt.show()
    
"""Multiple Reg"""

regr = linear_model.LinearRegression()
regr.fit(X_train, Y_train)
predicted = regr.predict(X_test)
y_pred = regr.predict(X_test)
print('-Multiple Regression:')
print('Mean Square Error', metrics.mean_squared_error(np.asarray(Y_test), y_pred))
print('Accuracy', "%.4f" % (metrics.r2_score(Y_test, y_pred)), '\n')

"""Polynomial Reg"""

# Features
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)

# fit features to Linear Regression model
poly_model = LinearRegression()
poly_model.fit(X_train_poly, Y_train)

# predicting on training data
y_train_predicted = poly_model.predict(X_train_poly)
y_pred = poly_model.predict(poly_features.transform(X_test))

# predicting on test data-set
prediction = poly_model.predict(poly_features.fit_transform(X_test))

# print('Co-efficient of linear regression', poly_model.coef_)
# print('Intercept of linear regression model', poly_model.intercept_)
print('-Polynomail Regression:')
print('Mean Square Error', metrics.mean_squared_error(Y_test, prediction))
print('Accuracy', "%.4f" % (metrics.r2_score(Y_test, prediction)), '\n')
for col in X_train:
    # Fitting Polynomial Regression to the dataset
    X = np.array(X_train[col]).reshape(-1, 1)
    Y = np.array(Y_train).reshape(-1, 1)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    poly.fit(X_poly, Y)
    lin2 = LinearRegression()
    lin2.fit(X_poly, Y)
    # Visualising the Polynomial Regression results
    plt.scatter(X, Y, color='blue')

    plt.plot(X, lin2.predict(poly.fit_transform(X)), color='red')
    plt.title('Polynomial Regression')
    plt.xlabel(col)
    plt.ylabel('Average User Rating')

    # plt.show()
"""Gradient Boosting Reg"""
est = GradientBoostingRegressor(n_estimators=46, max_depth=3)

est.fit(X_train, Y_train)
# predict class labels
pred = est.predict(X_test)
# score on test data (accuracy)
print('-Gradient Boosting Regressor:')
print('Mean Square Error', metrics.mean_squared_error(Y_test, pred))
print('Accuracy', "%.4f" % (metrics.r2_score(Y_test, pred)), '\n')
