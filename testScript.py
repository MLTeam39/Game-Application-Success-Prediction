# import re
# import nltk
import pandas as pd
import testPreprocessing as testPre
# import preprocessingFunctions as preFun
from main import genres, genresFreq
# from nltk.tokenize import word_tokenize


"""Take user's to-be-tested data"""

print("Welcome to our Game Application Success Prediction Model")
print("Where We are using your game's information to predict it's Average User Rating")
print("Please enter your game's:")

columns = ['URL', 'ID', 'Name', 'Subtitle', 'Icon URL', 'User Rating Count', 'Price', 'In-app Purchases', 'Description', 'Developer', 'Age Rating', 'Languages', 'Size', 'Primary Genre', 'Genres', 'Original Release Date', 'Current Version Release Date']
testData = pd.DataFrame();

# for col in columns:
#     colVal = input('{}: '.format(col))
#     testData[col] = colVal;
#
# print('Please Wait........')

"""Preprocessing Test Data"""

# testData = testPre.drop_test(testData)
# # print(testData.columns)
#
# # print(testData['Languages'].isnull().sum())
# testData = testPre.fill_nulls(testData)
# # print(testData['Languages'].isnull().sum())

# testData = testPre.inApp_test(testData)
# testData = testPre.developer_test(testData)
# testData = testPre.avgRating_test(testData)
# testData = testPre.languages_test(testData)
# testData = testPre.primary_test(testData)
# testData = testPre.genres_test(testData, genresFreq, genres)
# testData = testPre.dates_test(testData)


# testData = testPre.scaler(testData, 'Developer')
# testData = testPre.scaler(testData, 'Price')
# testData = testPre.scaler(testData, 'Size')
# testData = testPre.scaler(testData, 'User Rating Count')
#
# for row in testData['Description']:
#     row = word_tokenize(row)
#     row = [preFun.remove_NewLine(i) for i in row]
#     row = [i for i in row if i == re.sub(r'//', '', i)]
#     row = [i for i in row if i == re.sub(r'https', '', i)]
#     row = [re.sub(r'[^a-zA-Z0-9\s]+', '', preFun.remove_punc(i)) for i in row]
#     row = [preFun.remove_numbers(i) for i in row]
#     row = [word for word in row if word not in stop_words]
#     row = [i for i in row if i != '']
#     row = [stemmer.stem(word) for word in row]
#     row = [lemmatizer.lemmatize(word, pos='v') for word in row]
#     DesColtest.append(len(set(row)))
#
# testData['Description']= DesColtest
# testData = testPre.scaler(testData,'Description')
