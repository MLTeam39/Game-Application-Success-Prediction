import re
import joblib
import pickle
import statistics
import numpy as np
import pandas as pd
import preprocessingFunctions as preFun
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder


def drop_test(x):
    # TODO: Load training dropped columns
    with open('Dropped.txt', 'r') as file:
        dropped = [line.strip() for line in file.readlines()]
    file.close()
    x = x.drop(dropped, axis=1)
    return x


def fill_nulls(x):
    # TODO: Load training filled columns values
    with open('Filled.pkl', 'rb') as f:
        filled = pickle.load(f)
    print(filled)
    for colName in filled.keys():
        x[colName].fillna(filled[colName], inplace=True)
    return x


def scaler_fun(x, col_name):
    reshaped_test_col = np.array(x[col_name]).reshape(-1, 1)
    scaler_path = col_name + ' Scaler.gz'
    scaler = joblib.load(scaler_path)
    x[col_name] = scaler.transform(reshaped_test_col)
    return x


def in_app_test(x):
    test_row = []
    test_col = []
    for test in x['In-app Purchases']:
        test_row = [float(v) for v in test.split(', ')]
        test_col.append(statistics.mean(test_row))

    x['In-app Purchases'] = test_col
    return x


def description_text(x):
    des_col_list = []
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    for row in x['Description']:
        row = word_tokenize(row)
        row = [preFun.remove_new_line(i) for i in row]
        row = [i for i in row if i == re.sub(r'//', '', i)]
        row = [i for i in row if i == re.sub(r'https', '', i)]
        row = [re.sub(r'[^a-zA-Z0-9\s]+', '', preFun.remove_punc(i)) for i in row]
        row = [preFun.remove_numbers(i) for i in row]
        row = [word for word in row if word not in stop_words]
        row = [i for i in row if i != '']
        row = [stemmer.stem(word) for word in row]
        row = [lemmatizer.lemmatize(word, pos='v') for word in row]
        des_col_list.append(len(set(row)))

    x['Description'] = des_col_list
    return x


def developer_test(x):
    encoder = LabelEncoder()
    encoder.classes_ = np.load('Developer Encoder.npy', allow_pickle=True)
    x['Developer'] = x['Developer'].map(lambda s: 'unknown' if s not in encoder.classes_ else s)
    encoder.classes_ = np.append(encoder.classes_, 'unknown')
    x['Developer'] = encoder.transform(x['Developer'])
    return x


def avg_rating_test(x):
    age_avg = (4.0 + 3.0 + 2.0 + 1.0) / 4
    not_in_rate = ['4+', '9+', '12+', '17+']
    col = x['Age Rating']
    for i in range(len(col.index)):
        if col.iloc[i] not in not_in_rate:
            col.iloc[i] = age_avg

    x['Age Rating'] = col
    x.loc[x['Age Rating'].isin(['17+']), 'Age Rating'] = 4
    x.loc[x['Age Rating'].isin(['12+']), 'Age Rating'] = 3
    x.loc[x['Age Rating'].isin(['9+']), 'Age Rating'] = 2
    x.loc[x['Age Rating'].isin(['4+']), 'Age Rating'] = 1

    x['Age Rating'] = x['Age Rating'].astype(np.int64)
    return x


def languages_test(x):
    languages = []
    for row in x['Languages']:
        # print(len(row.split()))
        languages.append(len(row.split(', ')))

    x['Languages'] = languages
    return x


def primary_test(x):
    x = x.drop(['Primary Genre'], axis=1)
    return x


def genres_test(x):
    # TODO: Load all genres and their frequency data
    genres = pd.read_csv('Genres.csv')

    with open('Genres_Frequency.pkl', 'rb') as f:
        genres_freq = pickle.load(f)

    genres_freq.loc['unknown'] = 0.0  # adding a row
    # print('genres Frequency : \n', genresFreq)

    test_rows = []
    # print('genres', genres)
    for row in x['Genres']:
        genre_row = [check_existence(gen, genres) for gen in row.split(', ')]
        # print(genreRow)
        test_rows.append(genres_freq[genre_row].sum())

    # print(x['Genres'])
    x['Genres'] = test_rows
    # print(x['Genres'])
    return x


def dates_test(x):
    # print(X_test['Original Release Date'])
    x['Original Release Date'] = pd.to_datetime(x['Original Release Date'], dayfirst=True)
    x['Original Release Year'] = x['Original Release Date'].dt.year
    x['Original Release Month'] = x['Original Release Date'].dt.month
    x['Original Release Day'] = x['Original Release Date'].dt.day
    # print(X_test['Original Release Year'])
    # print(X_test['Original Release Month'])
    # print(X_test['Original Release Day'])
    x = x.drop(['Original Release Date'], axis=1)

    x['Current Version Release Date'] = pd.to_datetime(x['Current Version Release Date'], dayfirst=True)
    x['Current Version Release Year'] = x['Current Version Release Date'].dt.year
    x['Current Version Release Month'] = x['Current Version Release Date'].dt.month
    x['Current Version Release Day'] = x['Current Version Release Date'].dt.day
    x = x.drop(['Current Version Release Date'], axis=1)

    return x


def check_existence(gen, genres):
    if gen in genres.values:
        return gen
    else:
        return 'unknown'
