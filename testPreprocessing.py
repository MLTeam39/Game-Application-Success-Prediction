import joblib
import statistics
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

droppedCols = []
filledColumns = {}


def drop_test(x):
    # print(droppedCols)
    x = x.drop(droppedCols, axis=1)
    return x


def fill_nulls(x):
    # print(filledColumns)
    for colName in filledColumns.keys():
        x[colName].fillna(filledColumns[colName], inplace=True)
    return x


def scaler(x, col_name):
    reshaped_test_col = np.array(x[col_name]).reshape(-1, 1)
    scaler_path = col_name + ' Scaler.gz'
    scaler = joblib.load(scaler_path)
    x[col_name] = scaler.transform(reshaped_test_col)
    return x


def inApp_test(x):
    testRow = []
    testCol = []
    for test in x['In-app Purchases']:
        testRow = [float(v) for v in test.split(', ')]
        testCol.append(statistics.mean(testRow))

    x['In-app Purchases'] = testCol
    return x


def developer_test(x):
    encoder = LabelEncoder()
    encoder.classes_ = np.load('Developer Encoder.npy', allow_pickle=True)
    x['Developer'] = x['Developer'].map(lambda s: 'unknown' if s not in encoder.classes_ else s)
    encoder.classes_ = np.append(encoder.classes_, 'unknown')
    x['Developer'] = encoder.transform(x['Developer'])
    return x


def avgRating_test(x):
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


def genres_test(x, genresFreq, genres):
    genresFreq.loc['unknown'] = 0.0  # adding a row
    # print('genres Frequency : \n', genresFreq)

    testRows = []
    # print('genres', genres)
    for row in x['Genres']:
        genreRow = [checkExistence(gen, genres) for gen in row.split(', ')]
        # print(genreRow)
        testRows.append(genresFreq[genreRow].sum())

    # print(x['Genrfes'])
    x['Genres'] = testRows
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


def checkExistence(gen, genres):
    if gen in genres.values:
        return gen
    else:
        return 'unknown'