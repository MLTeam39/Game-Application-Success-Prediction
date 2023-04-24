# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 10:22:45 2023

@author: Tech
"""

import numpy as np
from sklearn.preprocessing import MaxAbsScaler


def count_nulls(col, col_name):
    null_percentage = col.isna().sum() / len(col) * 100
    print(col_name, "Null Percentage =", "%.2f" % null_percentage, "%")
    return null_percentage


def count_unique(col, col_name):
    unique_val = col.nunique(dropna=False)
    unique_percentage = unique_val / len(col) * 100
    print(col_name, "Unique Percentage = ", "%.2f" % unique_percentage, "%")
    return unique_percentage


def feature_selection(cols):
    for col in cols:
        nulls_percentage = count_nulls(cols[col], col)
        unique_percentage = count_unique(cols[col], col)

        if nulls_percentage >= 70:
            cols = cols.drop(col, axis=1)
            print('Dropped!')

        elif unique_percentage >= 98:
            cols = cols.drop(col, axis=1)
            print('Dropped!')

    print(cols.columns.values)
    return cols


def feature_scaling(x,y, col_name):
    temp_feat_x = np.array(x[col_name]).reshape(-1, 1)
    temp_feat_y = np.array(y[col_name]).reshape(-1, 1)
    # TODO: Check scaling range
    scaler = MaxAbsScaler()
    scaler.fit(temp_feat_x)
    scaled_feat_x = scaler.transform(temp_feat_x)
    scaled_feat_y = scaler.transform(temp_feat_y)
    x[col_name] = scaled_feat_x.reshape(1, -1)[0]
    y[col_name] = scaled_feat_y.reshape(1, -1)[0]
    # TODO: Scaling from 0 -> 1 or -1 -> 1
    # ResPrice = (PriceCol - PriceCol.min()) / (PriceCol.max() - PriceCol.min())
    # X_train['Price'] = ResPrice
    