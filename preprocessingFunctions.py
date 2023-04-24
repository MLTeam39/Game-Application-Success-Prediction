import numpy as np
from sklearn.preprocessing import MaxAbsScaler

droppedCols = []


def count_nulls(col, col_name):
    null_percentage = col.isna().sum() / len(col) * 100
    print(col_name, "Null Percentage =", "%.2f" % null_percentage, "%")
    return null_percentage


def count_unique(col, col_name):
    unique_val = col.nunique(dropna=False)
    unique_percentage = unique_val / len(col) * 100
    print(col_name, "Unique Percentage = ", "%.2f" % unique_percentage, "%")
    return unique_percentage


def feature_selection(x):
    for col_name in x:
        nulls_percentage = count_nulls(x[col_name], col_name)
        unique_percentage = count_unique(x[col_name], col_name)

        if nulls_percentage >= 70:
            x = x.drop(col_name, axis=1)
            droppedCols.append(col_name)
            print('Dropped!')

        elif unique_percentage >= 98:
            x = x.drop(col_name, axis=1)
            droppedCols.append(col_name)
            print('Dropped!')

        print('\n')

    print('Dropped Columns : ', droppedCols, '\n')
    return x


# def feature_scaling(cols, col_name):
#     temp_feat = np.array(cols[col_name]).reshape(-1, 1)
#     # TODO: Check scaling range
#     scaler = MaxAbsScaler()
#     scaler.fit(temp_feat)
#     scaled_feat = scaler.transform(temp_feat)
#     cols[col_name] = scaled_feat.reshape(1, -1)[0]
#     # TODO: Scaling from 0 -> 1 or -1 -> 1
#     # ResPrice = (PriceCol - PriceCol.min()) / (PriceCol.max() - PriceCol.min())
#     # X_train['Price'] = ResPrice
#     return cols


def feature_scaling(x_train, x_test, col_name):
    # print('before reshaping : \n', x_train[col_name], '\n')
    reshaped_train_col = np.array(x_train[col_name]).reshape(-1, 1)
    # print('after reshaping : \n', reshaped_train_col, '\n')
    # TODO: Check scaling range
    scaler = MaxAbsScaler()
    scaled_train_col = scaler.fit_transform(reshaped_train_col)
    # scaler.fit(reshaped_train_col)
    # scaled_feat = scaler.transform(reshaped_train_col)
    print('before reshaping : \n', scaled_train_col, '\n')
    x_train[col_name] = scaled_train_col.reshape(1, -1)[0]
    print('after reshaping : \n', x_train[col_name], '\n')
    # TODO: Scaling from 0 -> 1 or -1 -> 1
    # ResPrice = (PriceCol - PriceCol.min()) / (PriceCol.max() - PriceCol.min())
    # X_train['Price'] = ResPrice
    return x_train