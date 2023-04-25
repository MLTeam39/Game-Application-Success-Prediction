import joblib
import numpy as np
import preprocessingFunctions as preFun


def drop_test(x):
    dropped = preFun.droppedCols
    x = x.drop(dropped, axis=1)
    return x


def scaler(x, col_name):
    reshaped_test_col = np.array(x[col_name]).reshape(-1, 1)
    scaler_path = 'preprocessingData\\' + col_name + ' Scaler.gz'
    scaler = joblib.load(scaler_path)
    x[col_name] = scaler.transform(reshaped_test_col)
    return x