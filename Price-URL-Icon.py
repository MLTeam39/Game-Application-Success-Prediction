import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from  sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler
data = pd.read_csv("games-regression-dataset.csv")
PriceCol = X['Price']
#check Missing values of Price and Average User Rating columns
print('num of missing values in Price:',PriceCol.isna().sum())
print('num of missing values in Y :',data['Average User Rating'].isna().sum())
# make feature scaling to Price column
ResPrice = (PriceCol - PriceCol.min())/(PriceCol.max()-PriceCol.min())
data['Price'] = ResPrice
# print('----------------------------')##
#---------------------------------------------------------------------
#check if IconURL and URL are unique
ResIconURL = data['Icon URL'].nunique(dropna=False)
ResURL = data['URL'].nunique(dropna=False)
# print("length of unique Iconvalues :" , ResIconURL, "\n percentage  : ",(ResIconURL/len(data['IconURL']) *100))
# print("length of unique URLvalues :" , ResURL, "\n percentage : " , (ResURL/len(data['URL'])*100))
#drop two columns
data = data.drop(data.columns[[0,4]], axis=1)
pd.set_option('display.max_columns', 18)

