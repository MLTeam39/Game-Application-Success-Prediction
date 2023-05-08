import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Loading data
data = pd.read_csv('games-regression-dataset.csv')

# Deal with missing values
subtitleNull =  data['Subtitle'].isna().sum()/len(data['Subtitle'])*100
developerNull =  data['Developer'].isna().sum()/len(data['Developer'])*100
print("Subtitle nulls percentage = ", "%.2f" % subtitleNull, "%")
print("Developer nulls percentage = ", "%.2f" % developerNull, "%")

# Separate features from labels
X=data.iloc[:, 0:17]
Y=data['Average User Rating']

# Check Unique percentage
devColUni = set(X['Developer'])
print("Developer unique percentage = ", "%.2f" % (len(devColUni)/len(X['Developer'])), "%")
print(devColUni)
# print(Y)

# Feature Encoding
print(X['Developer'])

enc = LabelEncoder()
enc.fit(list(X['Developer'].values))
X['Developer'] = enc.transform(list(X['Developer'].values))


print(X['Developer'])

# def Feature_Encoder(X,cols):
#     for c in cols:
#         lbl = LabelEncoder()
#         lbl.fit(list(X[c].values))
#         X[c] = lbl.transform(list(X[c].values))
#     return X
