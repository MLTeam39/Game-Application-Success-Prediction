import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
import statistics

from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from scipy import stats


# Loading data
data = pd.read_csv('games-regression-dataset.csv')
X = data.drop(["Average User Rating"], axis=1)  # features
y = data["Average User Rating"]  # label
# print(X)

# Split the data to training and testing sets
y = np.array(y)
y = y.reshape((X.shape[0], 1))
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
PriceCol = X_train['Price']
#check Missing values of Price and Average User Rating columns
print('num of missing values in Price:',PriceCol.isna().sum())
print('num of missing values in Y :', data['Average User Rating'].isna().sum())
# make feature scaling to Price column
ResPrice = (PriceCol - PriceCol.min())/(PriceCol.max()-PriceCol.min())
X_train['Price'] = ResPrice
# print('----------------------------')##
#---------------------------------------------------------------------
#check if IconURL and URL are unique
ResIconURL = X_train['Icon URL'].nunique(dropna=False)
ResURL = X_train['URL'].nunique(dropna=False)
# print("length of unique Iconvalues :" , ResIconURL, "\n percentage  : ",(ResIconURL/len(data['IconURL']) *100))
# print("length of unique URLvalues :" , ResURL, "\n percentage : " , (ResURL/len(data['URL'])*100))


"""count duplicates in Names"""
print(X_train)
#Name
len(X_train['Name']) - len(X_train['Name'].drop_duplicates())
print(X_train)

#ID
len(X_train['ID']) - len(X_train['Name'].drop_duplicates())
print(X_train)
#Dropping features
X_train['NewAgeRate'] = X_train['Age Rating']
X_train = X_train.drop(X_train.columns[[0, 1, 2, 4, 10]], axis=1) #droped columns
print(X_train)
"""Age Rating"""
X_train.loc[X_train['NewAgeRate'].isin(['17+']), 'NewAgeRate'] = '3'
X_train.loc[X_train['NewAgeRate'].isin(['12+']), 'NewAgeRate'] = '2'
X_train.loc[X_train['NewAgeRate'].isin(['9+']), 'NewAgeRate'] = '1'
X_train.loc[X_train['NewAgeRate'].isin(['4+']), 'NewAgeRate'] = '0'
"""counts nulls"""
counts = X_train.isna().sum()

# Deal with missing values
subtitleNull = X_train['Subtitle'].isna().sum() / len(X_train['Subtitle']) * 100
developerNull = X_train['Developer'].isna().sum() / len(X_train['Developer']) * 100
print("Subtitle nulls percentage = ", "%.2f" % subtitleNull, "%")
print("Developer nulls percentage = ", "%.2f" % developerNull, "%")

X_train = X_train.drop(['Subtitle'], axis=1)
# Separate features from labels


# Check Unique percentage
DescriptionColUni = set(X_train['Description'])
print("Description unique percentage = ", "%.2f" % (len(DescriptionColUni) / len(X_train['Description'])), "%")
print(DescriptionColUni)
X_train = X_train.drop(['Description'], axis=1)
# print(Y)

# Feature Encoding
print(X_train['Developer'])

enc = LabelEncoder()
enc.fit(list(X_train['Developer'].values))
X_train['Developer'] = enc.transform(list(X_train['Developer'].values))


print(X_train['Developer'])

Size_new=pd.read_csv("games-regression-dataset.csv",usecols=['Size'])

tempFeat = np.array(Size_new['Size']).reshape(-1, 1)
scaler = MaxAbsScaler()
scaler.fit(tempFeat)
scaledFeat = scaler.transform(tempFeat)
Size_new['Size'] = scaledFeat.reshape(1, -1)[0]
data_cpycpy = X_train.drop(['Size'], axis=1)
X_train = pd.DataFrame(data_cpycpy)
X_train['Size'] = Size_new['Size']

##########################################
##'Primary Genre'
##'Genres'
PrimaryGenre = pd.read_csv("games-regression-dataset.csv", usecols=['Primary Genre'])
Genres = pd.read_csv("games-regression-dataset.csv", usecols=['Genres'])

count = 0

for i in range(len(PrimaryGenre.index)):
    if (
    PrimaryGenre.iloc[i].to_string().split(' ')[5].__eq__((Genres.iloc[i].to_string()).split(', ')[0].split(' ')[4])):
        count = count + 1

flag = "%.2f" % (count / len(PrimaryGenre) * 100)

if (float(flag) >= 99):
    X_train = X_train.drop(['Primary Genre'], axis=1)

##########################################
##'Genres'

output = X_train['Genres'].str.get_dummies(sep=', ')

# data=pd.concat(output)
# data=data(output.columns.values.tolist())

# data["Action","Adventure","Board","Books","Business","Card","Casino","Casual","Education","Entertainment" ," Reference","Role", "Playing","Simulation", "Social Networking","Sports","Strategy","Travel","Trivia","Utilities","Word"]=output


print(output)

# Missing Values
print(X_train['Original Release Date'].isnull().sum())  # there is no missing value
print(X_train['Current Version Release Date'].isnull().sum())  # there is no missing value
print(X_train['Languages'].isnull().sum())  # there is 11 missing value

X_train['Languages'].value_counts()  # most_frequent is'EN'=2728
most_frequent = 'EN'
data['Languages'].fillna(most_frequent, inplace=True)

# Reformate Date
X_train['Original Release Date'] = pd.to_datetime(X_train['Original Release Date'], dayfirst=True)
X_train['Original Release Year'] = X_train['Original Release Date'].dt.year
X_train['Original Release Month'] = X_train['Original Release Date'].dt.month
X_train['Original Release Day'] = X_train['Original Release Date'].dt.day
X_train['Original Release Year'] = X_train['Original Release Year'].astype(float)
X_train['Original Release Month'] = X_train['Original Release Month'].astype(float)
X_train['Original Release Day'] = X_train['Original Release Day'].astype(float)
# print(data['Original Release Date'].corr(data['Average User Rating']))

X_train['Current Version Release Date'] = pd.to_datetime(X_train['Current Version Release Date'], dayfirst=True)
X_train['Current Version Release Year'] = X_train['Current Version Release Date'].dt.year
X_train['Current Version Release Month'] = X_train['Current Version Release Date'].dt.month
X_train['Current Version Release Day'] = X_train['Current Version Release Date'].dt.day
X_train['Current Version Release Year'] = X_train['Current Version Release Year'].astype(float)
X_train['Current Version Release Month'] = X_train['Current Version Release Month'].astype(float)
X_train['Current Version Release Day'] = X_train['Current Version Release Day'].astype(float)
# print(data['Original Release Date'].corr(data['Average User Rating']))

# Languages
i = 0
for row in X_train['Languages']:
    X_train.at[i, 'Num_Languages'] = len(row.split())
    i += 1
X_train['Num_Languages'] =  X_train['Num_Languages'].astype(int)

X_train = X_train.drop(['Languages'], axis=1)
X_train = X_train.drop(['Current Version Release Date'], axis=1)
X_train = X_train.drop(['Original Release Date'], axis=1)
# data pre-processing
data_cpy = X_train



# input data


print(data_cpy.describe())
df_data_cpy = pd.DataFrame(data_cpy)
from matplotlib import pyplot as plt
from scipy.stats import norm
import matplotlib


print(df_data_cpy['In-app Purchases'].isna().sum() / len(df_data_cpy['In-app Purchases'].index))
df_data_cpy['In-app Purchases'].fillna(df_data_cpy['In-app Purchases'].mode()[0], inplace=True)

tempFeat = np.array(df_data_cpy['User Rating Count']).reshape(-1, 1)
scaler = MaxAbsScaler()
scaler.fit(tempFeat)
scaledFeat = scaler.transform(tempFeat)
df_data_cpy['User Rating Count'] = scaledFeat.reshape(1, -1)[0]
print(df_data_cpy['User Rating Count'])

splitted = []
out = []
out2 = []
for i in df_data_cpy['In-app Purchases']:
    splitted = i.split(', ')

    for item in splitted:
        out.append(float(item))

    out2.append(statistics.mean(out))

data_cpycpy = df_data_cpy.drop(['In-app Purchases'], axis=1)
df_data_cpycpy = pd.DataFrame(data_cpycpy)
df_data_cpycpy['In-app Purchases'] = out2

matplotlib.rcParams['figure.figsize'] = (10,6)
plt.hist(df_data_cpycpy['In-app Purchases'],bins=50,rwidth=.8)

plt.xlabel('in app purchases')
plt.ylabel('Count')
plt.show()



print(df_data_cpycpy['In-app Purchases'].isna().sum())

print(X_train['User Rating Count'].corr(data['Average User Rating']))


df_data_cpycpy.to_csv('preProc1.csv', index=False)