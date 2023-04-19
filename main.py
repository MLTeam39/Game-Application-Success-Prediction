import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
import statistics
from sklearn.preprocessing import MaxAbsScaler
from scipy import stats


# Loading data
data = pd.read_csv('games-regression-dataset.csv')

# data pre-processing
data_cpy = data

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

print(data['User Rating Count'].corr(data['Average User Rating']))
