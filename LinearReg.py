import numpy as np # for numerical operations
import pandas as pd # for handling input data
import matplotlib.pyplot as plt # for data visualization 
"""import seaborn as sns # for data visualization """
import timeit
import sklearn 
import random
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
"""from Pre_processing import *"""

from sklearn.preprocessing import LabelEncoder
import numpy as np

data = pd.read_csv("C:/Users/Tech/Downloads/Lab3/games-regression-dataset.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

"""Linear reg """


model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X_test)
r_sq = model.score(X, y)
print(f"coefficient of determination: {r_sq}")
"""multiple reg"""

regr = linear_model.LinearRegression()
regr.fit(X, y)
predicted = regr.predict(X_test)
# compare between expected and trained answers
