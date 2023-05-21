import joblib
import numpy as np
import pandas as pd
import testPreprocessing as testPre
import preprocessingFunctions as preFun
from nltk.tokenize import word_tokenize

"""Take user's to-be-tested data"""

print("Welcome to our Game Application Success Prediction Model")
print("Where We are using your game's information to predict it's Rate")
print("Please enter your game's:")

columns = ['URL', 'ID', 'Name', 'Subtitle', 'Icon URL', 'User Rating Count', 'Price', 'In-app Purchases', 'Description', 'Developer', 'Age Rating', 'Languages', 'Size', 'Primary Genre', 'Genres', 'Original Release Date', 'Current Version Release Date']

tst = []
for col in columns:
    colVal = input('{}: '.format(col))
    if colVal == '':
        colVal = None
    tst.append(colVal)

print(tst)
testData = pd.DataFrame(columns=columns)
testData.loc[len(testData.index)] = tst

print('Your Game\'s Data is:\n', testData)
print('Please Wait.....')

"""Preprocessing Test Data"""

# Drop columns
# print(testData.columns)
testData = testPre.drop_test(testData)
# print(testData.columns)

# Fill Nulls
# print(testData['Languages'].isnull().sum())
testData = testPre.fill_nulls(testData)
# print(testData['Languages'].isnull().sum())

# Apply training preprocessing on test data
testData = testPre.in_app_test(testData)
testData = testPre.description_text(testData)
testData = testPre.developer_test(testData)
testData = testPre.avg_rating_test(testData)
testData = testPre.languages_test(testData)
testData = testPre.primary_test(testData)
testData = testPre.genres_test(testData)
testData = testPre.dates_test(testData)

# Apply training scaler preprocessing on test data
testData = testPre.scaler_fun(testData, 'Description')
testData = testPre.scaler_fun(testData, 'Developer')
testData = testPre.scaler_fun(testData, 'Price')
testData = testPre.scaler_fun(testData, 'Size')
testData = testPre.scaler_fun(testData, 'User Rating Count')

"""Load Regression Models"""
linear_reg_model = joblib.load(open('Linear_Reg_Model', 'rb'))
multiple_reg_model = joblib.load(open('Multiple_Reg_Model', 'rb'))
polynomial_reg_model = joblib.load(open('Polynomial_Reg_Model', 'rb'))
gradient_reg_model = joblib.load(open('Gradient_Reg_Model', 'rb'))

"""Load Classification Models"""
decision_clf_model = joblib.load(open('Decision_Clf_Model', 'rb'))
knn_clf_model = joblib.load(open('KNN_Clf_Model', 'rb'))
svm_clf_model = joblib.load(open('SVM_Clf_Model', 'rb'))

"""Predict Test Data Result"""
print('Choose your Preferred Learning:\n1 -> Regression\n2 -> Classification')
learning_choice = input('Please Enter Your Choice (1 or 2): ')
if learning_choice == '1':
    # TODO: Loading...
    with open('Reg_Features.txt', 'r') as file:
        reg_features = [line.strip() for line in file.readlines()]
    file.close()

    with open('Indexing.txt', 'r') as file:
        indexing = [line.strip() for line in file.readlines()]
    file.close()

    """Linear Regression"""
    linear_test_col = testData['Current Version Release Year']
    linear_test_col = np.array(linear_test_col).reshape(1, -1)
    y_pred1 = linear_reg_model.predict(linear_test_col)

    testData = testData[reg_features]
    testData = testData.reindex(columns=indexing)
    testData = np.array(testData).reshape(-1, 1)
    # print('TEST DATA', testData, type(testData), testData.shape)

    y_pred2 = multiple_reg_model.predict(testData)
    y_pred2 = np.array([y for y in y_pred2 if y >= 0]).reshape(1, -1)
    y_pred3 = polynomial_reg_model.predict(testData)
    y_pred3 = np.array([y for y in y_pred3 if y >= 0]).reshape(1, -1)
    y_pred4 = gradient_reg_model.predict(testData)
    y_pred4 = np.array([y for y in y_pred4 if y >= 0]).reshape(1, -1)

    print('\nYour Game\'s Average User Rating using:')
    print('Linear Regression is:', y_pred1[0])
    print('Multiple Regression is:', y_pred2[0].tolist())
    print('Polynomial Regression is:', y_pred3[0].tolist())
    print('Gradient Regression is:', y_pred4[0].tolist())
    # print('Choose your Regression Model:\n1 -> Linear Regression\n2 -> Multiple Regression')
    # print('3 -> Polynomial Regression\n4 -> Gradient Regression')
    # reg_choice = input('Please Enter Your Choice (1, 2, 3 or 4):')

elif learning_choice == '2':
    # TODO: Loading...
    with open('Clf_Dropped_Features.txt', 'r') as file:
        clf_dropped_features = [line.strip() for line in file.readlines()]
    file.close()

    testData = testData.drop(clf_dropped_features, axis=1)
    d = {2: 'High', 1: 'Intermediate', 0: 'Low'}

    y_pred1 = decision_clf_model.predict(testData)
    y_pred2 = knn_clf_model.predict(testData)
    y_pred3 = svm_clf_model.predict(testData)

    print('\nYour Game\'s Rate using:')
    print('Decision Tree Classifier is:', y_pred1)
    print('KNN Classifier is:', y_pred2)
    print('SVM Classifier is:', y_pred3)
    # print('Choose your Classification Model:\n1 -> Decision Tree\n2 -> KNN\n3 -> SVM')
    # clas_choice = input('Please Enter Your Choice (1, 2 or 3):')
else:
    print('Invalid Choice, Try Again!')

