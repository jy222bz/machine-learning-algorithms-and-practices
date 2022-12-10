# Import the libs
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# Import the dataset
dataset = pd.read_csv('datasets/2_regression/multiple_linear_regression/50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print('The x-values before encoding the categorical data!')
print(x)
print('\n')

# Encoding the categorical data
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print('The x-values after encoding the categorical data!')
print(x)
print('\n')

# No need to apply feature scaling in multiple linear regression


# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)


# Training the model
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# Predicting the Test set results and comparing it with the test values!
y_prediction = regressor.predict(x_test)
np.set_printoptions(precision=2)
print('The first column is the predicted profits, while the second column is the actual profits!')
print(np.concatenate((y_prediction.reshape(len(y_prediction), 1), y_test.reshape(len(y_prediction), 1)), 1))
print('\n')



