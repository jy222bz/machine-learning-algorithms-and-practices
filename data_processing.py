from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# importing the data set
dataset = pd.read_csv('datasets/1_data_preprocessing/data.csv')
# this takes all the rows of all the columns -except the last column (-1 means the index of the last column).
x = dataset.iloc[:, :-1].values
# this takes all the rows of the last column, (-1 means the index of the last column).
y = dataset.iloc[:, -1].values

# printing the initial data
print(x)
print(y)

# processing missing data
# if the dataset is large then one can ignore the 1% missing data since it will not affect the
# the learning quality.

# it will replace all the missing values in the matrix by the mean itself.
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# it will compute the avearge and fit the missing values.
# it will exclude the columns with strings as it reads only numeric values.
imputer.fit(x[:, 1:3])

# it will apply the transformations and updates the columns.
x[:, 1:3] = imputer.transform(x[:, 1:3])


# printing the processed data
print(x)


# encoding Categorical data

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# printing the encoded data
print(x)

# encoding the dependent variable
le = LabelEncoder()
y = le.fit_transform(y)

# printing the encoded dependent variable
print(y)


# splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

# feature scaling - the goal is to have the values of the features in the same range.
# Do not apply the scaling on the dummy vars.
# standardisation: x_stand = x - mean(x) / standard deviation(x)
# will work all the time well.
# normalisation: x_norm = x - min(x) / max(x) - min(x)
# recommended for a normal distribution in most of the features.
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])
print(x_train)
print(x_test)
