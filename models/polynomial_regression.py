# Import the libs
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



# Import the dataset
dataset = pd.read_csv('datasets/2_regression/polynomial_regression/Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# Training the Linear Regression model
linear_regressor = LinearRegression()
linear_regressor.fit(x, y)

# Training the Polynomial Regression model
polynomial_regressor = PolynomialFeatures(degree=4)
x_polynomial = polynomial_regressor.fit_transform(x)
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(x_polynomial, y)


# Visualising the Linear Regression model on the entire dataset
plt.scatter(x, y, color = 'red')
plt.plot(x, linear_regressor.predict(x), color = 'blue')
plt.title('True or False - Linear Regression Model!')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression model on the entire dataset
plt.scatter(x, y, color = 'red')
plt.plot(x, linear_regressor_2.predict(x_polynomial), color = 'blue')
plt.title('True or False - Polynomial Regression Model!')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


## Visualising the Polynomial Regression results (for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, linear_regressor_2.predict(polynomial_regressor.fit_transform(x_grid)), color = 'blue')
plt.title('True or False - Polynomial Regression!')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
print('Predicting a new result with Linear Regression!')
print(linear_regressor.predict([[6.5]]))
print('\n')


# Predicting a new result with Polynomial Regression
print('Predicting a new result with Polynomial Regression!')
print(linear_regressor_2.predict(polynomial_regressor.fit_transform([[6.5]])))
print('\n')