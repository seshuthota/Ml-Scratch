"""
Linear Regression it assumes a linear relationship between dependent variable(y) and independent variable(X).

So the value of y can be calculated  using a linear combination of input variable X.

Very similar to how the equation of a straight line takes a linear relation between x-axis and y-axis as

y = m(x) + c

where:
    m -> slope of the line
    c -> y-intercept

we can also write equation for predicting value as

y = b0 + b1 * X

where b0 and b1 are the coefficients that we need to estimate from the training data.

Steps:
    1. Calculate the Mean and Variance
    2. Calculate covariance
    3. Estimate coefficients
    4. Make predictions 
    5. Evaluate algorithm
"""
# imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

random.seed(1)


# plt.scatter(X, y)
# plt.show()
# after looking at the scatter plot you can see there is a linear relation between X and  y.


# Step - 1 : Calculating Mean and variance of both X and y.

def mean(values):
    return sum(values) / float(len(values))


def variance(values, mean):
    return sum((np.array(values) - mean) ** 2) / float(len(values))


# Step -2 : Calculating Covariance of X and y

def covariance(x, mean_x, y, mean_y):
    x = np.array(x)
    y = np.array(y)
    return sum((x - mean_x) * (y - mean_y)) / float(len(x))


# Step - 3 :  Estimate coefficients
def coefficient(X, y):
    mean_x, mean_y = mean(X), mean(y)
    b1 = covariance(X, mean_x, y, mean_y) / variance(X, mean_x)
    b0 = mean_y - b1 * mean_x
    return [b0, b1]


def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)


# Step - 4 : Make predictions
def simple_linear_regression(X_train, y_train, X_test):
    b0, b1 = coefficient(X_train, y_train)
    predictions = b0 + b1 * X_test
    return predictions


def rmse_metric(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    prediction_error = actual - predicted
    rmse = np.sqrt(sum(prediction_error ** 2) / float(len(actual)))
    return rmse


def evaluate_algorithm(X, y, algorithm, split_size):
    X_train, X_test, y_train, y_test = split_data(X, y, split_size)
    predicted = algorithm(X_train, y_train, X_test)
    rmse = rmse_metric(y_test, predicted)
    return rmse


#
# print("X-stats : mean= %0.3f variance= %0.3f " % (mean_x, var_x))
# print("y-stats : mean= %0.3f variance= %0.3f " % (mean_y, var_y))
# print("Covariance of X, y : %0.3f" % (covarxy))
# print("coefficients  b0=%0.3f,. b1=%0.3f " % (b0, b1))

df = pd.read_excel(io="insurance.xls", encoding='ascii')
X = df["X"]
y = df["Y"]
test_size = 0.2

b0, b1 = coefficient(X, y)
y_hat = b0 + b1 * X

# plt.scatter(X, y, label="Data points")
# plt.plot(X, y_hat, color='#00ff00', label="LinearRegression")
# plt.legend()
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


def rSquare(X_train, X_test, y_train, y_test, algorithm):
    predicted = algorithm(X_train, y_train, X_test)
    y_mean_line = mean(y_test)
    squared_error_regr = sum((predicted - y_test)**2)
    squared_error_ymean = sum((y_test - y_mean_line)**2)
    return 1 - squared_error_regr / squared_error_ymean


rmse = evaluate_algorithm(X, y, simple_linear_regression, test_size)
print("RMSE from Math : %0.3f" % (rmse))
print("Rsquare error from Math: %0.3f " % (rSquare(X_train, X_test, y_train, y_test, simple_linear_regression)))

lr = LinearRegression()
lr.fit(np.array(X_train).reshape(-1, 1), y_train)
y_pred = lr.predict(np.array(X_test).reshape(-1, 1))
print("RMSE from sklearn ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Rsquare error : ",r2_score(y_test, y_pred))