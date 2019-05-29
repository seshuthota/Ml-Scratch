import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
pd.set_option('display.max_seq_items', None)


class LinearRegression:

    def __init__(self, learning_rate, n_iter):
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def initialize_weights(self, X):
        self.weights = np.random.rand(X.shape[1], 1)
        self.bias = np.ones((1,))

    def fit(self, X_train, y_train):
        self.initialize_weights(X_train)

        self.x_mean = X_train.mean(axis=0).T
        self.x_stddev = X_train.std(axis=0).T
        self.y_mean = y_train.mean()
        self.y_stddev = y_train.std()

        X_train, y_train = self.normalize(X_train, y_train)

        for i in range(self.n_iter):
            pred = np.dot(X_train, self.weights) + self.bias
            diff = np.subtract(pred,y_train)

            # calculate dw, db
            delta_w = np.mean(diff * X_train, axis=0, keepdims=True).T
            delta_b = np.mean(diff)

            # update weights and biases
            self.weights = self.weights - self.learning_rate * delta_w
            self.bias = self.bias - self.learning_rate * delta_b
        return self

    def predict(self, X):
        X = self.normalize(X)
        preds = np.dot(X, self.weights) + self.bias
        return preds * self.y_stddev + self.y_mean

    def error(self, y_true, y_pred):
        diff = y_true - y_pred
        return np.mean(diff ** 2)

    def normalize(self, X, y = None):
        X = (X - self.x_mean) / self.x_stddev

        if y is None:
            return X

        y = ((y - self.y_mean) / self.y_stddev).reshape(len(y), 1)
        return X, y


lr = LinearRegression(1e-3, 1000)

data = datasets.load_boston()
X = data["data"]
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

lr.fit(X_train, y_train)

y_pred_train = lr.predict(X_train)

y_pred_test = lr.predict(X_test)

train_error = lr.error(y_true=y_train, y_pred= y_pred_train)
test_error = lr.error(y_true=y_test, y_pred = y_pred_test)

print("Error on train : %0.3f and test : %0.3f " %(train_error, test_error))

