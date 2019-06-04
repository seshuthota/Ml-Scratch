"""
Unlike Linear Regression which is used for continuous dependent variable, Logistic regression is used for discrete dependent
variable.As in Logistic regression the dependent variable can take limited values only which means the dependent variable is
categorical.And if the number of possible outcomes is only 2 then it is called Binary Logistic Regression.


In Linear Regression the output is the weighted sum of the inputs.Logistic Regression is a generalized Linear Regression
in the sense that we don't output the weighteed sum, Instead we pass it through a function called sigmoid which will output
only in range [0, 1] for any real value as input.

If we take weighted sum then the output value varies in a wide range, which is why we cannot use it for classification.

Logistic Regression makes an assumption that
1. Dependent variable must be categorical
2. The independent variables must be independent of each other to avoid Multicollinearity.

"""

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from scipy.special import expit


# breast_cancer = datasets.load_breast_cancer()
# X = breast_cancer.data
# y = breast_cancer.target


class Logistic_Regression:

    def __init__(self, learning_rate=1e-2, n_iters=1000):
        self.n_iters = n_iters
        self.learning_rate = learning_rate

    def predict(self, X):
        X = self.normalize(X)
        linear = self.hypothesis(X)
        preds = self.sigmoid(linear)
        return (preds >= 0.5).astype(int)

    def sigmoid(self, z):
        return expit(
            z)  # using this instead of normal numpy exp as you will get  'RuntimeWarning: overflow encountered in exp" with np.exp

    def hypothesis(self, X):
        return np.dot(X, self.weights) + self.bias

    def initialize_weights(self, X):
        self.weights = np.random.rand(X.shape[1], 1)
        self.bias = np.zeros((1,))

    def fit(self, X_train, y_train):
        self.initialize_weights(X_train)
        self.x_mean = X_train.mean(axis=0).T
        self.x_stddev = X_train.std(axis=0).T

        X_train = self.normalize(X_train)
        for i in range(self.n_iters):
            probs = self.sigmoid(self.hypothesis(X_train))
            diff = probs - y_train
            # calculating dw, db
            dw = np.dot(X_train.T, diff) / (X_train.shape[0])
            db = np.mean(diff)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
        return self

    def normalize(self, X):
        return (X - self.x_mean) / self.x_stddev

    def accuracy(self, X, y):
        predict = self.predict(X)
        return np.mean(predict == y)

    def loss(self, X, y):
        probs = self.sigmoid(self.hypothesis(X))
        # loss when y is positive
        pos_log = y * np.log(probs + 1e-15)
        # loss when y is negative
        neg_log = (1 - y) * np.log((1 - probs) + 1e-15)
        return -np.mean(pos_log + neg_log)


data = pd.read_csv("heart.csv")
X = data.drop('target', 1)
y = np.array(data["target"]).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

lr = Logistic_Regression()
lr.fit(X_train, y_train)

print("Accuracy of training set : ", lr.accuracy(X_train, y_train))
print("Loss of training set : ", lr.loss(X_train, y_train))
print("Accuracy of test set : ", lr.accuracy(X_test, y_test))
print("Loss of test set : ", lr.loss(X_test, y_test))

# lr1  = LogisticRegression(solver='lbfgs')
# lr1.fit(X_train, y_train)
# print(lr1.score(X_train, y_train))
