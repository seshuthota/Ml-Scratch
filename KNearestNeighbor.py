"""
K nearest neighbor is an algorithm where first decide the k value which is the number of neighbors that you will use for prediction
and you will calculate eucledian distance.
Now let's the data for which you want to predict the class be A.

Now all we have to do is find the distance between k and rest of the points and sort the distance and slice it to k values.
Now of the k values the class with the highest number in the k values will be our prediction

As there is no training part it isk easier to deploy but is not efficient when data is very big as you have to calculate distances
for all the points.But it can be multi threaded to make it much faster

"""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

style.use("fivethirtyeight")

dataset = {"k": [[1, 2], [2, 3], [3, 4]], "r": [[6, 5], [7, 8], [8, 9]]}

new_features = [5, 7]


class K_nearest_neighbor:

    def __init__(self, k):
        self.k = k
        self.train_set = {}

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        for _ in np.unique(y_train):
            self.train_set[_] = []
        for i in range(len(X_train)):
            self.train_set[y_train[i]].append(X_train[i])

    def predict(self, X_test):
        distances = []
        for group in self.train_set:
            for features in self.train_set[group]:
                eucledian_distance = np.linalg.norm(np.array(features) - np.array(X_test))
                distances.append([eucledian_distance, group])
        votes = [i[1] for i in sorted(distances)[:self.k]]
        vote_result = Counter(votes).most_common(1)[0][0]
        confidence = Counter(votes).most_common(1)[0][1] / self.k
        return vote_result, confidence

    def accuracy(self, X_test, y_test):
        test_set = {}
        for _ in np.unique(y_test):
            test_set[_] = []
        for i in range(len(y_test)):
            test_set[y_test[i]].append(X_test[i])

        vote_result_list = []
        confidence_list = []
        correct = 0.0
        total_count = 0.0
        for _ in test_set:
            for features in test_set[_]:
                vote_result, confidence = self.predict(features)
                if vote_result == _:
                    correct += 1
                total_count += 1
                vote_result_list.append([vote_result, confidence])
                confidence_list.append(confidence)
        accuracy = correct / total_count
        return accuracy


breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

train_set = {1: [], 0: []}
test_set = {1: [], 0: []}

for _ in range(len(y_train)):
    train_set[y_train[_]].append(X_train[_])

for _ in range(len(y_test)):
    test_set[y_test[_]].append(X_test[_])

correct = 0
total = 0.0
k = [int(x) for x in range(1, 51)]
Accuracy = []
for i in k:
    clf2 = K_nearest_neighbor(k=i)
    clf2.train(X_train, y_train)
    accuracy = clf2.accuracy(X_test, y_test)
    Accuracy.append(accuracy)
    print("Accuracy from KNN for {} is {} ".format(i, accuracy))

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
print("Accuracy from sklkearn : %0.4f " % (clf.score(X_test, y_test)))
plt.plot(k, Accuracy)
plt.show()
