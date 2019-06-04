"""
K nearest neighbor is an algorithm where first decide the k value which is the number of neighbors that you will use for prediction
and you will calculate eucledian distance.
Now let's the data for which you want to predict the class be A.

Now all we have to do is find the distance between k and rest of the points and sort the distance and slice it to k values.
Now of the k values the class with the highest number in the k values will be our prediction

As there is no training part it isk easier to deploy but is not efficient when data is very big as you have to calculate distances
for all the points.But it can be multi threaded to make it much faster

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

style.use("fivethirtyeight")

dataset = {"k": [[1, 2], [2, 3], [3, 4]], "r": [[6, 5], [7, 8], [8, 9]]}

new_features = [5, 7]


def k_nearest_neighbor(data, predict, k):
    if len(data) >= k:
        warnings.warn("K should not be bigger than the length of the data")
    distances = []
    for group in data:
        for features in data[group]:
            eucledian_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([eucledian_distance, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k
    return vote_result, confidence


# result = k_nearest_neighbor(dataset, new_features, 3)
#
# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], color=result, marker='*')
# plt.show()


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

for group in test_set:
    for data in test_set[group]:
        vote, confidence = k_nearest_neighbor(train_set, data, 5)
        if group == vote:
            correct += 1
        total += 1
print("Accuracy : %0.4f " % (correct / total))

clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
clf.fit(X_train, y_train)
print("Accuracy from sklkearn : %0.4f " % (clf.score(X_test, y_test)))
