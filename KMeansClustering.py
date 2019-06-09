"""
This is an unsupervised machine learning algorithm. where we can use this algorithm to classify the unlabeled
data. Is is also referred to as the flat clustering .Where it will take the input data and think of of it as data in
high dimensional space.And will start with pre defined number of clusters we will start and initialize clusters  with
random locations. Now we will find Euclidean distance between clusters and the data points around it.And classify
each data point based on the closest centroid.And now calculate the new centroids based on the classified points and
iterate until we get to the point where the movement centroid from previous to the new is less than or equal to the
tolerance.


"""

import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
import numpy as np

style.use('ggplot')

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

colors = ["g", 'r', 'c', 'b', 'k']


class K_Means:

    def __init__(self, n_clusters=2, tolerance=1e-3, max_iter=300):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.max_iter = max_iter

    def fit(self, data):
        self.centroids = {}

        for i in range(self.n_clusters):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.n_clusters):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker='o', color='k', s=150, linewidths=5
                )
for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidths=5)

plt.show()

clf2 = KMeans(n_clusters=2, n_jobs=-1)
clf2.fit(X)

print("Cluster Centroids : ", clf.centroids)
print("Clusters Centroids  by Sklearn KMeans ", clf2.cluster_centers_)