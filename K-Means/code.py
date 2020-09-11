import numpy as np
import matplotlib.pyplot as plt

weights = np.array([0.20, 0.50, 0.30])
mu = np.array([[0, 0],
               [3, 0],
               [0, 3]])
sigma = np.array([[[1, 0],
                   [0, 1]],
                  [[1, 0],
                   [0, 1]],
                  [[1, 0],
                   [0, 1]]])

from sklearn import mixture
gmix = mixture.GaussianMixture(n_components=2, covariance_type='full')
gmix.weights_ = weights
gmix.means_ = mu
gmix.covariances_ = sigma

gmix.precisions_cholesky_ = 0
X, y = gmix.sample(500)
plt.scatter(X[:, 0], X[:, 1], s=10)


class K_Means:
    def __init__(self, k, max_iter):
        self.k = k
        self.max_iter = max_iter

    def fit(self, data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        L_list = []
        for i in range(self.max_iter):
            l = 0
            L = 0
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            for classification in self.classifications:
                for featureset in self.classifications[classification]:
                    l = np.linalg.norm(featureset - self.centroids[classification])
                    L += l

            L_list.append(L)

        return L_list

K_2 = K_Means(2, 20).fit(X)
K_3 = K_Means(3, 20).fit(X)
K_4 = K_Means(4, 20).fit(X)
K_5 = K_Means(5, 20).fit(X)

plt.figure(figsize=(15,15))
iter = np.arange(1, 21, 1)
plt.plot(iter, K_2, marker='o', label="K=2")
plt.plot(iter, K_3, marker='o', label="K=3")
plt.plot(iter, K_4, marker='o', label="K=4")
plt.plot(iter, K_5, marker='o', label="K=5")
plt.xlabel("Iteration")
plt.ylabel("L")
plt.title("Object function value per iteration")
plt.legend()
plt.show()

model = K_Means(3, 20)
model.fit(X)
colors = ["b", "g", "r", "c", "m", "k"]

plt.figure(figsize=(15,15))

for centroid in model.centroids:
    plt.scatter(model.centroids[centroid][0], model.centroids[centroid][1], marker="x", color="k", s=150)

for classification in model.classifications:
    color = colors[classification]
    for featureset in model.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], color = color, s=10, linewidths=5)

plt.title("K=3")
plt.show()

model = K_Means(5, 20)
model.fit(X)
colors = ["b", "g", "r", "c", "m", "k"]

plt.figure(figsize=(15,15))

for centroid in model.centroids:
    plt.scatter(model.centroids[centroid][0], model.centroids[centroid][1], marker="x", color="k", s=150)

for classification in model.classifications:
    color = colors[classification]
    for featureset in model.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], color = color, s=10, linewidths=5)

plt.title("K=5")
plt.show()