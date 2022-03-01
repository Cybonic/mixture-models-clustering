import imageio
import matplotlib.animation as ani
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
from PIL import Image
from sklearn import datasets
from sklearn.cluster import KMeans

import gmm



iris = datasets.load_iris()
X = iris.data
X[:20]


x0  = np.array([[0.05, 1.413, 0.212], [0.85, -0.3, 1.11], [11.1, 0.4, 1.5], [0.27, 0.12, 1.44], [88, 12.33, 1.44]])
mu  = np.mean(x0, axis=0)
cov = np.dot((x0 - mu).T, x0 - mu) / (x0.shape[0] - 1)

y = gmm.gaussian(x0, mu=mu, cov=cov)
y


n_clusters = 3
n_epochs = 50

clusters, likelihoods, scores, sample_likelihoods, history = gmm.train_gmm(X, n_clusters, n_epochs)

gmm.create_cluster_animation(X, history, scores)

plt.figure(figsize=(10, 10))
plt.title('Log-Likelihood')
plt.plot(np.arange(1, n_epochs + 1), likelihoods)
plt.show()



