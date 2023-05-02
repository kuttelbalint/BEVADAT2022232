import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import sklearn.datasets as datasets


class KMeansOnDigits:
    def __init__(self, n_clusters, random_state):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.digits = None
        self.clusters = None
        self.labels = None
        self.accuracy = None
        self.mat = None

    def load_dataset(self):
        self.digits = datasets.load_digits()

    def predict(self):
        self.clusters = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit_predict(self.digits.data)
        return self.clusters

    def get_labels(self):
        self.labels = np.zeros_like(self.clusters)
        for i in range(10):
            mask = (self.clusters == i)
            self.labels[mask] = mode(self.digits.target[mask])[0]
        return self.labels

    def calc_accuracy(self):
        self.accuracy = accuracy_score(self.digits.target, self.labels)
        return round(self.accuracy, 2)

    def confusion_matrix(self):
        self.mat = confusion_matrix(self.digits.target, self.labels)
        return self.mat