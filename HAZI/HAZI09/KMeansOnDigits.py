from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from statistics import mode
import numpy as np
import seaborn as sns
import statistics


class KMeansOnDigits:
    def __init__(self, n_clusters=10, random_state=0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.digits = None
        self.model = None
        self.clusters = None
        self.labels = None
        self.accuracy = None
        self.mat = None

    def load_dataset(self):
        self.digits = load_digits()

    def predict(self):
        self.clusters = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit_predict(self.digits.data)
        return self.clusters

    def get_labels(self):
        self.labels = np.zeros_like(self.clusters)
        for i in range(10):
            mask = (self.clusters == i)
            self.labels[mask] = mode(self.digits.target[mask])[0]
        return self.labels

    def calc_accuracy(self, target_labels:np.ndarray,predicted_labels:np.ndarray):
        self.accuracy = accuracy_score(target_labels, predicted_labels)
        return round(self.accuracy, 2)

    def confusion_matrix(self):
        self.mat = confusion_matrix(self.digits.target, self.labels)
        return self.mat
