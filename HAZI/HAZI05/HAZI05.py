import pandas as pd
import seaborn as sns
from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix, euclidean_distances

class KNNClassifier:

    @property
    def k_neighbors(self) -> int:
        return self.k
    
    def __init__(self, k:int, test_split_ratio:float):
        self.k = k
        self.test_split_ratio = test_split_ratio
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    @staticmethod
    def load_csv(csv_path: str):
        df = pd.read_csv(csv_path, header=None)
        df = df.sample(frac=1, random_state=42)
        X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
        return X, y
    
    def train_test_split(self, features:pd.DataFrame, labels:pd.Series) -> None:
        test_size = int(len(features) * self.test_split_ratio)
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size, "Size mismatch!"

        self.X_train, self.y_train = features.iloc[:train_size,:], labels.iloc[:train_size]
        self.X_test, self.y_test = features.iloc[train_size:train_size+test_size,:], labels.iloc[train_size:train_size + test_size]
    
    def euclidean(self, element_of_x: pd.Series) -> pd.Series:
        return ((self.X_train - element_of_x)**2).sum(axis=1).apply(lambda x: x**0.5)

    
    def predict(self, x_test: pd.DataFrame) -> pd.Series:
        labels_pred = []

        for index, x_test_element in x_test.iterrows():
            distances = self.euclidean(x_test_element)
            distances = pd.concat([distances, self.y_train], axis=1).sort_values(by=0)
            label_pred = mode(distances.iloc[:self.k, 1])[0][0]
            labels_pred.append(label_pred)

        self.y_preds = pd.Series(labels_pred, dtype=int)

        return self.y_preds

    
    def accuracy(self) -> float:
        true_positive = (self.y_test == self.y_preds).sum()
        return true_positive / len(self.y_test) * 100

    def plot_confusion_matrix(self) -> pd.DataFrame:
        conf_matrix = pd.crosstab(index=self.y_test, columns=self.y_preds, rownames=['True'], colnames=['Predicted'])
        return conf_matrix.values
