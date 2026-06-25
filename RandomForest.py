import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from DecisionTree import DecisionTree

def accuracy(y_test, y_pred):
  return np.sum(y_test == y_pred) / len(y_test)

class RandomForest:
  def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
    self.n_trees = n_trees
    self.max_depth = max_depth
    self.min_samples_split=min_samples_split
    self.n_features = n_features
    self.root = []
  
  def _bootstrap_samples(self, X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]
  
  def fit(self, X, y):
    self.trees = []
    for _ in range(self.n_trees):
      tree = DecisionTree(
        max_depth = self.max_depth,
        min_sample_split=self.min_samples_split,
        n_features=self.n_features
      )
      X_sample, y_sample = self._bootstrap_samples(X, y)
      tree.fit(X_sample, y_sample)
      self.trees.append(tree)
  
  def _most_common_label(self, y):
    counter = Counter(y)
    return counter.most_common(1)[0][0]
  
  def predict(self, X):
    predictions = [ tree.predict(X) for tree in self.trees ]
    tree_pred = np.swapaxes(predictions, 0, 1)
    predictions = [ self._most_common_label(pred) for pred in tree_pred ]
    return predictions
  
data = datasets.load_breast_cancer()
X, y = data.data, data.target
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

forest = RandomForest()
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)

print(
  accuracy(y_test, y_pred)
)