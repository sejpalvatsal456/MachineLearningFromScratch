import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
  def __init__(self):
    self.weights = None
    self.bias = None

  def fit(self, X, y, lr, epochs):
    n_sample, n_features = X.shape
    self.weights = np.zeros((n_features, 1))
    self.bias = 0
    for _ in range(epochs):
      y_pred = (np.dot(X, self.weights) + self.bias).reshape(-1, 1)
      error = np.sum((y-y_pred)**2)/n_sample
      print(error)
      dw = 2*np.dot(X.T, (y_pred-y).reshape(-1,1))/n_sample
      db = 2*np.sum(y_pred-y)/n_sample
      self.weights = self.weights - lr*dw
      self.bias = self.bias - lr*db
    self.y_pred_ = y_pred
  
  def predict(self, X):
    return np.dot(X, self.weights) + self.bias

X_simple = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y_simple = np.array([2,4,5,4,5,7,8,9,10,12]).reshape(-1,1)


slr = LinearRegression()
slr.fit(X_simple, y_simple, 0.002, 1000)

plt.scatter(X_simple, y_simple, color='blue', label='Data')
plt.plot(X_simple, slr.y_pred_, color='red', label='Regression Line')
plt.title("Simple Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()