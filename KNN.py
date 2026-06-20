import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def distance(point1, point2):
  return np.sqrt(np.sum(( np.array(point1) - np.array(point2) )**2))

def knn_predict(training_data, training_labels, test_point, k):
  distances = []
  for i in range(len(training_data)):
    d = distance(test_point, training_data[i])
    distances.append((d, training_labels[i]))
  distances.sort(key=lambda x: x[0])
  k_nearest = [label for _, label in distances[:k]]
  return Counter(k_nearest).most_common(1)[0][0]

training_data = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
training_labels = ['A', 'A', 'A', 'B', 'B']
test_point = [6, 6]
k = 3

X_A = []
X_B = []
Y_A = []
Y_B = []

for i in range(len(training_data)):    
  if training_labels[i] == 'A':
    X_A.append(training_data[i][0])
    Y_A.append(training_data[i][1])
  else:
    X_B.append(training_data[i][0])
    Y_B.append(training_data[i][1])

plt.scatter(X_A, Y_A, color='blue', label='A')
plt.scatter(X_B, Y_B, color='red', label='B')
test_class = knn_predict(training_data, training_labels, test_point, k)
plt.scatter(test_point[0], test_point[1], color=('blue' if test_class == 'A' else 'red'), label='Test')
plt.legend()
plt.show()