#neural networks

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

#loaddataset
iris = load_iris()
X = iris.data
y = iris.target

#data spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=36)

#20 neurons in one hidden layer
model = MLPClassifier(hidden_layer_sizes=(20),max_iter=2000)

#training and prediction
model.fit(X_train,y_train)
predictions = model.predict(X_test)
print("Predictions:",predictions)

#evaluation
manual_accuracy = accuracy_score(y_test, predictions)
print("Manual Accuracy:", manual_accuracy)
cm = confusion_matrix(y_test, predictions)
print("\nConfusion Matrix:")
print(cm)