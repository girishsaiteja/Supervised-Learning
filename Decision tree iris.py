import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
iris_data = load_iris()
iris = pd.DataFrame(
    np.c_[iris_data['data'], iris_data['target']],
    columns = iris_data['feature_names'] + ['species']
)
iris

# Split dataset
np.random.seed(678)
s = np.random.choice(iris.index, 100, replace=False)
iris_train = iris.loc[s]
iris_test = iris.drop(s)
X_train = iris_train.drop(columns=['species'])
y_train = iris_train['species']
X_test = iris_test.drop(columns=['species'])
y_test = iris_test['species']

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Plot tree
plt.figure(figsize=(16, 10))
plot_tree(
    model,
    filled=True,
    feature_names=iris_data['feature_names'],
    class_names=iris_data['target_names'],
    rounded=True,
    fontsize=10
)
plt.show()

# Predict and evaluate
y_pred = model.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
clas_report = classification_report(y_test, y_pred)
print(clas_report)
accuracy = accuracy_score(y_test, y_pred) * 100
print("Confusion Matrix:\n", conf_mat)
print(f"Model Accuracy: {accuracy:.2f}%")
# Error rate
error_rate = 100 - accuracy
print(f"Model Error Rate: {error_rate:.2f}%")

