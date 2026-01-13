from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
import pandas as pd

# Load dataset
cancer = load_breast_cancer()

# Convert to DataFrame
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

# Show summary of data
#569 samples (rows), 30 features (columns)
#Target: 0 = malignant, 1 = benign
print(df.describe())
print(df.head())
print(df.info())

# Use only first two features for visualization
X = cancer.data[:, :2]
y = cancer.target

# Train SVM model
svm = SVC(kernel="linear")
svm.fit(X, y)
# Plot decision boundary
DecisionBoundaryDisplay.from_estimator(
    svm,
    X,
    response_method="predict",
    alpha=0.8,
    cmap="Pastel1",
    xlabel=cancer.feature_names[0],
    ylabel=cancer.feature_names[1],
)
# Scatter plot of data points
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors="k")
plt.show()


