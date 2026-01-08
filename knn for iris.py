# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 09:22:53 2025

@author: MANEET KAUR
"""

# Load required libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load iris dataset from seaborn
df = sns.load_dataset('iris')

# See the structure
print(df.head())

# Generate random indices for 90% of rows
np.random.seed(42)
ran = np.random.choice(df.index, size=int(0.9 * len(df)), replace=False)
ran
print(df.info())

# Normalization using MinMaxScaler (same as R normalization function)
scaler = MinMaxScaler()
iris_norm = pd.DataFrame(scaler.fit_transform(df.iloc[:, 0:4]), 
                         columns=df.columns[0:4])

# Summaries
print("\nOriginal Data Summary:")
print(df.describe())
print("\nNormalized Data Summary:")
print(iris_norm.describe())

# Extract training and testing sets
iris_train = iris_norm.iloc[ran, :]
iris_test = iris_norm.drop(ran)
iris_target_category = df.iloc[ran, 4]
iris_test_category = df.drop(ran)['species']

# Run KNN
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(iris_train, iris_target_category)
pr = knn.predict(iris_test)

# Confusion matrix
tab = confusion_matrix(iris_test_category, pr)
print("\nConfusion Matrix:\n", tab)

# Accuracy
acc = accuracy_score(iris_test_category, pr) * 100
print("\nAccuracy: {:.2f}%".format(acc))
