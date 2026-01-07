# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 09:47:15 2025

@author: MANEET KAUR
"""

# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load data 
dia = sns.load_dataset('diamonds')

# See the structure
print(dia.head())
print(dia.info())
print(dia.describe())

# Create a random number equal to 90% of total number of rows
np.random.seed(42)
ran = np.random.choice(dia.index, size=int(0.9 * len(dia)), replace=False)

# Normalization function using MinMaxScaler (similar to R function 'nor')
cols_to_normalize = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
scaler = MinMaxScaler()
dia_nor = pd.DataFrame(scaler.fit_transform(dia[cols_to_normalize]), 
                       columns=cols_to_normalize)

print("\nNormalized data summary:")
print(dia_nor.describe())

# Extract training and testing datasets
dia_train = dia_nor.iloc[ran, :]
dia_test = dia_nor.drop(ran)

# Target column ('cut'), converted to categorical
dia_target = dia.loc[ran, 'cut']
test_target = dia.drop(ran)['cut']

# Run KNN function (k = 22)
knn = KNeighborsClassifier(n_neighbors=22)
knn.fit(dia_train, dia_target)
pr = knn.predict(dia_test)

# Create confusion matrix
tb = confusion_matrix(test_target, pr, labels=dia['cut'].unique())
print("\nConfusion Matrix:\n", tb)

# Calculate accuracy (same as R’s accuracy function)
acc = accuracy_score(test_target, pr) * 100
print("\nModel Accuracy: {:.2f}%".format(acc))
