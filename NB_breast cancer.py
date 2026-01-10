# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 09:15:56 2025

@author: MANEET KAUR
"""

# Required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Replace with your actual file path
bc = pd.read_csv("D:/TEACHING/2024 teaching academic research/R/programs/datasets/BREAST CANCER.csv")

# View dataset structure
print(bc.info())
print(bc.head())

#Split into train and test
train = bc.iloc[:450, :]
test = bc.iloc[451:, :]

# Normalize numeric features
scaler = MinMaxScaler()

# Copy to preserve original data
train_scaled = train.copy()
test_scaled = test.copy()

# Identify numeric columns (excluding 'diagnosis')
num_cols = train.select_dtypes(include=[np.number]).columns
train_scaled[num_cols] = scaler.fit_transform(train[num_cols])
test_scaled[num_cols] = scaler.transform(test[num_cols])

# Train Naive Bayes model
X_train = train_scaled.drop('diagnosis', axis=1)
y_train = train_scaled['diagnosis']
X_test = test_scaled.drop('diagnosis', axis=1)
y_test = test_scaled['diagnosis']
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Confusion matrix and accuracy
tab = confusion_matrix(y_test, pred)
print("\nConfusion Matrix:\n", tab)
accuracy = accuracy_score(y_test, pred) * 100
print(f"\nAccuracy: {accuracy:.2f}%")
