# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
wbcd = pd.read_csv("D:/TEACHING/2024 teaching academic research/R/programs/datasets/BREAST CANCER.csv")
# View structure
print(wbcd.info())
# Remove the ID column
wbcd = wbcd.drop(wbcd.columns[0], axis=1)
# Frequency table for diagnosis
print(wbcd['diagnosis'].value_counts())
# Relabel diagnosis column
wbcd['diagnosis'] = wbcd['diagnosis'].map({'B': 'Benign', 'M': 'Malignant'})
# Summary statistics
print(wbcd[['radius_mean', 'area_mean', 'smoothness_mean']].describe())
wbcd.info()
wbcd.describe()
# Normalize the features
#nor <-function(x) { (x -min(x))/(max(x)-min(x))}  
scaler = MinMaxScaler()
features = wbcd.columns[2:]
wbcd_n = pd.DataFrame(scaler.fit_transform(wbcd[features]), columns=features)
wbcd_n.describe()

# Split data into training and testing sets
X_train = wbcd_n.iloc[:469]
X_test = wbcd_n.iloc[469:]
y_train = wbcd['diagnosis'].iloc[:469]
y_test = wbcd['diagnosis'].iloc[469:]

# Apply KNN
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Calculate accuracy
#accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy * 100:.2f}%")

