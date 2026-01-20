import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("C:\\Users\\giris\\Documents\\Predective analysis\\unit-4\\diabetes.csv")
print(data.info())

X = data.drop("Outcome",axis=1)

#Normalization
scaler=MinMaxScaler()
X_scaled =  scaler.fit_transform(X)

wcss = []

for k in range(1,8):
    kmeans = KMeans(n_clusters=k, random_state=20)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    
    
plt.figure(figsize=(8,5))
plt.plot(range(1,8),wcss,marker='o')
plt.title("Elbow Method to determine optimal K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS (Within Cluster Sum of Squares")
plt.grid(True)
plt.show()

print("WCSS values for each k = 1 to 8:")
for i,val in enumerate(wcss,1):
    print(f"k = {i} -> WCSS = {val}")


