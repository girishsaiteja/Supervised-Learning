#hierarchical clustering for dataframe

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

#Sample Data
data = {
       'X': [1,2,3,8,9,10,1.5,2.5,8.5,9.2],
       'Y': [1,2,1,3,8,9,1.2,1.8,8.2,9.1]
        }

df = pd.DataFrame(data)
X = df[['X','Y']]
plt.figure(figsize=(10,5))
plt.title("Dendogram for Hierarchical Clustering")
dendrogram(linkage(X, method='ward'))
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

hc = AgglomerativeClustering(
    n_clusters=2,
    linkage='ward',
    metric='euclidean')

labels = hc.fit_predict(X)
print("Cluster Labels:", labels)

#plot
plt.scatter(df['X'], df['Y'],c=labels,cmap='rainbow')
plt.title("Hierarchical Clustering Result")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
