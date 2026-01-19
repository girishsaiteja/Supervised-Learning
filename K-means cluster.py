import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
        'X' : [1,2,3,8,9,10,1.5,2.5,8.5,9.2],
        'Y' : [1,2,1,8,9,8,1.2,1.8,8.2,9.1]
    
        }

df = pd.DataFrame(data)
X=df[['X','Y']]

wcss = [] #WCSS (Within Cluster Sum of Squares)
for k in range(1,8):
    kmeans = KMeans(n_clusters=k,random_state=0)    
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)                #inertia_ gives WCSS  (inertia = wcss)
    

#Plot
plt.plot(range(1,8),wcss,marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS (Within Cluster Sum of Squares")
plt.grid(True)
plt.show()
