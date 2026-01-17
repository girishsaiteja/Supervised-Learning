#Groceries Dataset

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
data=[]

for line in open("C:\\Users\\giris\\Documents\\Predective analysis\\unit-4\\groceries.csv"):
    data.append(line.strip().split(","))
    
print(data)
te = TransactionEncoder()
df = pd.DataFrame(te.fit_transform(data), columns = te.columns_)
frequent = apriori(df, min_support=0.02, use_colnames=True)

#Generate rules
rules = association_rules(frequent, metric="confidence",min_threshold=0.5)

print("\nFrequent Itemsets:\n",frequent)
print("\nAssociation Rules:\n,rules")
