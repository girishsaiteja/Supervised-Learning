import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


#Sample Dataset
data = {
        'Milk' :    [0,1,1,1,0],
        'Bread' :   [1,0,1,1,1],
        'Butter' :  [1,0,1,1,0],
        'Yogurt' :  [0,1,0,1,1],
        'Cheese' :  [1,0,0,0,1],
        'Juice' :   [0,1,1,0,1]
        }


df = pd.DataFrame(data)
df

#Apriori
frequent_itemssets = apriori(df,min_support = 0.4, use_colnames = True)
print("Frequent Itemsets:\n", frequent_itemssets)

#Association rules
rules = association_rules(frequent_itemssets, metric = 'confidence', min_threshold = 0.6)
print("\nAssociation Rules:\n", rules)