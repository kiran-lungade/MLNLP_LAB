import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_csv(r"basket_analysis.csv")

print(df.head())

# Drop the 'ID' column if it exists (as it's not needed for association rule mining)
df = df.drop('ID', axis=1, errors='ignore')

# Apply Apriori algorithm to find frequent itemsets
min_support = 0.02  # Adjust the minimum support as needed
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

# Generate association rules
min_confidence = 0.5  # Adjust the minimum confidence as needed
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)

# Display the frequent itemsets and association rules
print("\nFrequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence']])
