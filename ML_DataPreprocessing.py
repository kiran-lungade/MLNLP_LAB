import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

df = pd.read_csv(r"Iris.csv")

#rename the column if applicable
#df.columns = ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']

print(df.head())
print(df.tail())

print(df.info())
print(df.describe())

#check missing values
print("Missing values:\n", df.isnull().sum())
print()

#Data Analysis
#count each species
print("Species count:\n", df['Species'].value_counts())
print()

#Data Visualization
#pair plot
sns.pairplot(df, hue='Species', markers=["o", "s", "D"])
plt.show()

# Box plots 
plt.figure(figsize=(12, 8))
for i, feature in enumerate(df.columns[1:5]):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='Species', y=feature, data=df)
    plt.title(f'{feature} by Species')
plt.tight_layout()
plt.show()

# Distribution of each feature
plt.figure(figsize=(12, 8))
for i, feature in enumerate(df.columns[1:5]):
    plt.subplot(2, 2, i+1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

# Skewness and kurtosis
for feature in df.columns[1:5]:
    print(f'{feature} - Skewness: {skew(df[feature])}, Kurtosis: {kurtosis(df[feature])}')