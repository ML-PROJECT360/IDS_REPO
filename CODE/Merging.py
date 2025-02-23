import pandas as pd

# Load datasets
data = pd.read_csv('/content/drive/MyDrive/IDS/Data.csv')
labels = pd.read_csv('/content/drive/MyDrive/IDS/Label.csv')

print("Data Shape:", data.shape)
print("Labels Shape:", labels.shape)

print("\nData Head:\n", data.head())
print("\nLabels Head:\n", labels.head())

print("\nMissing Values in Data:\n", data.isnull().sum())
print("\nMissing Values in Labels:\n", labels.isnull().sum())

# Check label distribution (class balance)
print("\nLabel Distribution:\n", labels.value_counts())

print("\nUnique Classes in Labels:\n", labels.nunique())

# Merge datasets
combined_df = pd.concat([data, labels], axis=1)

# Save the merged dataset
combined_df.to_csv('Merged_Dataset.csv', index=False)

print("✅ Datasets merged successfully! Shape:", combined_df.shape)

import matplotlib.pyplot as plt
import seaborn as sns

# Check label distribution (last column is the label)
label_column = 'Label'
label_counts = combined_df[label_column].value_counts()

print("Class Distribution Before Balancing:\n", label_counts)

# Visualize class imbalance
plt.figure(figsize=(10, 6))
sns.barplot(x=label_counts.index, y=label_counts.values, palette='viridis')
plt.title('Class Distribution Before Balancing')
plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Visualize class imbalance (fixed warning)
plt.figure(figsize=(10, 6))
sns.barplot(x=label_counts.index, y=label_counts.values, hue=label_counts.index, legend=False, palette='viridis')
plt.title('Class Distribution Before Balancing')
plt.xlabel('Classes')
plt.ylabel('Frequency')
plt.show()


!pip install smote-variants # Install the smote_variants package

import numpy as np
import smote_variants as sv
from sklearn.model_selection import train_test_split

# Split features and labels
X = combined_df.drop(columns=[label_column])
y = combined_df[label_column]

# Check feature and label shape
print("Features Shape:", X.shape)
print("Labels Shape:", y.shape)

# SMOTE-PSOBAT Oversampling
smote_psobat = sv.SMOTE_PSOBAT()
X_res_psobat, y_res_psobat = smote_psobat.sample(X.to_numpy(), y.to_numpy())

print("After SMOTE-PSOBAT: Features:", X_res_psobat.shape, " Labels:", y_res_psobat.shape)

# Save balanced dataset
balanced_psobat_df = pd.DataFrame(X_res_psobat, columns=X.columns)
balanced_psobat_df[label_column] = y_res_psobat
balanced_psobat_df.to_csv('Balanced_Dataset_PSOBAT.csv', index=False)
print("✅ Balanced dataset (SMOTE-PSOBAT) saved!")

# SMOTE-KMeans Oversampling
smote_kmeans = sv.SMOTE_KMeans()
X_res_kmeans, y_res_kmeans = smote_kmeans.sample(X.to_numpy(), y.to_numpy())

print("After SMOTE-KMeans: Features:", X_res_kmeans.shape, " Labels:", y_res_kmeans.shape)

# Save balanced dataset
balanced_kmeans_df = pd.DataFrame(X_res_kmeans, columns=X.columns)
balanced_kmeans_df[label_column] = y_res_kmeans
balanced_kmeans_df.to_csv('Balanced_Dataset_KMeans.csv', index=False)
print("✅ Balanced dataset (SMOTE-KMeans) saved!")


# Check class distribution after SMOTE
print("Class Distribution (SMOTE-PSOBAT):\n", balanced_psobat_df[label_column].value_counts())
print("Class Distribution (SMOTE-KMeans):\n", balanced_kmeans_df[label_column].value_counts())

