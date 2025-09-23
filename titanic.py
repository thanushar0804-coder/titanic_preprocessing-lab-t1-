# Data Cleaning & Preprocessing with Titanic Dataset

## 1. Import the dataset and explore basic info

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Titanic dataset (from seaborn for convenience)
titanic = sns.load_dataset('titanic')

# Preview the data
print(titanic.head())

# Check for nulls and data types
print(titanic.info())
print(titanic.isnull().sum())


## 2. Handle missing values


# Fill 'age' nulls with median
titanic['age'].fillna(titanic['age'].median(), inplace=True)

# Fill 'embarked' nulls with the most frequent value (mode)
titanic['embarked'].fillna(titanic['embarked'].mode()[0], inplace=True)

# Drop 'deck' column (too many missing values)
titanic.drop(columns=['deck'], inplace=True)

# Drop rows where 'embark_town' is missing (optional, very few missing)
titanic.dropna(subset=['embark_town'], inplace=True)


## 3. Convert categorical features to numerical (encoding)

# Encode 'sex' (male=0, female=1)
titanic['sex'] = titanic['sex'].map({'male': 0, 'female': 1})

# One-hot encode 'embarked'
titanic = pd.get_dummies(titanic, columns=['embarked'], drop_first=True)


## 4. Normalize/Standardize numerical features


from sklearn.preprocessing import StandardScaler

# List of numerical features
num_features = ['age', 'fare', 'sibsp', 'parch']

# Create scaler object
scaler = StandardScaler()

# Fit and transform
titanic[num_features] = scaler.fit_transform(titanic[num_features])


## 5. Visualize outliers with boxplots and remove them


# Visualize outliers in 'fare'
plt.figure(figsize=(8, 4))
sns.boxplot(x=titanic['fare'])
plt.title('Boxplot of Fare')
plt.show()

# Remove outliers in 'fare' (e.g., keep within 1.5*IQR)
Q1 = titanic['fare'].quantile(0.25)
Q3 = titanic['fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

titanic = titanic[(titanic['fare'] >= lower_bound) & (titanic['fare'] <= upper_bound)]

## Final Check

print(titanic.info())
print(titanic.head())



