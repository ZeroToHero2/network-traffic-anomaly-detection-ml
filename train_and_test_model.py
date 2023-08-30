
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('UNSW_NB15_training-set.csv')

# Remove irrelevant features
df.drop(['id', 'proto', 'service', 'state', 'attack_cat'], axis=1, inplace=True)

# Encode categorical variables
cat_cols = ['spkts', 'dpkts']
for col in cat_cols:
    df[col] = pd.factorize(df[col])[0]

# Split into features and labels
X = df.drop(['label'], axis=1)
y = df['label']

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train a random forest classifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Load the test dataset
df_test = pd.read_csv('UNSW_NB15_testing-set.csv')

# Remove irrelevant features
df_test.drop(['id', 'proto', 'service', 'state',
             'attack_cat'], axis=1, inplace=True)

# Encode categorical variables
for col in cat_cols:
    df_test[col] = pd.factorize(df_test[col])[0]

# Split into features and labels
X_test = df_test.drop(['label'], axis=1)
y_test = df_test['label']

# Scale the features
X_test = scaler.transform(X_test)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))
