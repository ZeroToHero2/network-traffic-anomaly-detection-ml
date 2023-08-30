import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('UNSW_NB15_training-set.csv')

def preprocess_data(df):
    df['attack_cat'] = np.where(df['attack_cat'] == 'Normal', 0, 1)
    
    # One-hot encode the categorical features
    df = pd.get_dummies(df, columns=['proto', 'service', 'state'])

    # Scale the numerical features
    scaler = MinMaxScaler()
    numerical_cols = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss',
                      'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack',
                      'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
                      'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df


def detect_anomalies(df, k):
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(df)

    distances, indices = nbrs.kneighbors(df)
    distances = np.delete(distances, 0, axis=1)
    avg_distance = np.mean(distances, axis=1)

    is_anomaly = avg_distance > np.mean(avg_distance) + np.std(avg_distance)

    return is_anomaly


# Load the dataset
#df = pd.read_csv('UNSW_NB15_testing-set.csv')
df = pd.read_csv('UNSW_NB15_training-set.csv')

# Preprocess the dataset
df = preprocess_data(df)

# Detect anomalies using the KNN algorithm
is_anomaly = detect_anomalies(df, k=5)

# Plot the results
# Create a figure and axis objects
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the actual data
sns.lineplot(x=df.index, y=df['sbytes'], ax=ax)

# Plot the anomalies
anomaly_indices = [i for i, x in enumerate(is_anomaly) if x]
anomaly_values = df.iloc[anomaly_indices]['sbytes']
sns.scatterplot(x=anomaly_indices, y=anomaly_values,
                color='red', marker='X', s=50, ax=ax)

# Add titles and labels
ax.set_title('Network Anomaly Detection Results')
ax.set_xlabel('Network Operation Count (in seconds)')
ax.set_ylabel('Bytes')
# plt.plot(is_anomaly) # plot the anomaly detection results
plt.show()