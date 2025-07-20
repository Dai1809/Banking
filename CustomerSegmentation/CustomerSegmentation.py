import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('Mall_Customers.csv')

# Drop CustomerID and one-hot encode Gender
df = df.drop("CustomerID", axis=1)
df = pd.get_dummies(df, drop_first=True)  # Gender_Male will be added

# Save original column names (except Gender_Male) for plotting after unscaling
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Scale all features at once
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[features])

# Add the encoded gender column back
scaled_df = pd.DataFrame(scaled_data, columns=features)
scaled_df['Gender_Male'] = df['Gender_Male'].values

# Elbow method
WCSS = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(scaled_df)
    WCSS.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), WCSS, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid()
plt.show()

# Apply KMeans with chosen cluster count
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(scaled_df)
scaled_df['Cluster'] = clusters

# Unscale the data for plotting
unscaled_features = scaler.inverse_transform(scaled_df[features])
unscaled_df = pd.DataFrame(unscaled_features, columns=features)
unscaled_df['Gender_Male'] = df['Gender_Male'].values
unscaled_df['Cluster'] = clusters

# Plot Age vs Spending Score
plt.figure(figsize=(8, 6))
sns.scatterplot(data=unscaled_df, x='Age', y='Spending Score (1-100)', hue='Cluster', palette='viridis')
plt.title("Customer Segments (Age vs Spending Score)")
plt.grid(True)
plt.show()


fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Plot unscaled values
ax.scatter(
    unscaled_df['Age'],
    unscaled_df['Annual Income (k$)'],
    unscaled_df['Spending Score (1-100)'],
    c=clusters,
    cmap='rainbow'
)

ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.title('3D Clustering of Customers (Unscaled)')
plt.show()


