import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df_features = pd.read_csv('data/features.csv')
df_open_data = pd.read_csv('open_data.csv')
sns.pairplot(df_features[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']])
plt.show()
df_open_data = df_open_data[['lat', 'lon', 'tavg', 'tmin', 'tmax', 'prcp']].dropna()


X = df_features[['2', '3']]
X1 = df_features[['3', '6']]
X2 = df_features[['3', '7']]
X3 = df_features[['0', '2']]
X4 = df_features[['0', '7']]
def clustering(num_feature, num_clusters, X):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    df_features[f'cluster_{num_feature}'] = cluster_labels

clustering(0, 2, X)
clustering(1, 3, X1)
clustering(2, 3, X2)
clustering(3, 2, X3)
clustering(4, 3, X4)

print(df_features['cluster_2'].value_counts())

df_features = df_features.merge(df_open_data, on=['lat', 'lon'], how='inner')

df_features.to_csv('tmp_features/edit_features.csv')