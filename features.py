import pandas as pd

df_features = pd.read_csv('data/features.csv')
df_open_data = pd.read_csv('open_data.csv')

df_open_data = df_open_data[['lat', 'lon', 'tavg', 'tmin', 'tmax', 'prcp']].dropna()

df_features = df_features.merge(df_open_data, on=['lat', 'lon'], how='inner')

df_features.to_csv('tmp_features/edit_features.csv')