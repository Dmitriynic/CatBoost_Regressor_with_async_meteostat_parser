import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import numpy as np
from sklearn.neighbors import BallTree

df_train = pd.read_csv('data/train.csv')
df_feature = pd.read_csv('tmp_features/edit_features.csv')
tree = BallTree(df_feature[['lat', 'lon']], leaf_size=15)

distances, indices = tree.query(df_train[['lat', 'lon']], k=1)

result_df = pd.DataFrame()

for train_index, feature_index in enumerate(indices.flatten()):
    row_train = df_train.loc[train_index]
    if not pd.isna(feature_index):
        row_feature = df_feature.drop(['lat', 'lon'], axis=1).loc[feature_index]
        result_row = pd.concat([row_train, row_feature], axis=0)
    else:
        result_row = row_train.append(pd.Series([pd.NA]*(len(df_feature.columns) - 2)), index=df_feature.drop(['lat', 'lon'], axis=1).columns)
    result_df = pd.concat([result_df, result_row.to_frame().T], ignore_index=True)

df_train = result_df

X = df_train.drop(columns=['score'])
y = df_train['score']
#features selected by result of model.select_features
X = X[['0', '12', '18', '19', '34', '42', '78', '91', '102', '146', '178', '183', '208', '221', '281', '303', '326', '330', '336', 'prcp']]

k_folds = 5
random_state = 800

model = CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, loss_function='MAE', verbose=False, random_seed = random_state)

mae_scores = []

kf = KFold(n_splits=k_folds, shuffle=True, random_state = random_state)

for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    
    y_pred = model.predict(X_val)
    
    mae = mean_absolute_error(y_val, y_pred)
    mae_scores.append(mae)
    # summary = model.select_features(X_train, y_train, 
    #                                 eval_set=(X_val, y_val),
    #                                 features_for_select=['0-362','cluster_1','cluster_2','cluster_3', 'cluster_4','tavg','tmin','tmax','prcp'],
    #                                 num_features_to_select = 20,
    #                                 steps=1,
    #                                 train_final_model=False,
    #                                 logging_level='Silent')

    # print(summary['selected_features_names'])
    # print(f"Best loss: {summary['loss_graph']['loss_values'][-1]}")

mean_mae = np.mean(mae_scores)
print(mae_scores)
print("Mean MAE:", mean_mae)

model.save_model('catboost_model.bin')