import pandas as pd
from catboost import CatBoostRegressor
from sklearn.neighbors import BallTree

df_test = pd.read_csv('data/test.csv')
df_feature = pd.read_csv('tmp_features/edit_features.csv')

tree = BallTree(df_feature[['lat', 'lon']], leaf_size=15)

distances, indices = tree.query(df_test[['lat', 'lon']], k=1)

result_df = pd.DataFrame()

for test_index, feature_index in enumerate(indices.flatten()):
    row_test = df_test.loc[test_index]
    if not pd.isna(feature_index):
        row_feature = df_feature.drop(['lat', 'lon'], axis=1).loc[feature_index]
        result_row = pd.concat([row_test, row_feature], axis=0)
    else:
        result_row = row_test.append(pd.Series([pd.NA]*(len(df_feature.columns) - 2)), index=df_feature.drop(['lat', 'lon'], axis=1).columns)
    result_df = pd.concat([result_df, result_row.to_frame().T], ignore_index=True)

df_test = result_df

X_test = df_test

model = CatBoostRegressor()
model.load_model('catboost_model.bin')

y_pred_test = model.predict(X_test)

submission_df = pd.DataFrame({'id': df_test['id'].astype(int), 'score': y_pred_test})
submission_df.to_csv('submission.csv', index=False)