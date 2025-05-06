import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import numpy as np

df = pd.read_csv('EDA/train_newfeature.csv')

target = 'Rings'
id_col = 'id'

# Define feature columns (exclude ID and target)
features = [col for col in df.columns if col not in [id_col, target]]

# Specify categorical features
cat_features = ['Sex']


X_train, X_valid, y_train, y_valid = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42
)

train_pool = Pool(X_train, y_train, cat_features=cat_features)
valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)

model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    early_stopping_rounds=50,
    verbose=100
)

model.fit(train_pool, eval_set=valid_pool)


y_pred = model.predict(X_valid)
rmsle = np.sqrt(mean_squared_log_error(y_valid, y_pred))
print(f'Validation RMSLE: {rmsle:.4f}')