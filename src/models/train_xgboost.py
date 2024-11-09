import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb

# 1. Đọc dữ liệu
train_data = pd.read_csv('data/train/train_data.csv')
test_data = pd.read_csv('data/test/test_data.csv')

# define feature & target
feature_columns = ['long', 'lat', 'B4', 'B3', 'B2', 'B8', 'B11', 'B12', 'NDVI', 'EVI', 'SAVI', 'NDWI', 'NBR', 'CIgreen']
target_column = 'label'

# set up
X_train = train_data[feature_columns]
y_train = train_data[target_column]

X_test = test_data[feature_columns]
y_test = test_data[target_column]

# determine params 
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# init GridSearchCV vs cv=5
clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# start time
start_time = time.time()

# train
grid_search.fit(X_train, y_train)

# end time
end_time = time.time()

# best param
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# result
y_test_pred = grid_search.best_estimator_.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy:.2f}')

test_report = classification_report(y_test, y_test_pred)
print(f'Test Classification Report:\n{test_report}')

# calc ROC-AUC score
roc_auc = roc_auc_score(y_test, y_test_pred)
print(f'ROC-AUC score: {roc_auc:.2f}')

# print time train model
print(f"Time taken: {end_time - start_time:.2f} seconds")
