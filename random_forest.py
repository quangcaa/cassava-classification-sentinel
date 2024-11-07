import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
import time 

# load training data
train_data = pd.read_csv('train_data/train_data.csv')
test_data = pd.read_csv('train_data/test_data.csv')

# define feature & target
feature_columns = ['long', 'lat', 'rgb_band_1', 'rgb_band_2', 'rgb_band_3', 'ndvi_band_1']
target_column = 'label'

# training data
X_train = train_data[feature_columns]
y_train = train_data[target_column]

# test data
X_test = test_data[feature_columns]
y_test = test_data[target_column]

# init: Random Forest classifier
clf = RandomForestClassifier(random_state=42)

# Định nghĩa bộ tham số cần thử nghiệm
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'class_weight': ['balanced', None]
}

# khởi tạo GridSearchCV vs cv=5
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
print(f'Test Accuracy: {test_accuracy}')

test_report = classification_report(y_test, y_test_pred)
print(f'Test Classification Report: {test_report}')

# calc ROC-AUC score
roc_auc = roc_auc_score(y_test, y_test_pred)
print(f'ROC-AUC score: {roc_auc}')

# print time train model
print(f"Time taken: {end_time - start_time} seconds")