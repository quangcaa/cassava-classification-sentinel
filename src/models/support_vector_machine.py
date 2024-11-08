import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time 
import numpy as np

# load training data
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

# init: Support Vector Classifier (SVC)
clf = SVC(random_state=42)

# determine params 
param_grid = {
    'C': [10, 100, 1000, 10000],  # Log scale for more efficient search
    'kernel': ['rbf'],
    'gamma': ['scale', 'auto'],
    'shrinking': [True],
    'class_weight': ['balanced']
}

# init GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# start time
start_time = time.time()

# train
grid_search.fit(X_train, y_train)

# end time
end_time = time.time()

# best params found
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# result
y_test_pred = grid_search.best_estimator_.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy}')

test_report = classification_report(y_test, y_test_pred)
print(f'Test Classification Report: {test_report}')

roc_auc = roc_auc_score(y_test, y_test_pred)
print(f'ROC-AUC score: {roc_auc}')

# print time taken to train model
print(f"Time taken: {end_time - start_time} seconds")
