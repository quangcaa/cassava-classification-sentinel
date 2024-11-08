import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV
import time 

# Load training and test data
train_data = pd.read_csv('data/train/train_data.csv')
test_data = pd.read_csv('data/test/test_data.csv')

# Define feature and target columns
feature_columns = ['long', 'lat', 'B4', 'B3', 'B2', 'B8', 'B11', 'B12', 'NDVI', 'EVI', 'SAVI', 'NDWI', 'NBR', 'CIgreen']
target_column = 'label'

# Set up features and target for training and test sets
X_train = train_data[feature_columns]
y_train = train_data[target_column]

X_test = test_data[feature_columns]
y_test = test_data[target_column]

# Initialize K-Nearest Neighbors classifier
clf = KNeighborsClassifier()

# Define parameter grid for KNeighborsClassifier
param_grid = {
    'n_neighbors': [3, 5, 7, 10, 15, 20],    # Số lượng hàng xóm gần nhất
    'weights': ['uniform', 'distance'],      # Cách tính trọng số
    'metric': ['euclidean', 'manhattan', 'minkowski'],  # Các loại khoảng cách
    'p': [1, 2],                             # Chỉ số của metric Minkowski (p=1 là Manhattan, p=2 là Euclidean)
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Thuật toán tìm kiếm (auto tự động chọn)
    'leaf_size': [10, 20, 30, 40, 50]        # Kích thước lá cho các thuật toán như BallTree hoặc KDTree
}

# Set up GridSearchCV with cross-validation (cv=5)
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

# Start time
start_time = time.time()

# Train model
grid_search.fit(X_train, y_train)

# End time
end_time = time.time()

# Output best parameters and cross-validation score
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# Evaluate on test data
y_test_pred = grid_search.best_estimator_.predict(X_test)

# Calculate and print accuracy, classification report, and ROC-AUC score
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy}')

test_report = classification_report(y_test, y_test_pred)
print(f'Test Classification Report: {test_report}')

# Calculate and print ROC-AUC score
roc_auc = roc_auc_score(y_test, y_test_pred)
print(f'ROC-AUC score: {roc_auc}')

# Print training time
print(f"Time taken: {end_time - start_time} seconds")
