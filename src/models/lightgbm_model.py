import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import randint

# load training data
train_data = pd.read_csv('data/train/train_data.csv')

X = train_data.drop(columns=['label', 'long', 'lat'])
y = train_data['label']

# split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# define initial params
initial_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'verbose': -1,
    'early_stopping_rounds': 10
}

# train initial model with early stopping
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

model = lgb.train(initial_params, lgb_train, valid_sets=[lgb_val])

# predict on val set and evaluate
y_pred = model.predict(X_val, num_iteration=model.best_iteration)
y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

print("Initial Model Accuracy:", accuracy_score(y_val, y_pred_binary))
print(classification_report(y_val, y_pred_binary))

# determine params 
param_dist = {
    'num_leaves': randint(10, 60),            
    'learning_rate': [0.01, 0.05, 0.1, 0.2],         
    'n_estimators': [50, 100, 200, 300, 500],    
    'max_depth': [3, 5, 7, 10],              
    'subsample': [0.7, 0.8, 0.9],                
    'colsample_bytree': [0.6, 0.7, 0.8],
    'feature_fraction': [0.7, 0.8, 0.9],          
    'bagging_fraction': [0.7, 0.8, 0.9],    
    'bagging_freq': [1, 5, 10],  
    'lambda_l1': [0.0, 0.1, 0.5],                 
    'lambda_l2': [0.0, 0.1, 0.5],    
    'min_data_in_leaf': [10, 20, 30],            
}

lgb_estimator = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', boosting_type='gbdt', random_state=42)

# perform randomized search for faster tuning
random_search = RandomizedSearchCV(estimator=lgb_estimator, param_distributions=param_dist, n_iter=50, cv=3, scoring='accuracy', verbose=1, random_state=42)
random_search.fit(X_train, y_train)

print("Best parameters found by RandomizedSearchCV:", random_search.best_params_)
print("Best cross-validated accuracy found by RandomizedSearchCV:", random_search.best_score_)

# train final model /w best params
best_params = random_search.best_params_
final_model = lgb.LGBMClassifier(**best_params, random_state=42)
final_model.fit(X_train, y_train)

# predict on test data
test_data = pd.read_csv('data/test/test_data.csv')
X_test = test_data.drop(columns=['label', 'long', 'lat'])
y_test = test_data['label']
y_test_pred = final_model.predict(X_test)

# evaluate final model on test set
print("Final Model Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
