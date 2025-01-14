#%% working directory

import os
import joblib

path = "D:\Studia\semestr7\inźynierka\Market-analysis"
path = "C:\Studia\Market-analysis"
os.chdir(path)

#%% categorize y

def categorize_y(x, y, pred_name="close"):
    last_column_name = [col for col in x.columns if col.startswith(pred_name)][-1]
    last_data = x[last_column_name]
    res = y - last_data
    res[res >= 0] = 1
    res[res < 0] = 0
    return res

#%% reading data (categorized y)

import pandas as pd
import numpy as np

def load_dataset(directory_name, name, target_horizon):
    df_file_path = os.path.join(directory_name, f"df_{name}_{target_horizon}.csv")
    y_file_path = os.path.join(directory_name, f"y_{name}_{target_horizon}.csv")
    if os.path.isfile(df_file_path) and os.path.isfile(y_file_path):
        df = pd.read_csv(df_file_path)
        y = pd.read_csv(y_file_path)
        return df, y

directory_name = "datasets"

df_train, y_train = load_dataset(directory_name, "train", 1)
df_val, y_val = load_dataset(directory_name, "val", 1)
df_test, y_test = load_dataset(directory_name, "test", 1)

y_train = categorize_y(df_train, y_train.y)
y_val = categorize_y(df_val, y_val.y)
y_test = categorize_y(df_test, y_test.y)

#%% evaluation

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

def evaluate_model_on_cat(model, x, y):
    y_pred = model.predict(x)
    y_pred = categorize_pred(y_pred)
    results = {}
    name_columns = [col for col in x.columns if col.startswith("name")]
    for col in name_columns:
        name = col.lstrip("name_")
        name_index = x[x[f"name_{name}"] == 1].index
        
        y_inc = y[name_index]
        pred_inc = y_pred.reshape(-1,)[name_index]
        
        mse = accuracy_score(y_inc, pred_inc)
        r2 = r2_score(y_inc, pred_inc)
        results[name] = {"acc": mse, "r2": r2}
    return results

def count_acc(model, x, y):
    y_pred = model.predict(x)
    acc = accuracy_score(y, y_pred)
    return acc

def categorize_pred(pred):
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    return pred

#%% plotting results

import matplotlib.pyplot as plt

def plot_cat(model, x, y):
    y_pred = model.predict(x)
    y_pred = categorize_pred(y_pred)
    name_columns = [col for col in df_test.columns if col.startswith("name")]
    for col in name_columns:
        name = col.lstrip("name_")
            
        name_index = x[x[f"name_{name}"] == 1].index
        y_cat = y[name_index]
        pred_cat = y_pred[name_index]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(name_index, y_cat, alpha=0.7, label='original', color='blue')
        plt.scatter(name_index, pred_cat, label='pred', alpha=0.7, color='orange')
        plt.title(f'{name} close price prediction', fontsize=16)
        plt.xlabel('index', fontsize=12)
        plt.ylabel('close price', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()
        
#%% LogisticRegression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(df_train, y_train)
y_pred = model.predict(df_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy) # Accuracy: 0.47019867549668876
print("Confusion Matrix:\n", conf_matrix)

evaluate_model_on_cat(model, df_test, y_test)
plot_cat(model, df_test, y_test)

#%% tuning LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import Parallel, delayed
import threading

penalties = ['l1', 'l2', 'elasticnet', None]
C_values = [0.1, 1, 10, 100]
solvers = ['liblinear', 'lbfgs', 'saga']

# Function for training and evaluating a model
def train_and_evaluate(penalty, C, solver):
    try:
        model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=1000, n_jobs=-1)
        model.fit(df_train, y_train)
        val_pred = model.predict(df_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        print(f"Penalty: {penalty}, C: {C}, Solver: {solver}, Val Accuracy: {val_accuracy}")
        return (val_accuracy, {'penalty': penalty, 'C': C, 'solver': solver}, model)
    except Exception as e:
        print(f"Skipping combination Penalty: {penalty}, C: {C}, Solver: {solver} due to error: {e}")
        return (0, None, None)

# Parallelize hyperparameter search
results = Parallel(n_jobs=-1)(delayed(train_and_evaluate)(penalty, C, solver) for penalty in penalties for C in C_values for solver in solvers)

# Select the best result
best_result = max(results, key=lambda x: x[0])
best_accuracy, best_params, best_model = best_result

print("Best Parameters:", best_params)
# {Penalty: l1, C: 0.1, Solver: liblinear, Val Accuracy: 0.5581881533101045}

#%% tuned LogisticRegression

model = LogisticRegression(penalty="l1", C=0.1, solver="liblinear", max_iter=1000, n_jobs=-1)
model.fit(df_train, y_train)
y_pred = model.predict(df_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy) # Accuracy: 0.5291162365836949
print("Confusion Matrix:\n", conf_matrix)

evaluate_model_on_cat(model, df_test, y_test)
plot_cat(model, df_test, y_test)

#%% RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
model.fit(df_train, y_train)
y_pred = model.predict(df_test)
y_pred = categorize_pred(y_pred)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy) # Accuracy: 0.4985156428408312
print("Confusion Matrix:\n", conf_matrix)

evaluate_model_on_cat(model, df_test, y_test)
plot_cat(model, df_test, y_test)

#%% fine-tuning random forest

best_score = 0
best_params = None
best_model = None

n_estimators_list = [100, 200, 300]
max_depths = [10, 20, None]

for n_estimators in n_estimators_list:
    for max_depth in max_depths:
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        rf.fit(df_train, y_train)
        val_preds = rf.predict(df_val)
        val_preds = categorize_pred(val_preds)
        score = accuracy_score(y_val, val_preds)
        print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, Acc: {score}")
        print()
        if score > best_score:
            best_score = score
            best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
            best_model = rf

print(f"\nBest Params: {best_params}, Best Validation Acc: {best_score}")
# Best Params: {'n_estimators': 300, 'max_depth': 10}, Best Validation MSE: 0.5101045296167247

#%% tuned RandomForestRegressor

model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
model.fit(df_train, y_train)
y_pred = model.predict(df_test)
y_pred = categorize_pred(y_pred)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy) # Accuracy: 0.5003425439598082
print("Confusion Matrix:\n", conf_matrix)

evaluate_model_on_cat(model, df_test, y_test)
plot_cat(model, df_test, y_test)

#%% xgboost

import xgboost as xgb

model = xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=5, n_jobs=-1)
model.fit(df_train, y_train)
y_pred = model.predict(df_test)
y_pred = categorize_pred(y_pred)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy) # Accuracy: 0.49737382964147064
print("Confusion Matrix:\n", conf_matrix)

evaluate_model_on_cat(model, df_test, y_test)
plot_cat(model, df_test, y_test)

#%% xgboost tuning

best_score = 0
best_params = None
best_model = None

for n_estimators in [100, 200, 300]:
    for max_depth in [3, 5, 7]:
        for learning_rate in [0.01, 0.1, 0.2]:
            for alpha in [0, 0.1, 1]:
                    for gamma in [0, 0.1, 1]:
                        xgb_model = xgb.XGBRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            alpha=alpha,
                            gamma=gamma,
                            n_jobs=-1,
                            random_state=42
                        )
                        xgb_model.fit(df_train, y_train)
                        val_preds = xgb_model.predict(df_val)
                        val_preds = categorize_pred(val_preds)
                        score = mean_squared_error(y_val, val_preds)
                        print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, learning_rate: {learning_rate}, "
                              f"alpha: {alpha}, gamma: {gamma}, MSE: {score}")
                        print()                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                "n_estimators": n_estimators,
                                "max_depth": max_depth,
                                "learning_rate": learning_rate,
                                "alpha": alpha,
                                "gamma": gamma
                            }
                            best_model = xgb_model

print(f"\nBest Params: {best_params}")
print(f"Best Validation MSE: {best_score}")
# Best Params: {'n_estimators': 100, 'max_depth': 7, 'learning_rate': 0.2, 'alpha': 0.1, 'gamma': 0.1}
# Best Validation MSE: 0.5156794425087108

#%% tuned xgboost

model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=7,
    learning_rate=0.2,
    alpha=0.1,
    gamma=0.1,
    n_jobs=-1,
    random_state=42
)
model.fit(df_train, y_train)
y_pred = model.predict(df_test)
y_pred = categorize_pred(y_pred)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy) # Accuracy: 0.4966887417218543
print("Confusion Matrix:\n", conf_matrix)

evaluate_model_on_cat(model, df_test, y_test)
plot_cat(model, df_test, y_test)

#%% bagging

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

base_model = DecisionTreeRegressor(random_state=42)

model = BaggingRegressor(estimator=base_model, 
                                 n_estimators=100, 
                                 random_state=42, 
                                 n_jobs=-1)
model.fit(df_train, y_train)
y_pred = model.predict(df_test)
y_pred = categorize_pred(y_pred)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy) # Accuracy: 0.5158712034711121
print("Confusion Matrix:\n", conf_matrix)

evaluate_model_on_cat(model, df_test, y_test)
plot_cat(model, df_test, y_test)

#%% tuning bagging

best_score = 0
best_params = None
best_model = None

for n_estimators in [50, 100, 200]:
    for max_samples in [0.5, 0.7, 1.0]:
        for max_features in [0.5, 0.7, 1.0]:
            base_model = DecisionTreeRegressor(random_state=42)
            bagging_model = BaggingRegressor(
                estimator=base_model,
                n_estimators=n_estimators,
                max_samples=max_samples,
                max_features=max_features,
                random_state=42,
                n_jobs=-1
            )
            bagging_model.fit(df_train, y_train)
            val_preds = bagging_model.predict(df_val)
            val_preds = categorize_pred(val_preds)
            score = accuracy_score(y_val, val_preds)
            print(f"n_estimators: {n_estimators}, max_samples: {max_samples}, max_features: {max_features}, "
                  f"Custom Score: {score}\n")
            if score > best_score:
                best_score = score
                best_params = {
                    "n_estimators": n_estimators,
                    "max_samples": max_samples,
                    "max_features": max_features,
                }
                best_model = bagging_model

print(f"\nBest Params: {best_params}")
print(f"Best Validation Custom Score: {best_score}")
# Best Params: {'n_estimators': 100, 'max_samples': 0.7, 'max_features': 0.5}
# Best Validation Custom Score: 0.5139372822299652

#%% tuned bagging

base_model = DecisionTreeRegressor(random_state=42)
bagging_model = BaggingRegressor(
    estimator=base_model,
    n_estimators=100,
    max_samples=0.7,
    max_features=0.5,
    random_state=42,
    n_jobs=-1
)

model.fit(df_train, y_train)
y_pred = model.predict(df_test)
y_pred = categorize_pred(y_pred)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy) # Accuracy: 0.5158712034711121
print("Confusion Matrix:\n", conf_matrix)

evaluate_model_on_cat(model, df_test, y_test)
plot_cat(model, df_test, y_test)
