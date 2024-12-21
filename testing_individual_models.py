#%% working directory

import os
import joblib

path = "D:\Studia\semestr7\inźynierka\Market-analysis"
# path = "C:\Studia\Market-analysis"
os.chdir(path)

#%% categorize y

def categorize_y(x, y, pred_name="close"):
    last_column_name = [col for col in x.columns if col.startswith(pred_name)][-1]
    last_data = x[last_column_name]
    res = y - last_data
    res[res >= 0] = 1
    res[res < 0] = 0
    return res

#%%  y to increments

def y_to_increments(x, y, pred_name="close"):
    last_column_name = [col for col in x.columns if col.startswith(pred_name)][-1]
    last_data = x[last_column_name]
    res = (y - last_data) / last_data
    return res

#%% reading data (y as returns)

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

y_train = y_to_increments(df_train, y_train.y)
y_val = y_to_increments(df_val, y_val.y)
y_test = y_to_increments(df_test, y_test.y)

#%% spliiting data by name

def split_data(x, y, name_columns):
    res_set_x = {}
    res_set_y = {}
    for col in name_columns:
        name = col.lstrip("name_")
        x_name = x[x[f"name_{name}"] == 1]
        name_index = x_name.index
        y_name = y[name_index]
        res_set_x[name] = x_name.reset_index(drop=True)
        res_set_y[name] = y_name.reset_index(drop=True)
    return res_set_x, res_set_y

name_columns = [col for col in df_train.columns if col.startswith("name")]
data_x_train, data_y_train = split_data(df_train, y_train, name_columns)
data_x_val, data_y_val = split_data(df_val, y_val, name_columns)
data_x_test, data_y_test = split_data(df_test, y_test, name_columns)

#%% evaluation

from sklearn.metrics import mean_squared_error, r2_score

def count_acc(model, x, y):
    y_pred = model.predict(x)
    
    y_inc = y.copy()
    y_inc[y_inc >= 0] = 1
    y_inc[y_inc < 0] = 0
    
    pred_inc = y_pred
    pred_inc[pred_inc >= 0] = 1
    pred_inc[pred_inc < 0] = 0
    
    acc = 1 - mean_squared_error(y_inc, pred_inc)
    
    return acc

#%% plot prediction

import matplotlib.pyplot as plt

def plot_prediction(models, x_set, y_set, names):
    for name in names:
        model = models[name]
        x = x_set[name]
        y = y_set[name]
        pred = model.predict(x)
        plt.figure(figsize=(10, 6))
        plt.scatter(y.index, y, alpha=0.7, label='original', color='blue', s=8)
        plt.scatter(y.index, pred, label='pred', alpha=0.7, color='orange', s=8)
        plt.title(f'{name} close price prediction', fontsize=16)
        plt.xlabel('index', fontsize=12)
        plt.ylabel('close price', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()

#%%

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

def tune_ElasticNet(df_train, y_train, df_val, y_val, alphas, l1_ratios, random_state=42):
    best_mse = float('inf')
    best_params = None
    best_model = None
    for alpha in alphas:
        for l1_ratio in l1_ratios:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
            model.fit(df_train, y_train)
            y_val_pred = model.predict(df_val)
            val_mse = mean_squared_error(y_val, y_val_pred)
            print(f"Alpha: {alpha}, L1 Ratio: {l1_ratio}, Validation MSE: {val_mse}\n")
            if val_mse < best_mse:
                best_mse = val_mse
                best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}
                best_model = model
    return best_mse, best_params, best_model

def tune_RandomForestRegressor(df_train, y_train, df_val, y_val, n_estimators_list, max_depths, random_state=42):
    best_mse = float('inf')
    best_params = None
    best_model = None
    for n_estimators in n_estimators_list:
        for max_depth in max_depths:
            rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
            rf.fit(df_train, y_train)
            val_preds = rf.predict(df_val)
            mse = mean_squared_error(y_val, val_preds)
            print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, MSE: {mse}\n")
            if mse < best_mse:
                best_mse = mse
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
                best_model = rf
    return best_mse, best_params, best_model

def tune_xgboost(df_train, y_train, df_val, y_val, n_estimators_list, max_depths, learning_rates, alphas, gammas, random_state=42):
    best_mse = float('inf')
    best_params = None
    best_model = None
    for n_estimators in n_estimators_list:
        for max_depth in max_depths:
            for learning_rate in learning_rates:
                for alpha in alphas:
                        for gamma in gammas:
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
                            mse = mean_squared_error(y_val, val_preds)
                            print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, learning_rate: {learning_rate}, "
                                  f"alpha: {alpha}, gamma: {gamma}, MSE: {mse}\n")
                            print()                        
                            if mse < best_mse:
                                best_mse = mse
                                best_params = {
                                    "n_estimators": n_estimators,
                                    "max_depth": max_depth,
                                    "learning_rate": learning_rate,
                                    "alpha": alpha,
                                    "gamma": gamma
                                }
                                best_model = xgb_model
    return best_mse, best_params, best_model

def tune_BaggingRegressor(df_train, y_train, df_val, y_val, n_estimators_list, max_samples_list, max_features_list, random_state=42):
    best_mse = float('inf')
    best_params = None
    best_model = None
    for n_estimators in n_estimators_list:
        for max_samples in max_samples_list:
            for max_features in max_features_list:
                base_model = DecisionTreeRegressor(random_state=random_state)
                bagging_model = BaggingRegressor(
                    estimator=base_model,
                    n_estimators=n_estimators,
                    max_samples=max_samples,
                    max_features=max_features,
                    random_state=random_state,
                    n_jobs=-1
                )
                bagging_model.fit(df_train, y_train)
                val_preds = bagging_model.predict(df_val)
                mse = mean_squared_error(y_val, val_preds)
                print(f"n_estimators: {n_estimators}, max_samples: {max_samples}, max_features: {max_features}, "
                      f"MSE: {mse}\n")
                if mse < best_mse:
                    best_mse = mse
                    best_params = {
                        "n_estimators": n_estimators,
                        "max_samples": max_samples,
                        "max_features": max_features,
                    }
                    best_model = bagging_model
    return best_mse, best_params, best_model

#%% training models

models = {}
history = {}
name_columns = [col for col in df_train.columns if col.startswith("name")]
names = [col.lstrip("name_") for col in name_columns]

alphas = [0.01, 0.1, 1, 10]
l1_ratios = [0.2, 0.5, 0.8]

n_estimators_list = [100, 200, 300]
max_depths = [10, 20, None]

n_estimators_list = [100, 200, 300]
max_depths = [3, 5, 7]
learning_rates = [0.01, 0.1, 0.2]
alphas = [0, 0.1, 1]
gammas = [0, 0.1, 1]

n_estimators_list = [50, 100, 200]
max_samples_list = [0.5, 0.7, 1.0]
max_features_list = [0.5, 0.7, 1.0]

for col in name_columns:
    name = col.lstrip("name_")
    print(f"Tuning {name}")
    mse_lr, params_lr, model_lr = tune_ElasticNet(data_x_train[name], data_y_train[name],
                               data_x_val[name], data_y_val[name],
                               alphas, l1_ratios)
    mse_rf, params_rf, model_rf = tune_RandomForestRegressor(data_x_train[name], data_y_train[name],
                               data_x_val[name], data_y_val[name],
                               n_estimators_list, max_depths)
    mse_xgb, params_xgb, model_xgb = tune_xgboost(data_x_train[name], data_y_train[name],
                               data_x_val[name], data_y_val[name],
                               n_estimators_list, max_depths, learning_rates, alphas, gammas)
    mse_bag, params_bag, model_bag = tune_BaggingRegressor(data_x_train[name], data_y_train[name],
                               data_x_val[name], data_y_val[name],
                               n_estimators_list, max_samples_list, max_features_list)
    best_index = np.argmin([mse_lr, mse_rf, mse_xgb, mse_bag])
    mse = [mse_lr, mse_rf, mse_xgb, mse_bag][best_index]
    params = [params_lr, params_rf, params_xgb, params_bag][best_index]
    model = [model_lr, model_rf, model_xgb, model_bag][best_index]
    
    models[name] = model
    history[name] = {"mse": mse, "params": params}
    
mean_mse = np.mean([value["mse"] for key, value in history.items()])    

cum_acc = 0
for name in names:
    print(name)
    acc = count_acc(models[name], data_x_test[name], data_y_test[name])
    print(acc)
    cum_acc += acc
cum_acc /= len(names)
print(cum_acc)
        
plot_prediction(models, data_x_train, data_y_train, names)

for name in names:
    joblib.dump(models[name], "models/individual/name.pkl")
    # model = joblib.load("models/individual/name.pkl")


