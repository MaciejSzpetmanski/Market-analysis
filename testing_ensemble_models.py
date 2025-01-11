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
names = [col.lstrip("name_") for col in name_columns]
data_x_train, data_y_train = split_data(df_train, y_train, name_columns)
data_x_val, data_y_val = split_data(df_val, y_val, name_columns)
data_x_test, data_y_test = split_data(df_test, y_test, name_columns)

#%% synchronizing dates

def add_date_column(df):
    selected_date_columns = ['date_year_y', 'date_month_y', 'date_day_of_month_y']
    df['date'] = ''
    for date_column in sorted(selected_date_columns, reverse=True):
        df['date'] += "-" + df[date_column].astype(int).astype(str)
    return df

for name in names:
    add_date_column(data_x_train[name])
    add_date_column(data_x_val[name])
    add_date_column(data_x_test[name])

common_train_dates = data_x_train[names[0]][['date']]
common_val_dates = data_x_val[names[0]][['date']]
common_test_dates = data_x_test[names[0]][['date']]
for name in names:
    common_train_dates = pd.merge(data_x_train[name]['date'], common_train_dates, on='date', how='inner')
    common_val_dates = pd.merge(data_x_val[name]['date'], common_val_dates, on='date', how='inner')
    common_test_dates = pd.merge(data_x_test[name]['date'], common_test_dates, on='date', how='inner')

# selecting rows with common dates
data_x_train_trim = {}
data_x_val_trim = {}
data_x_test_trim = {}

data_y_train_trim = {}
data_y_val_trim = {}
data_y_test_trim = {}
for name in names:
    data_x_train_trim[name] = data_x_train[name][data_x_train[name]['date'].isin(common_train_dates['date'])]
    data_y_train_trim[name] = data_y_train[name][data_x_train_trim[name].index]
    
    data_x_val_trim[name] = data_x_val[name][data_x_val[name]['date'].isin(common_val_dates['date'])]
    data_y_val_trim[name] = data_y_val[name][data_x_val_trim[name].index]
    
    data_x_test_trim[name] = data_x_test[name][data_x_test[name]['date'].isin(common_test_dates['date'])]
    data_y_test_trim[name] = data_y_test[name][data_x_test_trim[name].index]
    
    # droping date
    data_x_train[name] = data_x_train[name].drop(columns=['date'])
    data_x_val[name] = data_x_val[name].drop(columns=['date'])
    data_x_test[name] = data_x_test[name].drop(columns=['date'])
    
    data_x_train_trim[name] = data_x_train_trim[name].drop(columns=['date'])
    data_x_val_trim[name] = data_x_val_trim[name].drop(columns=['date'])
    data_x_test_trim[name] = data_x_test_trim[name].drop(columns=['date'])
    
#%% first level models

# loading models
base_models = {}
for name in names:
    base_models[name] = joblib.load(f"models/individual/{name}.pkl")
    
pred_x_train = {}
pred_x_val = {}
pred_x_test = {}
for name in names:
    model = base_models[name]
    pred_train = model.predict(data_x_train_trim[name])
    pred_x_train[name] = pred_train
    
    pred_val = model.predict(data_x_val_trim[name])
    pred_x_val[name] = pred_val
    
    pred_test = model.predict(data_x_test_trim[name])
    pred_x_test[name] = pred_test
    
pred_x_train = pd.DataFrame(pred_x_train)
pred_x_val = pd.DataFrame(pred_x_val)
pred_x_test = pd.DataFrame(pred_x_test)

# reset y index
for name in names:
    data_y_train_trim[name] = data_y_train_trim[name].reset_index(drop=True)
    data_y_val_trim[name] = data_y_val_trim[name].reset_index(drop=True)
    data_y_test_trim[name] = data_y_test_trim[name].reset_index(drop=True)

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
        
def plot_prediction_ensemble(models, x, y_set, names):
    for name in names:
        model = models[name]
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
    mse_lr, params_lr, model_lr = tune_ElasticNet(pred_x_train, data_y_train_trim[name],
                               pred_x_val, data_y_val_trim[name],
                               alphas, l1_ratios)
    mse_rf, params_rf, model_rf = tune_RandomForestRegressor(pred_x_train, data_y_train_trim[name],
                               pred_x_val, data_y_val_trim[name],
                               n_estimators_list, max_depths)
    mse_xgb, params_xgb, model_xgb = tune_xgboost(pred_x_train, data_y_train_trim[name],
                               pred_x_val, data_y_val_trim[name],
                               n_estimators_list, max_depths, learning_rates, alphas, gammas)
    mse_bag, params_bag, model_bag = tune_BaggingRegressor(pred_x_train, data_y_train_trim[name],
                               pred_x_val, data_y_val_trim[name],
                               n_estimators_list, max_samples_list, max_features_list)
    best_index = np.argmin([mse_lr, mse_rf, mse_xgb, mse_bag])
    mse = [mse_lr, mse_rf, mse_xgb, mse_bag][best_index]
    params = [params_lr, params_rf, params_xgb, params_bag][best_index]
    model = [model_lr, model_rf, model_xgb, model_bag][best_index]
    
    models[name] = model
    history[name] = {"mse": mse, "params": params}
    
for name in names:
    joblib.dump(models[name], f"models/ensemble/{name}.pkl")
    
#%% results

models = {}
for name in names:
    models[name] = joblib.load(f"models/ensemble/{name}.pkl")

for name in names:
    print(name)
    mse_base = mean_squared_error(base_models[name].predict(data_x_test[name]), data_y_test[name])
    mse = mean_squared_error(models[name].predict(pred_x_test), data_y_test_trim[name])
    print(f"MSE on base model: {mse_base}")
    print(f"MSE on final model: {mse}\n")
    
for name in names:
    print(name)
    acc_base = count_acc(base_models[name], data_x_test[name], data_y_test[name])
    acc = count_acc(models[name], pred_x_test, data_y_test_trim[name])
    print(f"Acc on base model: {acc_base}")
    print(f"Acc on final model: {acc}\n")

counter = 0
cum_acc = 0
for name in names:
    print(name)
    acc_base = count_acc(base_models[name], data_x_test[name], data_y_test[name])
    acc = count_acc(models[name], pred_x_test, data_y_test_trim[name])
    print(f"Best acc: {max(acc_base, acc)}")
    counter += 1
    cum_acc += max(acc_base, acc)
    
print(cum_acc / counter)
    

plot_prediction_ensemble(models, pred_x_test, data_y_test_trim, names)


