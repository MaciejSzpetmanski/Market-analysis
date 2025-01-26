#%% packages

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
import warnings


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
names = [col.lstrip("name_") for col in  name_columns]
data_x_train, data_y_train = split_data(df_train, y_train, name_columns)
data_x_val, data_y_val = split_data(df_val, y_val, name_columns)
data_x_test, data_y_test = split_data(df_test, y_test, name_columns)

#%% evaluation

def count_acc(model, x, y):
    y_pred = model.predict(x)
    
    y_inc = y.copy()
    y_inc[y_inc >= 0] = 1
    y_inc[y_inc < 0] = 0
    
    pred_inc = y_pred.copy()
    pred_inc[pred_inc >= 0] = 1
    pred_inc[pred_inc < 0] = 0
    
    acc = 1 - mean_squared_error(y_inc, pred_inc)
    
    return acc

def count_acc_pred(y_pred, y):
    y_inc = y.copy()
    y_inc[y_inc >= 0] = 1
    y_inc[y_inc < 0] = 0
    
    pred_inc = y_pred.copy()
    pred_inc[pred_inc >= 0] = 1
    pred_inc[pred_inc < 0] = 0
    
    acc = 1 - mean_squared_error(y_inc, pred_inc)
    
    return acc

#%% plot prediction

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
        
def make_plot(y, pred):
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

name = "AAPL"
test = data_x_test[name]["close"]

forecast = []
width = 20
y_test = data_x_test[name]["close"][width:-1]
for i in range(0, len(test)-width-1):
    x = test[i:i+width]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auto_model = auto_arima(x, seasonal=False, stepwise=True, trace=False)
        model = ARIMA(x, order=auto_model.order)
        model_fit = model.fit()
    
    pred = model_fit.forecast(steps=1)
    forecast.append(pred)

mae = mean_absolute_error(y_test, forecast)
print(f"MAE: {mae}")

# plot results
plt.figure(figsize=(10,5))
plt.plot(y_test, label="Actual Prices", color='green')
plt.plot(y_test.index, forecast, label="Predicted Prices", color='red')
plt.legend()
plt.title("ARIMA stock price prediction")
plt.show()

len(forecast)
x_prev = data_x_test[name]["close"][width-1:-2]

y_inc = (y_test.reset_index(drop=True) - x_prev.reset_index(drop=True)) / x_prev.reset_index(drop=True)
pred_inc = (np.array(forecast).reshape(-1) - x_prev.reset_index(drop=True)) / x_prev.reset_index(drop=True)
mae = mean_absolute_error(y_inc, pred_inc)
print(f"MAE on returns: {mae}")

count_acc_pred(pred_inc, y_inc)

#%%

width = 20
accuracy = []

for name in names:
    print(name)
    test = data_x_test[name]["close"]
    forecast = []
    y_test = data_x_test[name]["close"][width:-1]
    for i in range(0, len(test)-width-1):
        x = test[i:i+width]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auto_model = auto_arima(x, seasonal=False, stepwise=True, trace=False)
            model = ARIMA(x, order=auto_model.order)
            model_fit = model.fit()
        
        pred = model_fit.forecast(steps=1)
        forecast.append(pred)
    
    mae = mean_absolute_error(y_test, forecast)
    print(f"MAE: {mae}")
    
    make_plot(y_test, forecast)
    
    plt.figure(figsize=(10,5))
    plt.plot(y_test, label="Actual Prices", color='green')
    plt.plot(y_test.index, forecast, label="Predicted Prices", color='red')
    plt.legend()
    plt.title("ARIMA stock price prediction")
    plt.show()

    x_prev = data_x_test[name]["close"][width-1:-2]
    y_inc = (y_test.reset_index(drop=True) - x_prev.reset_index(drop=True)) / x_prev.reset_index(drop=True)
    pred_inc = (np.array(forecast).reshape(-1) - x_prev.reset_index(drop=True)) / x_prev.reset_index(drop=True)
    mae = mean_absolute_error(y_inc, pred_inc)
    print(f"MAE on returns: {mae}")

    acc = count_acc_pred(pred_inc, y_inc)
    print(f"Accuracy: {acc}")
    accuracy.append(acc)
    print()

print(f"Total accuracy: {np.mean(accuracy)}")

