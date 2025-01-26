#%% packages

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

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

#%% evaluation

def evaluate_model_on_inc(model, x, y):
    y_pred = model.predict(x)
    results = {}
    name_columns = [col for col in x.columns if col.startswith("name")]
    for col in name_columns:
        name = col.lstrip("name_")
        name_index = x[x[f"name_{name}"] == 1].index
        
        y_inc = y[name_index]
        pred_inc = y_pred.reshape(-1,)[name_index]
        
        mse = mean_squared_error(y_inc, pred_inc)
        r2 = r2_score(y_inc, pred_inc)
        results[name] = {"mse": mse, "r2": r2}
    return results

def evaluate_model_on_cat(model, x, y):
    y_pred = model.predict(x)
    results = {}
    name_columns = [col for col in x.columns if col.startswith("name")]
    for col in name_columns:
        name = col.lstrip("name_")
        name_index = x[x[f"name_{name}"] == 1].index
        
        y_inc = y[name_index]
        y_inc[y_inc >= 0] = 1
        y_inc[y_inc < 0] = 0
        
        pred_inc = y_pred[name_index]
        pred_inc[pred_inc >= 0] = 1
        pred_inc[pred_inc < 0] = 0
        
        acc = 1 - mean_squared_error(y_inc, pred_inc)
        r2 = r2_score(y_inc, pred_inc)
        results[name] = {"acc": acc, "r2": r2}
    return results

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

#%% plotting results

import matplotlib.pyplot as plt

def plot_inc(model, x, y):
    y_pred = model.predict(x)
    name_columns = [col for col in x.columns if col.startswith("name")]
    for col in name_columns:
        name = col.lstrip("name_")
            
        name_index = x[x[f"name_{name}"] == 1].index
        y_inc = y[name_index]
        pred_inc = y_pred.reshape(-1,)[name_index]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(name_index, y_inc, alpha=0.7, label='original', color='blue')
        plt.scatter(name_index, pred_inc, label='pred', alpha=0.7, color='orange')
        plt.title(f'{name} close price return prediction', fontsize=16)
        plt.xlabel('index', fontsize=12)
        plt.ylabel('close price', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()
      
def plot_cat(model, x, y):
    y_pred = model.predict(x)
    name_columns = [col for col in df_test.columns if col.startswith("name")]
    for col in name_columns:
        name = col.lstrip("name_")
            
        name_index = x[x[f"name_{name}"] == 1].index
        y_cat = y[name_index]
        y_cat[y_cat >= 0] = 1
        y_cat[y_cat < 0] = 0
        
        pred_cat = y_pred[name_index]
        pred_cat[pred_cat >= 0] = 1
        pred_cat[pred_cat < 0] = 0
        
        plt.figure(figsize=(10, 6))
        plt.scatter(name_index, y_cat, alpha=0.7, label='original', color='blue')
        plt.scatter(name_index, pred_cat, label='pred', alpha=0.7, color='orange')
        plt.title(f'{name} close price prediction', fontsize=16)
        plt.xlabel('index', fontsize=12)
        plt.ylabel('close price', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()

#%% linear regression

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(df_train, y_train)
y_pred = model.predict(df_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

evaluate_model_on_inc(model, df_test, y_test)
plot_inc(model, df_test, y_test)

evaluate_model_on_cat(model, df_test, y_test)
plot_cat(model, df_test, y_test)

count_acc(model, df_test, y_test)

#%% parameters for ElasticNet mse

from sklearn.linear_model import ElasticNet

alphas = [3, 5, 8, 10, 11, 12]
l1_ratios = [0.5, 0.8, 0.9]

best_params = {}
best_val_mse = float('inf')
best_model = None

for alpha in alphas:
    for l1_ratio in l1_ratios:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(df_train, y_train)
        y_val_pred = model.predict(df_val)
        val_mse = mean_squared_error(y_val, y_val_pred)
        print(f"Alpha: {alpha}, L1 Ratio: {l1_ratio}, Validation MSE: {val_mse}")
        print()
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}
            best_model = model

print("Best Parameters:", best_params)
print("Best Validation MSE:", best_val_mse)

#%% tuned ElasticNet

model = ElasticNet(alpha=3, l1_ratio=0.8, random_state=42)
model.fit(df_train, y_train)
y_pred = model.predict(df_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

evaluate_model_on_inc(model, df_test, y_test)
plot_inc(model, df_test, y_test)

evaluate_model_on_cat(model, df_test, y_test)
plot_cat(model, df_test, y_test)

count_acc(model, df_test, y_test)

#%% RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
model.fit(df_train, y_train)
y_pred = model.predict(df_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

evaluate_model_on_inc(model, df_test, y_test)
plot_inc(model, df_test, y_test)

evaluate_model_on_cat(model, df_test, y_test)
plot_cat(model, df_test, y_test)

count_acc(model, df_test, y_test)

#%% fine-tuning random forest

best_score = float('inf')
best_params = None
best_model = None

n_estimators_list = [100, 200, 300]
max_depths = [10, 20, None]

for n_estimators in n_estimators_list:
    for max_depth in max_depths:
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        rf.fit(df_train, y_train)
        val_preds = rf.predict(df_val)
        score = mean_squared_error(y_val, val_preds)
        print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, MSE: {score}")
        print()
        if score < best_score:
            best_score = score
            best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
            best_model = rf

print(f"\nBest Params: {best_params}, Best Validation MSE: {best_score}")

#%% tuned RandomForestRegressor

model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=10, n_jobs=-1)
model.fit(df_train, y_train)
y_pred = model.predict(df_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

evaluate_model_on_inc(model, df_test, y_test)
plot_inc(model, df_test, y_test)

evaluate_model_on_cat(model, df_test, y_test)
plot_cat(model, df_test, y_test)

count_acc(model, df_test, y_test)

#%% xgboost

import xgboost as xgb

model = xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=5, n_jobs=-1)
model.fit(df_train, y_train)
y_pred = model.predict(df_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

evaluate_model_on_inc(model, df_test, y_test)
plot_inc(model, df_test, y_test)

evaluate_model_on_cat(model, df_test, y_test)
plot_cat(model, df_test, y_test)

count_acc(model, df_test, y_test)

#%% xgboost tuning

best_score = float('inf')
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
                        score = mean_squared_error(y_val, val_preds)
                        print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, learning_rate: {learning_rate}, "
                              f"alpha: {alpha}, gamma: {gamma}, MSE: {score}")
                        print()                        
                        if score < best_score:
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

#%% tuned xgboost

import xgboost as xgb

model = xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=7, learning_rate=0.01, alpha=1, gamma=1, n_jobs=-1)
model.fit(df_train, y_train)
y_pred = model.predict(df_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

evaluate_model_on_inc(model, df_test, y_test)
plot_inc(model, df_test, y_test)

evaluate_model_on_cat(model, df_test, y_test)
plot_cat(model, df_test, y_test)

count_acc(model, df_test, y_test)

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

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

evaluate_model_on_inc(model, df_test, y_test)
plot_inc(model, df_test, y_test)

evaluate_model_on_cat(model, df_test, y_test)
plot_cat(model, df_test, y_test)

count_acc(model, df_test, y_test)

#%% tuning bagging

best_score = float('inf')
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
            score = mean_squared_error(y_val, val_preds)
            
            print(f"n_estimators: {n_estimators}, max_samples: {max_samples}, max_features: {max_features}, "
                  f"MSE: {score}")
            
            if score < best_score:
                best_score = score
                best_params = {
                    "n_estimators": n_estimators,
                    "max_samples": max_samples,
                    "max_features": max_features,
                }
                best_model = bagging_model

print(f"\nBest Params: {best_params}")
print(f"Best Validation MSE: {best_score}")

#%% tuned bagging

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

base_model = DecisionTreeRegressor(random_state=42)
bagging_model = BaggingRegressor(
    estimator=base_model,
    n_estimators=200,
    max_samples=0.5,
    max_features=0.5,
    random_state=42,
    n_jobs=-1
)

model.fit(df_train, y_train)
y_pred = model.predict(df_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

evaluate_model_on_inc(model, df_test, y_test)
plot_inc(model, df_test, y_test)

evaluate_model_on_cat(model, df_test, y_test)
plot_cat(model, df_test, y_test)

count_acc(model, df_test, y_test)

#%% NNs

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    Dense(256, input_dim=df_train.shape[1], activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(1)
])

initial_learning_rate = 0.001
decay_rate = 0.98
decay_steps = 100

lr_schedule = ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=True
)

model.compile(optimizer=Adam(learning_rate=lr_schedule),
              loss='mse',
              metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=np.inf, restore_best_weights=True)

history = model.fit(df_train, y_train,
                    validation_data=(df_val, y_val),
                    epochs=100,
                    batch_size=32,
                    callbacks=[early_stopping],
                    verbose=1)

y_pred = model.predict(df_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

evaluate_model_on_inc(model, df_test, y_test)
plot_inc(model, df_test, y_test)

evaluate_model_on_cat(model, df_test, y_test)
plot_cat(model, df_test, y_test)

count_acc(model, df_test, y_test)
