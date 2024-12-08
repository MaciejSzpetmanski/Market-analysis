#%% working directory

import os
import joblib

path = "D:\Studia\semestr7\inźynierka\Market-analysis"
# path = "C:\Studia\Market-analysis"
os.chdir(path)

#%% reading data

import pandas as pd
import numpy as np

# df_train = pd.read_csv("datasets/df_train.csv")
# df_val = pd.read_csv("datasets/df_val.csv")
# df_test = pd.read_csv("datasets/df_test.csv")

# y_train = pd.read_csv("datasets/y_train.csv")
# y_val = pd.read_csv("datasets/y_val.csv")
# y_test = pd.read_csv("datasets/y_test.csv")

def load_dataset(directory_name, name, target_horizon):
    df_file_path = os.path.join(directory_name, f"df_{name}_{target_horizon}.csv")
    y_file_path = os.path.join(directory_name, f"y_{name}_{target_horizon}.csv")
    if os.path.isfile(df_file_path) and os.path.isfile(y_file_path):
        df = pd.read_csv(df_file_path)
        y = pd.read_csv(y_file_path)
        return df, y

directory_name = "datasets"

#%%

df_train_1, y_train_1 = load_dataset(directory_name, "train", 1)
df_train_2, y_train_2 = load_dataset(directory_name, "train", 2)
df_train_3, y_train_3 = load_dataset(directory_name, "train", 3)
df_train_4, y_train_4 = load_dataset(directory_name, "train", 4)
df_train_5, y_train_5 = load_dataset(directory_name, "train", 5)

df_val_1, y_val_1 = load_dataset(directory_name, "val", 1)
df_val_2, y_val_2 = load_dataset(directory_name, "val", 2)
df_val_3, y_val_3 = load_dataset(directory_name, "val", 3)
df_val_4, y_val_4 = load_dataset(directory_name, "val", 4)
df_val_5, y_val_5 = load_dataset(directory_name, "val", 5)

df_test_1, y_test_1 = load_dataset(directory_name, "test", 1)
df_test_2, y_test_2 = load_dataset(directory_name, "test", 2)
df_test_3, y_test_3 = load_dataset(directory_name, "test", 3)
df_test_4, y_test_4 = load_dataset(directory_name, "test", 4)
df_test_5, y_test_5 = load_dataset(directory_name, "test", 5)

x_train_list = [df_train_1, df_train_2, df_train_3, df_train_4, df_train_5]
y_train_list = [y_train_1, y_train_2, y_train_3, y_train_4, y_train_5]

x_val_list = [df_val_1, df_val_2, df_val_3, df_val_4, df_val_5]
y_val_list = [y_val_1, y_val_2, y_val_3, y_val_4, y_val_5]

x_test_list = [df_test_1, df_test_2, df_test_3, df_test_4, df_test_5]
y_test_list = [y_test_1, y_test_2, y_test_3, y_test_4, y_test_5]

#%% train

df_train = df_train_1
y_train = y_train_1

for col in df_train.columns:
    if (df_train[col] == y_train.y).all():
        print("yes")
        
#%% val

df_val = df_val_1
y_val = y_val_1

for col in df_val.columns:
    if (df_val[col] == y_val.y).all():
        print("yes")
        
#%% test

df_test = df_test_1
y_test = y_test_1

for col in df_test.columns:
    if (df_test[col] == y_test.y).all():
        print("yes")

#%% getting full data - later

def load_data_with_max_horizon(directory_name, name, max_target_horizon):
    df = None
    y = None
    for k in range(1, max_target_horizon + 1):
        df_input, y_input = load_dataset(directory_name, name, k)
        df = pd.concat([df, df_input], ignore_index=True)
        y = pd.concat([y, y_input], ignore_index=True)
    return df, y

df_train, y_train = load_data_with_max_horizon(directory_name, "train", 5)
df_val, y_val = load_data_with_max_horizon(directory_name, "val", 5)
df_test, y_test = load_data_with_max_horizon(directory_name, "test", 5)

#%% merging data

def merge_sets(k, x_sets_list, y_sets_list):
    x = pd.concat(x_sets_list[:k], ignore_index=True)
    y = pd.concat(y_sets_list[:k], ignore_index=True)
    return x, y

#%% preparing trimmed data

k = 1
# k = 2
# k = 3
# k = 4
# k = 5

df_train, y_train = merge_sets(k, x_train_list, y_train_list)
df_val, y_val = merge_sets(k, x_val_list, y_val_list)
df_test, y_test = merge_sets(k, x_test_list, y_test_list)

#%% evaluate models on all the sets

from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, x_test_list, y_test_list):
    res = {}
    for i in range(len(x_test_list)):
        x = x_test_list[i]
        y = y_test_list[i]
        y_pred = model.predict(x)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        res[i] = {"mse": mse, "r2": r2}
    return res

def evaluate_model_by_names(model, x_test_list, y_test_list):
    name_columns = [col for col in x_test_list[0].columns if col.startswith("name")]
    results = {}
    for col in name_columns:
        x_test_list_filtered = [x_test[x_test[col] == 1] for x_test in x_test_list]
        indexes = [x_test.index for x_test in x_test_list_filtered]
        y_test_list_filtered = [y_test.iloc[index_list] for index_list, y_test in zip(indexes, y_test_list)]
        eval_res = evaluate_model(model, x_test_list_filtered, y_test_list_filtered)
        name = col.lstrip("name_")
        results[name] = eval_res
    return results

def print_eval_results(eval_results):
    for key, value in eval_results.items():
        print(key)
        print(value)
        print()

#%% plot results for one name

import matplotlib.pyplot as plt

def plot_prediction(name, model, x, y):
    name_index = x[x[f"name_{name}"] == 1].index
    y_name = y.iloc[name_index]
    y_pred = model.predict(x)
    pred_name = y_pred[name_index]
    plt.figure(figsize=(10, 6))
    plt.scatter(name_index, y_name, alpha=0.7, label='original', color='blue')
    plt.scatter(name_index, pred_name, label='pred', alpha=0.7, color='orange')
    plt.title(f'Scatter Plot of {name}', fontsize=16)
    plt.xlabel('', fontsize=12)
    plt.ylabel('', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_all_prediction(name, model, x_test_list, y_test_list):
    for i in range(len(x_test_list)):
        x = x_test_list[i]
        y = y_test_list[i]
        plot_prediction(name, model, x, y)
        
def plot_prediction_by_names(model, x_test_list, y_test_list):
    name_columns = [col for col in x_test_list[0].columns if col.startswith("name")]
    for col in name_columns:
        name = col.lstrip("name_")
        plot_all_prediction(name, model, x_test_list, y_test_list)

#%% linear regression

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(df_train, y_train)

y_pred = model.predict(df_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Display model coefficients
# print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

eval_results = evaluate_model(model, x_test_list, y_test_list)
print(eval_results)
full_eval_results = evaluate_model_by_names(model, x_test_list, y_test_list)
print_eval_results(full_eval_results)

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)
plot_prediction_by_names(model, x_test_list, y_test_list)

#%% parameters for ElasticNet

from sklearn.linear_model import ElasticNet

# alphas = [0.01, 0.1, 1, 10]
# l1_ratios = [0.2, 0.5, 0.8]

alphas = [0.001, 0.003, 0.005, 0.008, 0.01, 0.02]
l1_ratios = [0.2, 0.5, 0.8, 0.9]

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
        eval_results = evaluate_model(model, x_val_list, y_val_list)
        print(eval_results)
        print()
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}
            best_model = model

print("Best Parameters:", best_params)
print("Best Validation MSE:", best_val_mse)

# Best Parameters: {'alpha': 0.01, 'l1_ratio': 0.8}
# Best Parameters: {'alpha': 0.003, 'l1_ratio': 0.8}

#%% tuned ElasticNet

model = ElasticNet(alpha=0.003, l1_ratio=0.8, random_state=42)
model.fit(df_train, y_train)

y_pred = model.predict(df_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

eval_results = evaluate_model(model, x_test_list, y_test_list)
print(eval_results)
full_eval_results = evaluate_model_by_names(model, x_test_list, y_test_list)
print_eval_results(full_eval_results)

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)
plot_prediction_by_names(model, x_test_list, y_test_list)

non_zero_features = df_train.columns[model.coef_ != 0]
print(non_zero_features)

for name, coef in zip(df_train.columns, model.coef_):
    if coef != 0:
        print(f"{name}: {coef:.4f}")
        
joblib.dump(model, "models/elasticnet_model.pkl")
model = joblib.load("models/elasticnet_model.pkl")

#%% random forest

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)

model.fit(df_train, y_train)

y_pred = model.predict(df_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

eval_results = evaluate_model(model, x_test_list, y_test_list)
print(eval_results)
full_eval_results = evaluate_model_by_names(model, x_test_list, y_test_list)
print_eval_results(full_eval_results)

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)
plot_prediction_by_names(model, x_test_list, y_test_list)

#%% fine-tuning random forest

from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 20, None],
# }

param_grid = {
    'n_estimators': [250, 300, 350],
    'max_depth': [10, 20, None],
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=[(slice(None), slice(None))],
    n_jobs=-1
)

grid_search.fit(df_train, y_train)
val_mse = mean_squared_error(y_val, grid_search.best_estimator_.predict(df_val))
val_r2 = r2_score(y_val, grid_search.best_estimator_.predict(df_val))

print(f"Best Params: {grid_search.best_params_}, Validation MSE: {val_mse}, R-squared: {val_r2}")

# Best Params: {'max_depth': None, 'n_estimators': 300}, Validation MSE: 0.00883993516026214, R-squared: 0.9951073534338416
# Best Params: {'max_depth': None, 'n_estimators': 350}, Validation MSE: 0.008807718106650759, R-squared: 0.9951251846343953

#%% custom metric

def custom_score(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return r2 - mse

best_score = float('-inf')
best_params = None
best_model = None

for n_estimators in [100, 200, 300]:
    for max_depth in [10, 20, None]:
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        rf.fit(df_train, y_train)
        
        val_preds = rf.predict(df_val)
        score = custom_score(y_val, val_preds)
        
        print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, Custom Score: {score}")
        
        if score > best_score:
            best_score = score
            best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
            best_model = rf

print(f"\nBest Params: {best_params}, Best Validation Custom Score: {best_score}")
val_mse = mean_squared_error(y_val, grid_search.best_estimator_.predict(df_val))
val_r2 = r2_score(y_val, grid_search.best_estimator_.predict(df_val))
print(f"Best Params: {grid_search.best_params_}, Validation MSE: {val_mse}, R-squared: {val_r2}")

# Best Params: {'n_estimators': 300, 'max_depth': 10}, Best Validation Custom Score: 0.9865146257774043

#%% tuned random forest

model = RandomForestRegressor(n_estimators=350, random_state=42, max_depth=None, n_jobs=-1)
model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=10, n_jobs=-1)
model.fit(df_train, y_train)

y_pred = model.predict(df_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

eval_results = evaluate_model(model, x_test_list, y_test_list)
print(eval_results)
full_eval_results = evaluate_model_by_names(model, x_test_list, y_test_list)
print_eval_results(full_eval_results)

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)
plot_prediction_by_names(model, x_test_list, y_test_list)

non_zero_features = df_train.columns[model.coef_ != 0]
print(non_zero_features)

joblib.dump(model, "models/random_forest_model.pkl")
model = joblib.load("models/random_forest_model.pkl")

#%% xgboost

import xgboost as xgb
import matplotlib.pyplot as plt

model = xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=5, n_jobs=-1)

model.fit(df_train, y_train)

y_pred = model.predict(df_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

eval_results = evaluate_model(model, x_test_list, y_test_list)
print(eval_results)
full_eval_results = evaluate_model_by_names(model, x_test_list, y_test_list)
print_eval_results(full_eval_results)

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)
plot_prediction_by_names(model, x_test_list, y_test_list)

#%% trim

xgb.plot_importance(model)
plt.show()

#%% xgboost tunung

from sklearn.model_selection import GridSearchCV

def custom_score(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return r2 - mse  

best_score = float('-inf')
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
                        score = custom_score(y_val, val_preds)
                        
                        print(f"n_estimators: {n_estimators}, max_depth: {max_depth}, learning_rate: {learning_rate}, "
                              f"alpha: {alpha}, gamma: {gamma}, Custom Score: {score}")
                        
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
print(f"Best Validation Custom Score: {best_score}")
# Best Params: {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'alpha': 0.1, 'gamma': 1}
# Best Validation Custom Score: 0.9836375966155018

#%% tuned xgboost

model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, alpha=0.1, gamma=1, n_jobs=-1)
model.fit(df_train, y_train)

y_pred = model.predict(df_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

eval_results = evaluate_model(model, x_test_list, y_test_list)
print(eval_results)
full_eval_results = evaluate_model_by_names(model, x_test_list, y_test_list)
print_eval_results(full_eval_results)

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)
plot_prediction_by_names(model, x_test_list, y_test_list)

joblib.dump(model, "models/xgboost_model.pkl")
model = joblib.load("models/xgboost_model.pkl")

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

eval_results = evaluate_model(model, x_test_list, y_test_list)
print(eval_results)
full_eval_results = evaluate_model_by_names(model, x_test_list, y_test_list)
print_eval_results(full_eval_results)

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)
plot_prediction_by_names(model, x_test_list, y_test_list)

#%% tuning bagging

def custom_score(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return r2 - mse

best_score = float('-inf')
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
            score = custom_score(y_val, val_preds)
            
            print(f"n_estimators: {n_estimators}, max_samples: {max_samples}, max_features: {max_features}, "
                  f"Custom Score: {score}")
            
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
# Best Params: {'n_estimators': 200, 'max_samples': 0.5, 'max_features': 1.0}
# Best Validation Custom Score: 0.9883401679468514

#%% tuned bagging

base_model = DecisionTreeRegressor(random_state=42)

model = BaggingRegressor(estimator=base_model, 
                        n_estimators=200, max_samples=0.5, max_features=1.0,
                        n_jobs=-1)
model.fit(df_train, y_train)

y_pred = model.predict(df_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

eval_results = evaluate_model(model, x_test_list, y_test_list)
print(eval_results)
full_eval_results = evaluate_model_by_names(model, x_test_list, y_test_list)
print_eval_results(full_eval_results)

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)
plot_prediction_by_names(model, x_test_list, y_test_list)

joblib.dump(model, "models/bagging_model.pkl")
model = joblib.load("models/bagging_model.pkl")

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

# test_loss, test_mae = model.evaluate(df_test, y_test)
# print(f"Test Loss (MSE): {test_loss}")
# print(f"Test MAE: {test_mae}")

y_pred = model.predict(df_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

eval_results = evaluate_model(model, x_test_list, y_test_list)
print(eval_results)
full_eval_results = evaluate_model_by_names(model, x_test_list, y_test_list)
print_eval_results(full_eval_results)

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)
plot_prediction_by_names(model, x_test_list, y_test_list)

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

model.save("models/nn.keras")

from tensorflow.keras.models import load_model

model = load_model("models/nn.keras")

#%% plot results - on day forward

for col in df_test.columns:
    if col.startswith("date"):
        print(col)


date_last = pd.to_datetime(
    df_test.rename(columns={
        'date_year_19': 'year', 
        'date_month_19': 'month', 
        'date_day_of_month_19': 'day'
    })[['year', 'month', 'day']]
)

date_y = pd.to_datetime(
    df_test.rename(columns={
        'date_year_y': 'year', 
        'date_month_y': 'month', 
        'date_day_of_month_y': 'day'
    })[['year', 'month', 'day']]
)

date_y - date_last

(date_y - date_last).value_counts()

# only first period (short-term) - AAPL
test_index = df_test.drop_duplicates(df_test.columns[:6])[df_test.name_AAPL == 1].index

test_series = y_test.iloc[test_index]
pred_series = y_pred[test_index]

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(test_index, test_series, alpha=0.7, label='test', color='blue')
plt.scatter(test_index, pred_series, label='pred', alpha=0.7, color='orange')
plt.title('Scatter Plot of AAPL', fontsize=16)
plt.xlabel('', fontsize=12)
plt.ylabel('', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()





