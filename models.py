#%% working directory

import os

path = "D:\Studia\semestr7\inźynierka\Market-analysis"
# path = "C:\Studia\Market-analysis"
os.chdir(path)

#%% reading data

import pandas as pd

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
        # decide about error (cumulative or not)
        # x, y = merge_sets(i, x_test_list, y_test_list)
        x = x_test_list[i]
        y = y_test_list[i]
        y_pred = model.predict(x)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        res[i] = {"mse": mse, "r2": r2}
    return res

#%% plot results for one name

import matplotlib.pyplot as plt

def plot_prediction(name, model, x, y):
    # TODO add k and model name to title, possibly time
    name_index = x[x[f"name_{name}"] == 1].index
    y_name = y.iloc[name_index]
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

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)

#%% tree

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=42, max_depth=5)

model.fit(df_train, y_train)

y_pred = model.predict(df_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

feature_importances = model.feature_importances_
print("Feature Importances:")
for name, importance in zip(df_test.columns, feature_importances):
    if importance > 0:
        print(f"{name}: {importance:.4f}")
        
eval_results = evaluate_model(model, x_test_list, y_test_list)
print(eval_results)

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)

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

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)

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

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)

#%% trim

xgb.plot_importance(model)
plt.show()

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

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)

#%% SVR

from sklearn.svm import SVR

svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)

svr_model.fit(df_train, y_train)

y_pred = svr_model.predict(df_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

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





