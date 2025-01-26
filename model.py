#%% packages

# analyzing linear regression model for different target horizons

import os
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet

#%% reading data

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


#%% evaluate models on all the sets

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

#%% k=1

model1 = ElasticNet(alpha=0.003, l1_ratio=0.8, random_state=42)
model1.fit(df_train_1, y_train_1)

y_pred1 = model1.predict(df_test_1)

mse1 = mean_squared_error(y_test_1, y_pred1)
r2_1 = r2_score(y_test_1, y_pred1)

print(f"MSE: {mse1}")
print(f"R2: {r2_1}")

print("Intercept:", model1.intercept_)

eval_results_1 = evaluate_model(model1, x_test_list, y_test_list)
print(eval_results_1)
full_eval_results_1 = evaluate_model_by_names(model1, x_test_list, y_test_list)
print_eval_results(full_eval_results_1)

plot_prediction_by_names(model1, x_test_list, y_test_list)

non_zero_features_1 = df_train_1.columns[model1.coef_ != 0]
print(non_zero_features_1)

for name, coef in zip(df_train_1.columns, model1.coef_):
    if coef != 0:
        print(f"{name}: {coef}")
        
joblib.dump(model1, "models/elasticnet_model_1.pkl")
model1 = joblib.load("models/elasticnet_model_1.pkl")

#%% k=2

model2 = ElasticNet(alpha=0.003, l1_ratio=0.8, random_state=42)
model2.fit(df_train_2, y_train_2)

y_pred2 = model2.predict(df_test_2)

mse2 = mean_squared_error(y_test_2, y_pred2)
r2_2 = r2_score(y_test_2, y_pred2)

print(f"MSE: {mse2}")
print(f"R2: {r2_2}")

print("Intercept:", model2.intercept_)

eval_results_2 = evaluate_model(model2, x_test_list, y_test_list)
print(eval_results_2)
full_eval_results_2 = evaluate_model_by_names(model2, x_test_list, y_test_list)
print_eval_results(full_eval_results_2)

plot_prediction_by_names(model2, x_test_list, y_test_list)

non_zero_features_2 = df_train_2.columns[model2.coef_ != 0]
print(non_zero_features_2)

for name, coef in zip(df_train_2.columns, model2.coef_):
    if coef != 0:
        print(f"{name}: {coef}")
        
joblib.dump(model2, "models/elasticnet_model_2.pkl")
model2 = joblib.load("models/elasticnet_model_2.pkl")


#%% k=5

model5 = ElasticNet(alpha=0.003, l1_ratio=0.8, random_state=42)

model5.fit(df_train_5, y_train_5)

y_pred5 = model5.predict(df_test_5)

mse5 = mean_squared_error(y_test_5, y_pred5)
r2_5 = r2_score(y_test_5, y_pred5)

print(f"MSE: {mse5}")
print(f"R2: {r2_5}")

print("Intercept:", model5.intercept_)

eval_results_5 = evaluate_model(model5, x_test_list, y_test_list)
print(eval_results_5)
full_eval_results_5 = evaluate_model_by_names(model5, x_test_list, y_test_list)
print_eval_results(full_eval_results_5)

plot_prediction_by_names(model5, x_test_list, y_test_list)

non_zero_features_5 = df_train_5.columns[model5.coef_ != 0]
print(non_zero_features_5)

for name, coef in zip(df_train_5.columns, model5.coef_):
    if coef != 0:
        print(f"{name}: {coef}")
        
joblib.dump(model5, "models/elasticnet_model_5.pkl")
model5 = joblib.load("models/elasticnet_model_5.pkl")
