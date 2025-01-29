#%% packages

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, roc_curve

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

#%% merging data

def merge_sets(k, x_sets_list, y_sets_list):
    x = pd.concat(x_sets_list[:k], ignore_index=True)
    y = pd.concat(y_sets_list[:k], ignore_index=True)
    return x, y

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

#%% preparing trimmed data

k = 1
df_train, y_train = merge_sets(k, x_train_list, y_train_list)
df_val, y_val = merge_sets(k, x_val_list, y_val_list)
df_test, y_test = merge_sets(k, x_test_list, y_test_list)

y_cat_train = categorize_y(df_train, y_train.y)
y_cat_val = categorize_y(df_val, y_val.y)
y_cat_test = categorize_y(df_test, y_test.y)

y_inc_train = y_to_increments(df_train, y_train.y)
y_inc_val = y_to_increments(df_val, y_val.y)
y_inc_test = y_to_increments(df_test, y_test.y)

#%% increment sets

y_train_inc_list = [y_to_increments(x, y.y) for x, y in zip(x_train_list, y_train_list)]
y_val_inc_list = [y_to_increments(x, y.y) for x, y in zip(x_val_list, y_val_list)]
y_test_inc_list = [y_to_increments(x, y.y) for x, y in zip(x_test_list, y_test_list)]

y_train_cat_list = [categorize_y(x, y.y) for x, y in zip(x_train_list, y_train_list)]
y_val_cat_list = [categorize_y(x, y.y) for x, y in zip(x_val_list, y_val_list)]
y_test_cat_list = [categorize_y(x, y.y) for x, y in zip(x_test_list, y_test_list)]

#%% predictability of features

from sklearn.tree import DecisionTreeClassifier

def count_gini_reduction(df, y):
    results = {}
    for col in df.columns:
        tree = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=42)
        tree.fit(df[[col]], y)
        tree_structure = tree.tree_
        
        if tree_structure.node_count > 1:
            gini_parent = tree_structure.impurity[0]
            n_samples_parent = tree_structure.n_node_samples[0]
            
            gini_left = tree_structure.impurity[1]
            n_samples_left = tree_structure.n_node_samples[1]

            gini_right = tree_structure.impurity[2]
            n_samples_right = tree_structure.n_node_samples[2]

            weighted_gini_children = (
                (n_samples_left / n_samples_parent) * gini_left +
                (n_samples_right / n_samples_parent) * gini_right
            )
            gini_reduction = gini_parent - weighted_gini_children
        else:
            gini_reduction = 0.0
        
        results[col] = gini_reduction
    
    results_df = pd.DataFrame(list(results.items()), columns=['feature', 'gini_reduction'])
    results_df = results_df.sort_values(by='gini_reduction', ascending=False).reset_index(drop=True)
    
    return results_df
        
reduction = count_gini_reduction(df_train, y_cat_train)
print(reduction)

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
        
def evaluate_model_on_inc(model, x, y):
    y_pred = model.predict(x)
    results = {}
    name_columns = [col for col in x.columns if col.startswith("name")]
    for col in name_columns:
        name = col.lstrip("name_")
        name_index = x[x[f"name_{name}"] == 1].index
        
        y_inc = y_to_increments(x, y.y)[name_index]
        pred_inc = y_to_increments(x, y_pred.reshape(-1,))[name_index]
        
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
        
        y_inc = categorize_y(x, y.y)[name_index]
        pred_inc = categorize_y(x, y_pred.reshape(-1,))[name_index]
        
        acc = 1 - mean_squared_error(y_inc, pred_inc)
        r2 = r2_score(y_inc, pred_inc)
        results[name] = {"acc": acc, "r2": r2}
    return results

#%% plot results for one name

import matplotlib.pyplot as plt

def plot_prediction(name, model, x, y):
    name_index = x[x[f"name_{name}"] == 1].index
    y_name = y.iloc[name_index]
    y_pred = model.predict(x)
    pred_name = y_pred[name_index]
    plt.figure(figsize=(10, 6))
    plt.scatter(name_index, y_name, alpha=0.7, label='original', color='blue', s=8)
    plt.scatter(name_index, pred_name, label='pred', alpha=0.7, color='orange', s=8)
    plt.title(f'{name} close price prediction', fontsize=16)
    plt.xlabel('index', fontsize=12)
    plt.ylabel('close price', fontsize=12)
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
        
def plot_inc(model, x, y):
    y_pred = model.predict(x)
    name_columns = [col for col in x.columns if col.startswith("name")]
    for col in name_columns:
        name = col.lstrip("name_")
            
        name_index = x[x[f"name_{name}"] == 1].index
        y_inc = y_to_increments(df_test, y_test.y)[name_index]
        pred_inc = y_to_increments(df_test, y_pred.reshape(-1,))[name_index]
        
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
        y_cat = categorize_y(df_test, y_test.y)[name_index]
        pred_cat = categorize_y(df_test, y_pred.reshape(-1,))[name_index]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(name_index, y_cat, alpha=0.7, label='original', color='blue')
        plt.scatter(name_index, pred_cat, label='pred', alpha=0.7, color='orange')
        plt.title(f'{name} close price prediction', fontsize=16)
        plt.xlabel('index', fontsize=12)
        plt.ylabel('close price', fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.show()
        
def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line (random model)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

#%% linear regression

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(df_train, y_train)

y_pred = model.predict(df_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")

### cat

y_cat = categorize_y(df_test, y_test.y)
pred_cat = categorize_y(df_test, y_pred.reshape(-1,))

mse = mean_squared_error(y_cat, pred_cat)
r2 = r2_score(y_cat, pred_cat)
print(f"Accuracy: {1 - mse}")
print(f"R2: {r2}")

plot_cat(model, df_test, y_test)
evaluate_model_on_cat(model, df_test, y_test)

### inc

y_inc = y_to_increments(df_test, y_test.y)
pred_inc = y_to_increments(df_test, y_pred.reshape(-1,))

mse = mean_squared_error(y_inc, pred_inc)
r2 = r2_score(y_cat, pred_cat)
print(f"MSE: {mse}")
print(f"R2: {r2}")

plot_inc(model, df_test, y_test)
evaluate_model_on_inc(model, df_test, y_test)


for name, coef in zip(df_train.columns, model.coef_[0]):
    if abs(coef) > 0.1:
        print(f"{name}: {coef:.4f}")

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

alphas = [0.001, 0.003, 0.005, 0.008, 0.01, 0.02]
l1_ratios = [0.2, 0.5, 0.8, 0.9]

best_params = {}
best_val_mse = float('inf')
best_model = None

for alpha in alphas:
    for l1_ratio in l1_ratios:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(df_train, y_inc_train)
        y_val_pred = model.predict(df_val)
        val_mse = mean_squared_error(y_inc_val, y_val_pred)
        print(f"Alpha: {alpha}, l1 Ratio: {l1_ratio}, val MSE: {val_mse}")
        eval_results = evaluate_model(model, x_val_list, y_val_list)
        print(eval_results)
        print()
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}
            best_model = model

print("Best parameters:", best_params)
print("Best val MSE:", best_val_mse)

#%% tuned ElasticNet

model = ElasticNet(alpha=0.003, l1_ratio=0.8, random_state=42)
model.fit(df_train, y_train)

y_pred = model.predict(df_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")

feature_importances = np.abs(model.coef_)
importance_df = pd.DataFrame({'Feature': df_train.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df[:20]

### cat

y_cat = categorize_y(df_test, y_test.y)
pred_cat = categorize_y(df_test, y_pred.reshape(-1,))

mse = mean_squared_error(y_cat, pred_cat)
r2 = r2_score(y_cat, pred_cat)
print(f"Accuracy: {1 - mse}")
print(f"R2: {r2}")

auc_score = roc_auc_score(y_cat, y_pred)
print(f'ROC AUC Score: {auc_score:.4f}')
plot_roc_curve(y_cat, y_pred)

plot_cat(model, df_test, y_test)
evaluate_model_on_cat(model, df_test, y_test)

### inc

y_inc = y_to_increments(df_test, y_test.y)
pred_inc = y_to_increments(df_test, y_pred.reshape(-1,))

mse = mean_squared_error(y_inc, pred_inc)
r2 = r2_score(y_cat, pred_cat)
print(f"MSE: {mse}")
print(f"R2: {r2}")


plot_inc(model, df_test, y_test)
evaluate_model_on_inc(model, df_test, y_test)

###

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

print(f"MSE: {mse}")
print(f"R2: {r2}")

eval_results = evaluate_model(model, x_test_list, y_test_list)
print(eval_results)
full_eval_results = evaluate_model_by_names(model, x_test_list, y_test_list)
print_eval_results(full_eval_results)

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)
plot_prediction_by_names(model, x_test_list, y_test_list)

### cat

y_cat = categorize_y(df_test, y_test.y)
pred_cat = categorize_y(df_test, y_pred.reshape(-1,))

mse = mean_squared_error(y_cat, pred_cat)
r2 = r2_score(y_cat, pred_cat)
print(f"Accuracy: {1 - mse}")
print(f"R2: {r2}")

auc_score = roc_auc_score(y_cat, y_pred)
print(f'ROC AUC Score: {auc_score:.4f}')
plot_roc_curve(y_cat, y_pred)

plot_cat(model, df_test, y_test)
evaluate_model_on_cat(model, df_test, y_test)

### inc

y_inc = y_to_increments(df_test, y_test.y)
pred_inc = y_to_increments(df_test, y_pred.reshape(-1,))

mse = mean_squared_error(y_inc, pred_inc)
r2 = r2_score(y_cat, pred_cat)
print(f"MSE: {mse}")
print(f"R2: {r2}")

plot_inc(model, df_test, y_test)
evaluate_model_on_inc(model, df_test, y_test)

#%% fine-tuning random forest

from sklearn.model_selection import GridSearchCV

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

print(f"Best params: {grid_search.best_params_}, val MSE: {val_mse}, R2: {val_r2}")

#%% tuned random forest

model = RandomForestRegressor(n_estimators=350, random_state=42, max_depth=None, n_jobs=-1)
model.fit(df_train, y_train)

y_pred = model.predict(df_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")

eval_results = evaluate_model(model, x_test_list, y_test_list)
print(eval_results)
full_eval_results = evaluate_model_by_names(model, x_test_list, y_test_list)
print_eval_results(full_eval_results)

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)
plot_prediction_by_names(model, x_test_list, y_test_list)

feature_importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': df_train.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df[:20]

joblib.dump(model, "models/random_forest_model.pkl", compress=3)
model = joblib.load("models/random_forest_model.pkl")

### cat

y_cat = categorize_y(df_test, y_test.y)
pred_cat = categorize_y(df_test, y_pred.reshape(-1,))

mse = mean_squared_error(y_cat, pred_cat)
r2 = r2_score(y_cat, pred_cat)
print(f"Accuracy: {1 - mse}")
print(f"R2: {r2}")

auc_score = roc_auc_score(y_cat, y_pred)
print(f'ROC AUC score: {auc_score}')
plot_roc_curve(y_cat, y_pred)

plot_cat(model, df_test, y_test)
evaluate_model_on_cat(model, df_test, y_test)

### inc

y_inc = y_to_increments(df_test, y_test.y)
pred_inc = y_to_increments(df_test, y_pred.reshape(-1,))

mse = mean_squared_error(y_inc, pred_inc)
r2 = r2_score(y_cat, pred_cat)
print(f"MSE: {mse}")
print(f"R2: {r2}")

plot_inc(model, df_test, y_test)
evaluate_model_on_inc(model, df_test, y_test)

#%% xgboost

import xgboost as xgb

model = xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=5, n_jobs=-1)

model.fit(df_train, y_train)

y_pred = model.predict(df_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")

eval_results = evaluate_model(model, x_test_list, y_test_list)
print(eval_results)
full_eval_results = evaluate_model_by_names(model, x_test_list, y_test_list)
print_eval_results(full_eval_results)

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)
plot_prediction_by_names(model, x_test_list, y_test_list)

### cat

y_cat = categorize_y(df_test, y_test.y)
pred_cat = categorize_y(df_test, y_pred.reshape(-1,))

mse = mean_squared_error(y_cat, pred_cat)
r2 = r2_score(y_cat, pred_cat)
print(f"Accuracy: {1 - mse}")
print(f"R2: {r2}")

auc_score = roc_auc_score(y_cat, y_pred)
print(f'ROC AUC Score: {auc_score:.4f}')
plot_roc_curve(y_cat, y_pred)

plot_cat(model, df_test, y_test)
evaluate_model_on_cat(model, df_test, y_test)

### inc

y_inc = y_to_increments(df_test, y_test.y)
pred_inc = y_to_increments(df_test, y_pred.reshape(-1,))

mse = mean_squared_error(y_inc, pred_inc)
r2 = r2_score(y_cat, pred_cat)
print(f"MSE: {mse}")
print(f"R2: {r2}")

plot_inc(model, df_test, y_test)
evaluate_model_on_inc(model, df_test, y_test)

#%% trim

xgb.plot_importance(model)
plt.show()

#%% xgboost tuning

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
                              f"alpha: {alpha}, gamma: {gamma}, custom score: {score}")
                        
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

#%% tuned xgboost

model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, alpha=0.1, gamma=1, n_jobs=-1)
model.fit(df_train, y_train)

y_pred = model.predict(df_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")

eval_results = evaluate_model(model, x_test_list, y_test_list)
print(eval_results)
full_eval_results = evaluate_model_by_names(model, x_test_list, y_test_list)
print_eval_results(full_eval_results)

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)
plot_prediction_by_names(model, x_test_list, y_test_list)

joblib.dump(model, "models/xgboost_model.pkl")
model = joblib.load("models/xgboost_model.pkl")

### cat

y_cat = categorize_y(df_test, y_test.y)
pred_cat = categorize_y(df_test, y_pred.reshape(-1,))

mse = mean_squared_error(y_cat, pred_cat)
r2 = r2_score(y_cat, pred_cat)
print(f"Accuracy: {1 - mse}")
print(f"R2: {r2}")

auc_score = roc_auc_score(y_cat, y_pred)
print(f'ROC AUC Score: {auc_score:.4f}')
plot_roc_curve(y_cat, y_pred)

plot_cat(model, df_test, y_test)
evaluate_model_on_cat(model, df_test, y_test)

### inc

y_inc = y_to_increments(df_test, y_test.y)
pred_inc = y_to_increments(df_test, y_pred.reshape(-1,))

mse = mean_squared_error(y_inc, pred_inc)
r2 = r2_score(y_cat, pred_cat)
print(f"MSE: {mse}")
print(f"R2: {r2}")

plot_inc(model, df_test, y_test)
evaluate_model_on_inc(model, df_test, y_test)

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

print(f"MSE: {mse}")
print(f"R2: {r2}")

eval_results = evaluate_model(model, x_test_list, y_test_list)
print(eval_results)
full_eval_results = evaluate_model_by_names(model, x_test_list, y_test_list)
print_eval_results(full_eval_results)

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)
plot_prediction_by_names(model, x_test_list, y_test_list)

### cat

y_cat = categorize_y(df_test, y_test.y)
pred_cat = categorize_y(df_test, y_pred.reshape(-1,))

mse = mean_squared_error(y_cat, pred_cat)
r2 = r2_score(y_cat, pred_cat)
print(f"Accuracy: {1 - mse}")
print(f"R2: {r2}")

auc_score = roc_auc_score(y_cat, y_pred)
print(f'ROC AUC Score: {auc_score:.4f}')
plot_roc_curve(y_cat, y_pred)

plot_cat(model, df_test, y_test)
evaluate_model_on_cat(model, df_test, y_test)

### inc

y_inc = y_to_increments(df_test, y_test.y)
pred_inc = y_to_increments(df_test, y_pred.reshape(-1,))

mse = mean_squared_error(y_inc, pred_inc)
r2 = r2_score(y_cat, pred_cat)
print(f"MSE: {mse}")
print(f"R2: {r2}")

plot_inc(model, df_test, y_test)
evaluate_model_on_inc(model, df_test, y_test)

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

#%% tuned bagging

base_model = DecisionTreeRegressor(random_state=42)

model = BaggingRegressor(estimator=base_model, 
                        n_estimators=200, max_samples=0.5, max_features=1.0,
                        n_jobs=-1)
model.fit(df_train, y_train)

y_pred = model.predict(df_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")

eval_results = evaluate_model(model, x_test_list, y_test_list)
print(eval_results)
full_eval_results = evaluate_model_by_names(model, x_test_list, y_test_list)
print_eval_results(full_eval_results)

name = "AAPL"
plot_all_prediction(name, model, x_test_list, y_test_list)
plot_prediction_by_names(model, x_test_list, y_test_list)

joblib.dump(model, "models/bagging_model.pkl")
model = joblib.load("models/bagging_model.pkl")

### cat

y_cat = categorize_y(df_test, y_test.y)
pred_cat = categorize_y(df_test, y_pred.reshape(-1,))

mse = mean_squared_error(y_cat, pred_cat)
r2 = r2_score(y_cat, pred_cat)
print(f"Accuracy: {1 - mse}")
print(f"R2: {r2}")

auc_score = roc_auc_score(y_cat, y_pred)
print(f'ROC AUC Score: {auc_score:.4f}')
plot_roc_curve(y_cat, y_pred)

plot_cat(model, df_test, y_test)
evaluate_model_on_cat(model, df_test, y_test)

### inc

y_inc = y_to_increments(df_test, y_test.y)
pred_inc = y_to_increments(df_test, y_pred.reshape(-1,))

mse = mean_squared_error(y_inc, pred_inc)
r2 = r2_score(y_cat, pred_cat)
print(f"MSE: {mse}")
print(f"R2: {r2}")

plot_inc(model, df_test, y_test)
evaluate_model_on_inc(model, df_test, y_test)

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
print(f"MSE: {mse}")
print(f"R2: {r2}")

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

#%%

from tensorflow.keras.models import load_model

model = load_model("models/nn.keras")

y_pred = model.predict(df_test)

plot_prediction_by_names(model, x_test_list, y_test_list)

### cat

y_cat = categorize_y(df_test, y_test.y)
pred_cat = categorize_y(df_test, y_pred.reshape(-1,))

mse = mean_squared_error(y_cat, pred_cat)
r2 = r2_score(y_cat, pred_cat)
print(f"Accuracy: {1 - mse}")
print(f"R2: {r2}")

plot_cat(model, df_test, y_test)
evaluate_model_on_cat(model, df_test, y_test)

### inc

y_inc = y_to_increments(df_test, y_test.y)
pred_inc = y_to_increments(df_test, y_pred.reshape(-1,))

mse = mean_squared_error(y_inc, pred_inc)
r2 = r2_score(y_cat, pred_cat)
print(f"MSE: {mse}") # MSE: 12.691536323513928
print(f"R2: {r2}")

plot_inc(model, df_test, y_test)
evaluate_model_on_inc(model, df_test, y_test)
