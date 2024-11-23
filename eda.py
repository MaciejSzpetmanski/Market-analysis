#%% working directory

import os

path = "D:\Studia\semestr7\inźynierka\Market-analysis"
# path = "C:\Studia\Market-analysis"
os.chdir(path)

#%% loading data

import numpy as np
import pandas as pd

def load_data(directory_name, suffix=""):
    df = None
    for file_name in os.listdir(directory_name):
        if not file_name.endswith(suffix):
            continue
        file_path = os.path.join(directory_name, file_name)
        if os.path.isfile(file_path) and file_path.endswith(".csv"):
            df_input = pd.read_csv(file_path)
            # renaming the first column to Date
            df_input.rename(columns={'Price': 'date'}, inplace=True)
            # dropping first 2 empty rows 
            df_input.drop([0, 1], inplace=True)
            
            # parsing filename
            file_name += "_"
            file_info = file_name.replace(".csv", "").split("_")
            # adding source information
            df_input["name"] = file_info[0]
            # df_input["Freq"] = file_info[2]
            
            df = pd.concat([df, df_input], ignore_index=True)
    return df


df = load_data("data", "_data_1d.csv")
# lowercase columns
df.columns = df.columns.str.lower()
df = df.rename(columns={"adj close": "adjusted_close"})

df.shape
df.head()
df.columns

#%% converting column types

df.describe()
df.info()

df['volume']
df['volume'] = df['volume'].astype('int64')

float_columns = ['close', 'high', 'low', 'open', 'adjusted_close']
for col in float_columns:
    df[col] = df[col].astype('float')

df.info()

#%% removing duplicates

df = df.drop_duplicates()
df = df.reset_index(drop=True)

#%% represent all prices in dollars

# GBP already exists in data -> "EURGBP=X" is useless
df[df.name == "GBPUSD=X"]

# counting GBPUSD
# [1759 rows x 8 columns]
# [52828 rows x 8 columns]
# eurusd_df = df[df.name == "EURUSD=X"][["date"] + float_columns]
# eurgbp_df = df[df.name == "EURGBP=X"][["date", "volume"] + float_columns]

# merged_df = pd.merge(eurusd_df, eurgbp_df, on="date", how="inner")

# for col in float_columns:
#     merged_df[col] = merged_df[col+"_x"] / merged_df[col+"_y"]

# merged_df["name"] = "GBPUSD=X"
# merged_df = merged_df[df.columns]
# df = pd.concat([df, merged_df], ignore_index=True)

# # remove EURGBP
df = df[~(df.name == "EURGBP=X")]
df = df.reset_index(drop=True)

#%% time handling

from feature_engine.datetime import DatetimeFeatures

df['date'] = pd.to_datetime(df['date'])

# saving date
date = df['date']

dt_features = DatetimeFeatures(
    variables=['date'],
    features_to_extract=[
        'month', 'quarter', 'semester', 'year', 'week',
        'day_of_week', 'day_of_month', 'day_of_year',
        'weekend', 'month_start', 'month_end',
        'quarter_start', 'quarter_end', 'year_start',
        'year_end', 'leap_year', 'days_in_month'
    ]
)

df = dt_features.fit_transform(df)
df['date'] = date

df.info()

#%% checking time

for name, group in df.groupby("name"):
    print(name)
    # print(f"{group.date.value_counts()}")
    print(f"unique dates: {group.date.nunique()}")
    print(f"all dates: {len(group)}")
    print(f"all dates: {group.date.nunique() / len(group)}")
    
#%% identify useless columns

for col in df.columns:
    print(f"{col}: {len(df[col].unique())}")
    
df.date_leap_year
df.date_year_end.value_counts()
df.date_weekend.value_counts() # there is some data from weekend
df[df.date_weekend == 1]["name"] # crypto


#%% name encoding (one-hot)

name_column = df["name"]

df = pd.get_dummies(df, columns=['name'], prefix='name',  dtype=int)
df.shape
df.columns

df["name"] = name_column

#%% split

from sklearn.model_selection import train_test_split

last_date = df['date'].max()
split_date = last_date - pd.DateOffset(years=1)

df_train = df[df['date'] <= split_date]
df_train.shape

# OPTIONAL change sizes
# stratify by date (day-month-year)
df_val, df_test = train_test_split(df[df['date'] > split_date], test_size=0.6, random_state=42, stratify=df[df['date'] > split_date]['date'])

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", Warning)
    # removing Date column
    for data in [df_train, df_val, df_test]:
        data.drop(columns=['date'], inplace=True)

print(f'train: {df_train.shape}')
print(f'val: {df_val.shape}')
print(f'test: {df_test.shape}')

#%% reset index

# TODO reset and remove reset from functions below

#%% mutable length formations

import importlib.util
import threading

def add_fractal_long_schemas(df, path, group_by_column, prefix, width):
    def apply_function(df, function, group_by_column, width):
        n = len(df)
        res = np.full(n, False)
        for i in range(width-1, n):
            data = df_train[i+1-width:i+1]
            res[i] = function(data)
        return res

    new_columns = {}
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path) and file_path.endswith(".py"):
            spec = importlib.util.spec_from_file_location("file_name", file_path)
            my_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(my_module)
            
            functions = {name: func for name, func in vars(my_module).items() if callable(func)}
            # selecting the last function from script
            function_name, function = list(functions.items())[-1]
            
            column_name = prefix + function_name.lstrip("wykryj_")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
                new_columns[column_name] = np.concatenate([apply_function(group, function, group_by_column, width) for _, group in df.groupby(group_by_column)]).astype(int)
    res_data = pd.concat([df.reset_index(drop=True)] + [pd.Series(value, name=key) for key, value in new_columns.items()], axis=1)
    return res_data

functions_path = os.path.join(path, 'Wskaźniki itp/Schemat zmienna długość')
functions_path = 'Wskaźniki itp/Schemat zmienna długość'
group_by_column = 'name'
width = 20

import time

start_time = time.time()

df_train = add_fractal_long_schemas(df_train, functions_path, group_by_column, 'long_formation_', width)
df_val = add_fractal_long_schemas(df_val, functions_path, group_by_column, 'long_formation_', width)
df_test = add_fractal_long_schemas(df_test, functions_path, group_by_column, 'long_formation_', width)

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")

for col in df_train.columns:
    if col.startswith("long_formation_"):
        print(f"{col}: {len(df_train[col][df_train[col] == True])}")

# for width=20 death_cross, exhaustion_gap, golden_cross, unaway_gap were not detected
# remove these features or check other widths

#%% cycle

import sys

sys.path.append(os.path.abspath("Wskaźniki itp"))
from cykl import wykryj_typ_cyklu

def add_cycle_columns(df, group_by_column, width):
    def apply_cycle_function(df, group_by_column, width):
        n = len(df)
        res = np.full(n, "", dtype=object)
        for i in range(width-1, n):
            data = df_train[i+1-width:i+1]
            # get cycle name only
            res[i] = wykryj_typ_cyklu(data)[1]
        return res
    
    cycle_names = {
       "Cykl Neutralny": "neutral",
       "Cykl Zbieżny": "convergent",
       "Cykl Rozbieżny": "divergent",
       "": "not_detected"
    }
    
    # TODO warnings
    df["cycle"] = np.concatenate([apply_cycle_function(group, group_by_column, width) for _, group in df.groupby(group_by_column)])
    df["cycle"] = df["cycle"].apply(lambda x: cycle_names[x])

    # one-hot encoding
    df = pd.get_dummies(df, columns=["cycle"], prefix="cycle",  dtype=int)
    return df


group_by_column = "name"
width = 20

df_train = add_cycle_columns(df_train, group_by_column, width)
df_val = add_cycle_columns(df_val, group_by_column, width)
df_test = add_cycle_columns(df_test, group_by_column, width)

# all values are present in df_train

#%% ema

from EMA import ema

def add_ema_columns(df, group_by_column, period):
    df[f"ema_{period}"] = np.concatenate([ema(group.close, period) for _, group in df.groupby(group_by_column)])
    return df

group_by_column = "name"
ema_periods = [10, 20, 50, 100, 200]

for period in ema_periods:
    df_train = add_ema_columns(df_train, group_by_column, period)
    df_val = add_ema_columns(df_val, group_by_column, period)
    df_test = add_ema_columns(df_test, group_by_column, period)
    
#%% constant length formations

# TODO reset index outside functions

def add_fractal_short_schemas(df, path, group_by_column, prefix):
    new_columns = {}
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path) and file_path.endswith(".py"):
            spec = importlib.util.spec_from_file_location("file_name", file_path)
            my_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(my_module)
            
            functions = {name: func for name, func in vars(my_module).items() if callable(func)}
            # selecting the last function from script
            function_name, function = list(functions.items())[-1]
            
            column_name = prefix + function_name.lstrip("wykryj_")
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)
                new_columns[column_name] = np.concatenate([function(group) for _, group in df.groupby(group_by_column)]).astype(int)

    res_data = pd.concat([df.reset_index(drop=True)] + [pd.Series(value, name=key) for key, value in new_columns.items()], axis=1)
    return res_data

functions_path = os.path.join(path, 'Wskaźniki itp/Schemat stała długość')
group_by_column = 'name'

df_train = add_fractal_short_schemas(df_train, functions_path, group_by_column, 'short_formation_')
df_val = add_fractal_short_schemas(df_val, functions_path, group_by_column, 'short_formation_')
df_test = add_fractal_short_schemas(df_test, functions_path, group_by_column, 'short_formation_')

for col in df_train.columns:
    if col.startswith("short_formation_"):
        print(f"{col}: {len(df_train[col][df_train[col] == True])}")
# no detections of concealing_baby_swallow

#%% removing empty columns

columns_to_remove = [col for col in df_train.columns if df_train[col].nunique() == 1]

df_train = df_train.drop(columns=columns_to_remove)
df_val = df_val.drop(columns=columns_to_remove)
df_test = df_test.drop(columns=columns_to_remove)

#%% adjusted_close and close comparison

df_train.groupby("name").apply(lambda x: (x.adjusted_close == x.close).sum() / len(x))
# in most groups adjusted_close is equal to close, in actions there is significant difference

#%% correlation - pearson

import matplotlib.pyplot as plt
import seaborn as sns

corr_matrix = df_train.drop(columns=['name']).corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Covariance Matrix Heatmap')
plt.show()

# detect highly correlated columns
threshold1 = 0.8
threshold2 = 1

res = corr_matrix\
    .where(lambda x: np.fromfunction(lambda i, j: i < j, x.shape))\
    .where(lambda x: (abs(x) > threshold1) & (abs(x) < threshold2))\
    .stack().reset_index()\
    .rename(columns={'level_0': 'Feature1', 'level_1': 'Feature2', 0: 'Correlation'})
    
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(res.sort_values("Correlation", ascending=False))
pd.reset_option('display.expand_frame_repr')
pd.reset_option('display.max_columns')
pd.reset_option('display.width')

# date columns have high correlation with each other

#%% check correlation with Close feature

# TODO check again after adding fractal info
corr_matrix["close"][abs(corr_matrix["close"]) > 0.6]

#%% correlation - spearman

corr_matrix = df_train.drop(columns=['name']).corr(method='spearman')

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Covariance Matrix Heatmap')
plt.show()

# detect highly correlated columns
threshold1 = 0.6
threshold2 = 1

res = corr_matrix\
    .where(lambda x: (abs(x) > threshold1) & (abs(x) < threshold2))\
    .stack().reset_index()\
    .rename(columns={'level_0': 'Feature1', 'level_1': 'Feature2', 0: 'Correlation'})
res.loc[res.Feature1 != res.Feature2]

# date columns have high correlation with each other

corr_matrix["close"][abs(corr_matrix["close"]) > 0.6]

#%% histograms

def draw_histograms(data, columns, group_by_column, output_path):
    grouped = data[columns].groupby(group_by_column)
    # create histograms for each group
    for name, group in grouped:
        group.hist(bins=50, figsize=(10, 8), label=name)
        plt.suptitle(name, fontsize=16)
        plt.savefig(f"{output_path}/{name}_histogram.png")
        plt.close() 
        
hist_columns = ["close", "high", "low", "open", "volume", "name"]
output_path = "plots/histograms"

draw_histograms(df_train, hist_columns, "name", output_path)

#%% boxplots

def draw_boxplots(data, columns, group_by_column, output_path):
    grouped = data[columns].groupby(group_by_column)
    # create boxplots for each group
    for name, group in grouped:
        fig, axes = plt.subplots(1, len(columns)-1, figsize=(10, 8))
        # plot each feature in a separate subplot
        for i, col in enumerate(columns):
            if col == group_by_column:
                continue
            group.boxplot(column=col, ax=axes[i])
            axes[i].set_title(col, fontsize=12)
            # axes[i].set_ylabel(col)
        plt.tight_layout()
        plt.suptitle(name, fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.savefig(f"{output_path}/{name}_boxplot.png")
        plt.close()
        
box_columns = ["close", "high", "low", "open", "volume", "name"]
output_path = "plots/boxplots"

draw_boxplots(df_train, box_columns, "name", output_path)

#%% standarization

df_train.info()

from sklearn.preprocessing import StandardScaler

columns_to_standarize = ["adjusted_close", "close", "high", "low", "open", "volume"]

scalers = {}

# train
for name, group in df_train.groupby('name'):
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(group[columns_to_standarize])
    df_train.loc[df_train['name'] == name, columns_to_standarize] = scaled_values
    scalers[name] = scaler

# val
for name, group in df_val.groupby('name'):
    scaler = scalers[name]
    scaled_values = scaler.transform(group[columns_to_standarize])
    df_val.loc[df_val['name'] == name, columns_to_standarize] = scaled_values

# test
for name, group in df_test.groupby('name'):
    scaler = scalers[name]
    scaled_values = scaler.transform(group[columns_to_standarize])
    df_test.loc[df_test['name'] == name, columns_to_standarize] = scaled_values

print(df_train)

#%% saving scalers

import pickle

scalers_path = 'scalers/scalers.pkl'

with open(scalers_path, 'wb') as file:
    pickle.dump(scalers, file)
    
#%% example inversion

with open(scalers_path, 'rb') as file:
    scalers = pickle.load(file)

columns_to_standarize = ["adjusted_close", "close", "high", "low", "open", "volume"]
AAPL_scaled = data[data['name'] == 'AAPL'][columns_to_standarize]
AAPL_original = scalers['AAPL'].inverse_transform(AAPL_scaled)
AAPL_original.shape
# close values
AAPL_original[:, 1]

#%% scaling remaining columns

# TODO scale or not (which columns?)
# MinMaxScaling
# sine and cosine transformation - cyclic features
# or leave as it is

# from sklearn.preprocessing import MinMaxScaler

# # optionally exclude Names
# columns_to_scale = [col for col in df_train.columns if col not in columns_to_standarize + ["name"]]

# # isconsistent types
# scaler = MinMaxScaler()
# df_train[columns_to_scale] = scaler.fit_transform(df_train[columns_to_scale])
# df_val[columns_to_scale] = scaler.transform(df_val[columns_to_scale])
# df_test[columns_to_scale] = scaler.transform(df_test[columns_to_scale])

# df_train.min()
# df_train.max()

# df_train.columns
# df_train.info()

#%% preparing data as time series

# TODO test after adding all fractal columns

def create_time_series(data, columns, group_by_column, target_column, sort_columns, n, k=1):
    """
    n - time series length
    k - target horizon
    """
    # setting chronological order
    data = data.sort_values(by=[group_by_column] + sort_columns)
    shifted_columns = {}
    # index _(n-1) is the latest data
    for i in range(1, n):
        for col in columns:
            shifted_columns[col + "_" + str(i)] = data.groupby(group_by_column, group_keys=False)[col].shift(-i)
    # target date
    date_columns = [col for col in columns if col.startswith("date")]
    for col in date_columns:
        shifted_columns[col + "_y"] = data.groupby(group_by_column, group_keys=False)[col].shift(-(n-1+k))
    # target column
    shifted_columns["y"] = data.groupby(group_by_column, group_keys=False)[target_column].shift(-(n-1+k))
    # collect columns - include all original columns
    res_data = pd.concat([data] + [pd.Series(value, name=key) for key, value in shifted_columns.items()], axis=1)
    # removing rows with missing values
    res_data = res_data.groupby(group_by_column, group_keys=False).head(-(n-1+k))
    return res_data

def create_wide_horizon_time_series(data, columns, group_by_column, target_column, sort_columns, n, max_target_horizon):
    df_full = None
    for k in range(1, max_target_horizon + 1):
        df = create_time_series(data, columns, group_by_column, target_column, sort_columns, n, k)
        df_full = pd.concat([df_full, df])
    return df_full

# omit Names
columns = [col for col in df_train.columns if not col.startswith("name")] # cols to shift
group_by_column = "name"
target_column = "close"
sort_columns = ["date_year", "date_month", "date_day_of_month"]
n = 20
max_target_horizon = 5

train_sets = {}
val_sets = {}
test_sets = {}

for k in range(1, max_target_horizon + 1):
    train_sets[k] = create_time_series(df_train, columns, group_by_column, target_column, sort_columns, n, k)
    val_sets[k] = create_time_series(df_val, columns, group_by_column, target_column, sort_columns, n, k)
    test_sets[k] = create_time_series(df_test, columns, group_by_column, target_column, sort_columns, n, k)

# df_train = create_wide_horizon_time_series(df_train, columns, group_by_column, target_column, sort_columns, n, max_target_horizon)
# df_val = create_wide_horizon_time_series(df_val, columns, group_by_column, target_column, sort_columns, n, max_target_horizon)
# df_test = create_wide_horizon_time_series(df_test, columns, group_by_column, target_column, sort_columns, n, max_target_horizon)

# reset index
for k in range(1, max_target_horizon + 1):
    train_sets[k] = train_sets[k].reset_index(drop=True)
    val_sets[k] = val_sets[k].reset_index(drop=True)
    test_sets[k] = test_sets[k].reset_index(drop=True)
    
# df_train = df_train.reset_index(drop=True)
# df_val = df_val.reset_index(drop=True)
# df_test = df_test.reset_index(drop=True)

#%% identify target column

y_train_sets = {}
y_val_sets = {}
y_test_sets = {}

for k in range(1, max_target_horizon + 1):
    y_train_sets[k] = train_sets[k]["y"]
    y_val_sets[k] = val_sets[k]["y"]
    y_test_sets[k] = test_sets[k]["y"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Warning)
        train_sets[k] = train_sets[k].drop(columns=["y"])
        val_sets[k] = val_sets[k].drop(columns=["y"])
        test_sets[k] = test_sets[k].drop(columns=["y"])

# y_train = df_train["y"]
# y_val = df_val["y"]
# y_test = df_test["y"]

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", Warning)
#     for data in [df_train, df_val, df_test]:
#         data.drop(columns=["y"], inplace=True)

#%% drop name

for k in range(1, max_target_horizon + 1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Warning)
        train_sets[k] = train_sets[k].drop(columns=["name"])
        val_sets[k] = val_sets[k].drop(columns=["name"])
        test_sets[k] = test_sets[k].drop(columns=["name"])

# df_train = df_train.drop(columns=["name"])
# df_val = df_val.drop(columns=["name"])
# df_test = df_test.drop(columns=["name"])

#%% save data

def save_data(X_sets, y_sets, directory, x_name, y_name):
    n = len(X_sets)
    for k in range(1, n + 1):
        X_path = os.path.join(directory, f"{x_name}_{k}.csv")
        y_path = os.path.join(directory, f"{y_name}_{k}.csv")
        X_sets[k].to_csv(X_path, index=False)
        y_sets[k].to_csv(y_path, index=False)
        

directory = "datasets"

save_data(train_sets, y_train_sets, directory, "df_train", "y_train")
save_data(val_sets, y_val_sets, directory, "df_val", "y_val")
save_data(test_sets, y_test_sets, directory, "df_test", "y_test")

# df_train.to_csv("datasets/df_train.csv", index=False)
# y_train.to_csv("datasets/y_train.csv", index=False)

# df_val.to_csv("datasets/df_val.csv", index=False)
# y_val.to_csv("datasets/y_val.csv", index=False)

# df_test.to_csv("datasets/df_test.csv", index=False)
# y_test.to_csv("datasets/y_test.csv", index=False)

#%%

# OPTIONAL once again check boxplots, histograms, correlation, pair-plots

#%%

# TODO add description of functions
# pca (high correlation) ?
# model
