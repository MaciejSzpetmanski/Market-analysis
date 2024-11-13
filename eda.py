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

#%% formations test

import importlib.util
import sys

# Specify the path to the file
file_path = os.path.join(path, 'Wskaźniki itp/Schemat stała długość/advance block.py')

# Load the module
spec = importlib.util.spec_from_file_location("advance_block", file_path)
my_module = importlib.util.module_from_spec(spec)
sys.modules["my_module"] = my_module
spec.loader.exec_module(my_module)

functions = {name: func for name, func in vars(my_module).items() if callable(func)}
# Now you can access my_function
my_function = my_module.wykryj_advance_block

res = df.groupby("name", group_keys=False).apply(lambda x: my_function(x))
res = np.concatenate([my_function(group) for _, group in df.groupby("name")])

df["advance_block"] = res

my_function(df)

#%% represent all prices in dollars

# counting GBPUSD
eurusd_df = df[df.name == "EURUSD=X"][["date"] + float_columns]
eurgbp_df = df[df.name == "EURGBP=X"][["date", "volume"] + float_columns]

merged_df = pd.merge(eurusd_df, eurgbp_df, on="date", how="inner")

for col in float_columns:
    merged_df[col] = merged_df[col+"_x"] / merged_df[col+"_y"]

merged_df["name"] = "GBPUSD=X"
merged_df = merged_df[df.columns]
df = pd.concat([df, merged_df], ignore_index=True)

# remove EURGBP
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

#%% identify useless columns (it may differ based on basic frequency)

# TODO identify useless columns - fractals

for col in df.columns:
    print(f"{col}: {len(df[col].unique())}")
    
df.date_leap_year
df.date_year_end.value_counts()
df.date_weekend.value_counts() # there is some data from weekend
df[df.date_weekend == 1]["name"] # crypto

# cols_to_remove = ["Freq", "Date_hour", "Date_minute", "Date_second"] # Freq, if only one frequency is used
# possibly remove these columns earlier
# df = df.drop(columns=cols_to_remove)

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

#%% creating new columns

# TODO fractals

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
    .where(lambda x: (abs(x) > threshold1) & (abs(x) < threshold2))\
    .stack().reset_index()\
    .rename(columns={'level_0': 'Feature1', 'level_1': 'Feature2', 0: 'Correlation'})
res.loc[res.Feature1 != res.Feature2]

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

#%% transformations


#%% standarization

df_train.info()

from sklearn.preprocessing import StandardScaler

columns_to_standarize = ["close", "high", "low", "open", "volume"]
# cast Volume to float before scaling
# possibly apply ln(x+1) or MinMaxScaler

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

#%% example inversion

columns_to_standarize = ["close", "high", "low", "open", "volume"]
AAPL_scaled = data[data['name'] == 'AAPL'][columns_to_standarize]
AAPL_original = scalers['AAPL'].inverse_transform(AAPL_scaled)
AAPL_original.shape

#%% scaling remaining columns

# drop Timestamp (but it enables more accurate prediction - to hours/minutes/...)
# MinMaxScaling
# sine and cosine transformation - cyclic features
# or leave as it is

from sklearn.preprocessing import MinMaxScaler

# optionally exclude Names
columns_to_scale = [col for col in df_train.columns if col not in columns_to_standarize + ["name"]]

# isconsistent types
scaler = MinMaxScaler()
df_train[columns_to_scale] = scaler.fit_transform(df_train[columns_to_scale])
df_val[columns_to_scale] = scaler.transform(df_val[columns_to_scale])
df_test[columns_to_scale] = scaler.transform(df_test[columns_to_scale])

df_train.min()
df_train.max()

df_train.columns
df_train.info()

#%% preparing data as time series

def create_time_series(data, columns, group_by_column, target_column, sort_columns, n):
    # setting chronological order
    data = data.sort_values(by=[group_by_column] + sort_columns)
    shifted_columns = {}
    # index _(N-1) is the latest data
    for i in range(1, n):
        for col in columns:
            shifted_columns[col + "_" + str(i)] = data.groupby(group_by_column, group_keys=False)[col].shift(-i)
    # target column
    shifted_columns["y"] = data.groupby(group_by_column, group_keys=False)[target_column].shift(-N)
    data = pd.concat([data] + [pd.Series(value, name=key) for key, value in shifted_columns.items()], axis=1)
    # removing rows with missing values
    data = data.groupby(group_by_column, group_keys=False).head(-N)
    return data

# omit Names
columns = [col for col in df_train.columns if not col.startswith("name")] # cols to shift
group_by_column = "name"
target_column = "close"
sort_columns = ["date_year", "date_month", "date_day_of_month"]
N = 20

# strange indexes
# OPTIONAL reset index
df_train = create_time_series(df_train, columns, group_by_column, target_column, sort_columns, N)
df_val = create_time_series(df_val, columns, group_by_column, target_column, sort_columns, N)
df_test = create_time_series(df_test, columns, group_by_column, target_column, sort_columns, N)

#%% identify target column

y_train = df_train["y"]
y_val = df_val["y"]
y_test = df_test["y"]

with warnings.catch_warnings():
    warnings.simplefilter("ignore", Warning)
    for data in [df_train, df_val, df_test]:
        data.drop(columns=["y"], inplace=True)

#%%

# TODO once again check boxplots, histograms, correlation, pair-plots

#%%

# outliers
# additional columns
# pca (high correlation) ?
