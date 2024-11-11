#%% working directory

import os

# path = "D:\Studia\semestr7\inźynierka\Market-analysis"
path = "C:\Studia\Market-analysis"
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
            df_input.rename(columns={'Price': 'Date'}, inplace=True)
            # dropping first 2 empty rows 
            df_input.drop([0, 1], inplace=True)
            
            # parsing filename
            file_name += "_"
            file_info = file_name.replace(".csv", "").split("_")
            # adding source information
            df_input["Name"] = file_info[0]
            df_input["Freq"] = file_info[2]
            
            df = pd.concat([df, df_input], ignore_index=True)
    return df


# df = load_data("data")
df = load_data("data", "_data_1d.csv") # only daily frequency ???

df.shape
df.head()
df.columns

#%% dropping Adj Close

df = df.drop(columns=["Adj Close"])

#%% converting column types

df.describe()
df.info()

df['Volume']
df['Volume'] = df['Volume'].astype('int64')

float_columns = ['Close', 'High', 'Low', 'Open'] #'Adj Close'
for col in float_columns:
    df[col] = df[col].astype('float')

df.info()

#%% removing duplicates

subset = list(filter(lambda x: x != 'Freq', df.columns))
df.drop_duplicates(subset=subset, inplace=True)
df = df.reset_index(drop=True)

#%% represent all prices in dollars

# OPTIONAL change labels to one pattern: EURUSD, USDJPY ???

# problem with EURGBP
df.loc[df.Name == "EURGBP=X", "Open"]

# counting GBPUSD
eurusd_df = df[df.Name == "EURUSD=X"][["Date"] + float_columns]
eurgbp_df = df[df.Name == "EURGBP=X"][["Date", "Volume", "Freq"] + float_columns]

merged_df = pd.merge(eurusd_df, eurgbp_df, on="Date", how="inner")

for col in float_columns:
    merged_df[col] = merged_df[col+"_x"] / merged_df[col+"_y"]

merged_df["Name"] = "GBPUSD=X"

merged_df = merged_df[df.columns]

df = pd.concat([df, merged_df], ignore_index=True)

# remove EURGBP
df = df[~(df.Name == "EURGBP=X")]
df = df.reset_index(drop=True)

#%% time handling

from feature_engine.datetime import DatetimeFeatures

df['Date'] = pd.to_datetime(df['Date'])
df['Timestamp'] = df['Date'].astype('int64') // 10**9

# saving date
date = df['Date']

dt_features = DatetimeFeatures(
    variables=['Date'],
    features_to_extract=[
        'month', 'quarter', 'semester', 'year', 'week', 
        'day_of_week', 'day_of_month', 'day_of_year', 
        'weekend', 'month_start', 'month_end', 
        'quarter_start', 'quarter_end', 'year_start', 
        'year_end', 'leap_year', 'days_in_month', 
        'hour', 'minute', 'second'
    ]
)

# "hour", "minute", "second" are not used

df = dt_features.fit_transform(df)
df['Date'] = date

df.info()

#%% identify useless columns (it may differ based on basic frequency)

# TODO identify useless columns - fractals

for col in df.columns:
    print(f"{col}: {len(df[col].unique())}")
    
df.Date_second
df.Date_minute
df.Date_hour
df.Date_leap_year
df.Date_year_end.value_counts()
df.Date_weekend.value_counts() # there is some data from weekend
df[df.Date_weekend == 1]["Name"] # crypto

cols_to_remove = ["Freq", "Date_hour", "Date_minute", "Date_second"] # Freq, if only one frequency is used
# possibly remove these columns earlier
df = df.drop(columns=cols_to_remove)

#%% name encoding (one-hot)

name_column = df["Name"]

df = pd.get_dummies(df, columns=['Name'], prefix='Name',  dtype=int)
df.shape
df.columns

df["Name"] = name_column

#%% split

from sklearn.model_selection import train_test_split

last_date = df['Date'].max()
split_date = last_date - pd.DateOffset(years=1)

df_train = df[df['Date'] <= split_date]
df_train.shape

# stratify by date (day-month-year)
df_val, df_test = train_test_split(df[df['Date'] > split_date], test_size=0.6, random_state=42, stratify=df[df['Date'] > split_date]['Date'])

# later
# y_train = df_train['Close']
# y_val = df_val['Close']
# y_test = df_test['Close']

import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", Warning)
    # removing Date column
    for data in [df_train, df_val, df_test]:
        data.drop(columns=['Date'], inplace=True)

print(f'train: {df_train.shape}')
print(f'val: {df_val.shape}')
print(f'test: {df_test.shape}')

#%% creating new columns

# TODO fractals

#%% correlation - pearson

import matplotlib.pyplot as plt
import seaborn as sns

# corr_matrix = df_train.drop(columns=['Freq']).corr()
corr_matrix = df_train.drop(columns=['Name']).corr()

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
corr_matrix["Close"][abs(corr_matrix["Close"]) > 0.6]

#%% correlation - spearman

corr_matrix = df_train.drop(columns=['Name']).corr(method='spearman')

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

corr_matrix["Close"][abs(corr_matrix["Close"]) > 0.6]

#%% histograms

def draw_histograms(data, columns, group_by_column, output_path):
    grouped = data[columns].groupby(group_by_column)
    # create histograms for each group
    for name, group in grouped:
        group.hist(bins=50, figsize=(10, 8), label=name)
        plt.suptitle(name, fontsize=16)
        plt.savefig(f"{output_path}/{name}_histogram.png")
        plt.close() 
        
hist_columns = ["Close", "High", "Low", "Open", "Volume", "Timestamp", "Name"]
output_path = "plots/histograms"

draw_histograms(df_train, hist_columns, "Name", output_path)
# Volume between currencies is always 0!

# OPTIONAL group by index group

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
        
box_columns = ["Close", "High", "Low", "Open", "Volume", "Timestamp", "Name"]
output_path = "plots/boxplots"

draw_boxplots(df_train, box_columns, "Name", output_path)

#%% outliers

# OPTIONAL rather not
# apply separately for every index

def get_quantiles(df, cols_to_modify):
    quantiles = []
    for col in cols_to_modify:
        upper_lim = df[col].quantile(.99)
        lower_lim = df[col].quantile(.01)
        quantiles.append((lower_lim, upper_lim))
    return quantiles

# zamiana wartosci odstających na skrajne, po 1% wartosci z góry i z dołu
def modify_outliers(df, cols_to_modify, quantiles):
    for i in range(len(cols_to_modify)):
        upper_lim = quantiles[i][1]
        lower_lim = quantiles[i][0]
        col = cols_to_modify[i]
        df.loc[df[col] <= lower_lim, col] = lower_lim
        df.loc[df[col] > upper_lim, col] = upper_lim

#%% transformations


#%% standarization

df_train.info()
# Timestamp ?

from sklearn.preprocessing import StandardScaler

columns_to_standarize = ["Close", "High", "Low", "Open", "Volume"]
# cast Volume to float before scaling
# possibly apply ln(x+1) or MinMaxScaler

scalers = {}

# train
for name, group in df_train.groupby('Name'):
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(group[columns_to_standarize])
    df_train.loc[df_train['Name'] == name, columns_to_standarize] = scaled_values
    scalers[name] = scaler

# val
for name, group in df_val.groupby('Name'):
    scaler = scalers[name]
    scaled_values = scaler.transform(group[columns_to_standarize])
    df_val.loc[df_val['Name'] == name, columns_to_standarize] = scaled_values
    
# test
for name, group in df_test.groupby('Name'):
    scaler = scalers[name]
    scaled_values = scaler.transform(group[columns_to_standarize])
    df_test.loc[df_test['Name'] == name, columns_to_standarize] = scaled_values

print(df_train)

#%% example inversion

columns_to_standarize = ["Close", "High", "Low", "Open", "Volume"]
AAPL_scaled = data[data['Name'] == 'AAPL'][columns_to_standarize]
AAPL_original = scalers['AAPL'].inverse_transform(AAPL_scaled)
AAPL_original.shape

#%% drop columns

columns_to_drop = ["Timestamp"] # Name will be used to group data, Timestamp may be useful

with warnings.catch_warnings():
    warnings.simplefilter("ignore", Warning)
    for data in [df_train, df_val, df_test]:
        data.drop(columns=columns_to_drop, inplace=True)

#%% scaling remaining columns

# drop Timestamp (but it enables more accurate prediction - to hours/minutes/...)
# MinMaxScaling
# sine and cosine transformation - cyclic features
# or leave as it is

from sklearn.preprocessing import MinMaxScaler

# optionally exclude Names
columns_to_scale = [col for col in df_train.columns if col not in columns_to_standarize + ["Name"]]

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
columns = [col for col in df_train.columns if not col.startswith("Name")] # cols to shift
group_by_column = "Name"
target_column = "Close"
sort_columns = ["Date_year", "Date_month", "Date_day_of_month"]
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
