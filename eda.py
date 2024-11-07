#%% working directory

import os

path = "D:\Studia\semestr7\inźynierka\Market-analysis"
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

#%% converting column types

df.describe()
df.info()

df['Volume']
df['Volume'] = df['Volume'].astype('int64')

float_columns = ['Adj Close', 'Close', 'High', 'Low', 'Open']
for col in float_columns:
    df[col] = df[col].astype('float')

df.info()

#%% removing duplicates

subset = list(filter(lambda x: x != 'Freq', df.columns))
df.drop_duplicates(subset=subset, inplace=True)
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

df = dt_features.fit_transform(df)
df['Date'] = date

df.info()

# TODO identify useless columns

#%% name encoding (one-hot)

df = pd.get_dummies(df, columns=['Name'], prefix='Name',  dtype=int)
df.shape
df.columns

#%% creating new columns

# TODO

#%% split

from sklearn.model_selection import train_test_split

last_date = df['Date'].max()
split_date = last_date - pd.DateOffset(years=1)

df_train = df[df['Date'] <= split_date]
df_train.shape

# TODO stratify by day-month-year
df_val, df_test = train_test_split(df[df['Date'] > split_date], test_size=0.6, random_state=42, stratify=df[df['Date'] > split_date]['Date_year'])

# OPTIONAL hide warnings
for data in [df_train, df_val, df_test]:
    data.drop(columns=['Date'], inplace=True)

print(f'train: {df_train.shape}')
print(f'val: {df_val.shape}')
print(f'test: {df_test.shape}')

#%% correlation

import matplotlib.pyplot as plt
import seaborn as sns

# pearson / spearman / ...
corr_matrix = df_train.drop(columns=['Freq']).corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Covariance Matrix Heatmap')
plt.show()

#%% histograms

df_train.hist(bins=50, figsize=(10, 8))
plt.suptitle('Histograms of DataFrame Columns')
plt.tight_layout()
plt.show()

#%% full boxplots

num_columns = df_train.shape[1]
fig, axes = plt.subplots(1, num_columns, figsize=(15, 6), sharey=False)

for i, column in enumerate(df_train.drop(columns=['Freq']).columns):
    sns.boxplot(y=df_train[column], ax=axes[i])
    axes[i].set_title(f'Boxplot of {column}')

plt.tight_layout()
plt.show()

df_train.describe()

#%% trimmed boxplots

num_columns = df_train.shape[1]
fig, axes = plt.subplots(1, num_columns, figsize=(15, 6), sharey=False)

for i, column in enumerate(df_train.drop(columns=['Freq']).columns):
    q = df_train[column].quantile(0.99)
    sns.boxplot(y=df_train[df_train[column] < q][column], ax=axes[i])
    axes[i].set_title(f'Boxplot of {column}')

plt.tight_layout()
plt.show()

df_train.describe()

#%% outliers

# TODO

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



#%% normalisation



#%%

# outliers
# additional columns
# pca (high correlation)
