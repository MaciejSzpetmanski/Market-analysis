import os
import numpy as np
import pandas as pd
from feature_engine.datetime import DatetimeFeatures
from sklearn.model_selection import train_test_split
import warnings
import importlib.util
import sys
sys.path.append(os.path.abspath("Wskaźniki itp"))
from cykl import wykryj_typ_cyklu
from EMA import ema
from sklearn.preprocessing import StandardScaler
import pickle
from data_pipeline import *
import data_pipeline

#%%

def create_time_series_vector(data, columns, n):
    shifted_columns = {}
    for i in range(1, n):
        for col in columns:
            shifted_columns[col + "_" + str(i)] = data[col].shift(-i)
    date_columns = [col for col in columns if col.startswith("date")]
    # for col in date_columns:
    #     shifted_columns[col + "_y"] = data.groupby(group_by_column, group_keys=False)[col].shift(-(n-1+k))
    # target column
    res_data = pd.concat([data] + [pd.Series(value, name=key) for key, value in shifted_columns.items()], axis=1)
    # removing rows with missing values
    res_data = res_data.groupby(group_by_column, group_keys=False).head(-(n-1))
    last_row = res_data.tail(1).reset_index(drop=True)
    return last_row

# TODO check weekend, free days
def validate_y_date(date, name):
    pass

def create_date_vector(date):
    df_date = pd.DataFrame()
    df_date["date"] = [date]
    df_date["date"] = pd.to_datetime(df_date["date"])
    df_date = add_time_columns(df_date)
    return df_date

def create_preddiction_date_vector(date):
    y_date = create_date_vector(date)
    y_date = y_date.drop(columns=["date"])
    y_date = y_date.add_suffix('_y')
    return y_date

#%%

def prepare_data_for_prediction(path, name, date):
    path = "data/^DJI_data_1d.csv"
    df = data_pipeline.load_data_from_file(path)
    # TODO checking input data size
    df = data_pipeline.convert_column_types(df)
    df = add_time_columns(df)
    
    df = df.sort_values(by=["date"])
    
    # TODO trimming, filtering data
    df = df.tail(20).reset_index(drop=True)
    
    columns = load_object("scalers/train_columns.pkl")
    name_columns = [col for col in columns if col.startswith("name_")]
    name = "^DJI"
    for col in name_columns:
        df[col] = 0
    df[f"name_{name}"] = 1
    # artificially add name column for groupping
    df["name"] = name
    
    functions_path = 'Wskaźniki itp/Schemat zmienna długość'
    group_by_column = 'name'
    width = 20

    print("Dodawanie cech fraktalnych o zmiennej długości")
    df = add_fractal_long_schemas(df, functions_path, group_by_column, "long_formation_", width)
    print("Dodawanie cech cykli cenowych")
    df = add_cycle_columns(df, group_by_column, width)
    
    ema_periods = [10, 20, 50, 100, 200]

    print("Dodawanie cech średniej kroczącej")
    df = add_ema_columns(df, group_by_column, ema_periods)
    
    functions_path = 'Wskaźniki itp/Schemat stała długość'

    print("Dodawanie cech fraktalnych o stałej długości")
    df = add_fractal_short_schemas(df, functions_path, group_by_column, "short_formation_")
    
    print("Wybór kolumn")
    # TODO niewykryte kolumny (cykle)
    missing_columns = [col for col in columns if col not in df.columns]
    for col in missing_columns:
        df[col] = 0
    df = df[columns] # removing date, setting order of columns
    
    columns_to_standarize = ["adjusted_close", "close", "high", "low", "open", "volume"]

    scalers = load_object("scalers/scalers.pkl")

    print("Standaryzacja kolumn")
    df.volume = df.volume.astype(float)
    df = standardize_columns(df, scalers, columns_to_standarize, group_by_column)

    print("Tworzenie szeregów czasowych")
    columns = [col for col in df.columns if not col.startswith("name")]
    n = 20
    output_vector = create_time_series_vector(df, columns, n)
    
    columns_to_drop = ["name"]
    print("Usuwanie kolumny z nazwą")
    output_vector = output_vector.drop(columns=columns_to_drop)
    
    # prediction date
    y_date = create_preddiction_date_vector(date)
    
    result = pd.concat([output_vector, y_date], axis=1)
    
    return result

    