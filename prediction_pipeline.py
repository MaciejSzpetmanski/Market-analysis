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

#%%

def prepare_data_for_prediction(path, name, ):
    df = load_data_from_file(path)
    df = convert_column_types(df)
    df = add_time_columns(df)
    
    columns = load_object("scalers/train_columns.pkl")
    name_columns = [col for col in columns if col.startswith("name_")]
    for col in name_columns:
        df[col] = 0
    df[f"name_{name}"] = 1
    # artificially add name column for groupping
    df["name"] = 1
    
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
    df = df[columns]
    
    columns_to_standarize = ["adjusted_close", "close", "high", "low", "open", "volume"]

    scalers = load_object("scalers/scalers.pkl")

    print("Standaryzacja kolumn")
    df = standarize_columns(df, scalers, columns_to_standarize, group_by_column)
    
    # TODO do not generate y
    # create_time_series(data, columns, group_by_column, target_column, sort_columns, n, k=1)
    
    columns_to_drop = ["name"]

    print("Usuwanie kolumny z nazwą")
    # TODO remove name column
    
    return df

    