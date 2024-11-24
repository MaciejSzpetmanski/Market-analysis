#%% imports

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

#%% loading data

def load_data(directory_name, suffix=""):
    """
    wczytuje wszystkie pliki csv o danym przyrostku do jednej ramki danych,
    dodając kolumnę "name" z nazwą pliku
    nazwy plików są w formacie: nazwa_data(_1d).csv
    kolumny mają nazwy: Price	Adj Close	Close	High	Low	Open	Volume
    pierwsza kolumna zawiera informacje o dacie
    dwa pierwsze wiersze są puste
    :param directory_name: nazwa folderu z danymi
    :param suffix: przyrostek nazwy plików, np. "_data_1d.csv"
    :return: ramka danych
    """
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
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"adj close": "adjusted_close"})
    return df

def convert_column_types(df):
    df['volume'] = df['volume'].astype('int64')
    df['date'] = pd.to_datetime(df['date'])
    float_columns = ['close', 'high', 'low', 'open', 'adjusted_close']
    for col in float_columns:
        df[col] = df[col].astype('float')
    return df

def remove_name_values(df, name_value):
    df = df[~(df.name == name_value)]
    df = df.reset_index(drop=True)
    return df

#%% validation


#%% transformations

def add_time_columns(df):
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
    return df

def encode_names(df):
    name_column = df["name"]
    df = pd.get_dummies(df, columns=['name'], prefix='name',  dtype=int)
    df["name"] = name_column
    return df


def split_data(df, offset_years=1, test_size=0.6, random_state=42):
    last_date = df['date'].max()
    split_date = last_date - pd.DateOffset(years=offset_years)
    
    df_train = df[df['date'] <= split_date]
    df_train.shape
    
    df_val, df_test = train_test_split(df[df['date'] > split_date], test_size=test_size, random_state=random_state, stratify=df[df['date'] > split_date]['date'])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Warning)
        # removing Date column
        for data in [df_train, df_val, df_test]:
            data.drop(columns=['date'], inplace=True)
            
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    return df_train, df_val, df_test

def add_fractal_long_schemas(df, path, group_by_column, prefix, width):
    """
    wykrywa schematy fraktalne o zmiennej długości w ramcę danych,
    dodaje odpowiadające im kolumny i zwraca kopię
    :param df: ramka danych z kolumnami 'close', 'high', 'low', 'open', 'adjusted_close'
    :param path: ścieżka do folderu z funkcjami fraktalnymi
    :param group_by_column: kolumna, po której dane są grupowane
    :param prefix: przedrostek do nazw nowych kolumn
    :param width: ilość obserwacji, na których będą wykrywane schematy
    :return: kopia wejściowej ramki danych z dodanymi kolumnami
    """
    def apply_function(df, function, width):
        n = len(df)
        res = np.full(n, False)
        for i in range(width-1, n):
            data = df[i+1-width:i+1]
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
                new_columns[column_name] = np.concatenate([apply_function(group, function, width) for _, group in df.groupby(group_by_column)]).astype(int)
    res_data = pd.concat([df] + [pd.Series(value, name=key) for key, value in new_columns.items()], axis=1)
    return res_data

def add_cycle_columns(df, group_by_column, width):
    """
    dodaje do ramki danych kolumny z informacją o cyklu w danych cenowych
    :param df: ramka danych z kolumnami 'close', 'high', 'low', 'open', 'adjusted_close'
    :param group_by_column: kolumna, po której dane są grupowane
    :param width: ilość obserwacji, na których będzie wykrywany cykl
    :return: ramka danych z dodanymi kolumnami
    """
    def apply_cycle_function(df, width):
        n = len(df)
        res = np.full(n, "", dtype=object)
        for i in range(width-1, n):
            data = df[i+1-width:i+1]
            # get cycle name only
            res[i] = wykryj_typ_cyklu(data)[1]
        return res
    
    cycle_names = {
       "Cykl Neutralny": "neutral",
       "Cykl Zbieżny": "convergent",
       "Cykl Rozbieżny": "divergent",
       "": "not_detected"
    }
    
    df["cycle"] = np.concatenate([apply_cycle_function(group, width) for _, group in df.groupby(group_by_column)])
    df["cycle"] = df["cycle"].apply(lambda x: cycle_names[x])

    df = pd.get_dummies(df, columns=["cycle"], prefix="cycle",  dtype=int)
    return df

def add_ema_column(df, group_by_column, period):
    """
    dodaje do dancyh kolumny o wykładniczej sredniej ruchomej
    :param df: ramka danych z kolumnami 'close', 'high', 'low', 'open', 'adjusted_close'
    :param group_by_column: kolumna, po której dane są grupowane
    :param period: okres, w których liczona jest średnia
    :return: ramka danych z dodaną kolumną
    """
    df[f"ema_{period}"] = np.concatenate([ema(group.close, period) for _, group in df.groupby(group_by_column)])
    return df

def add_ema_columns(df, group_by_column, periods):
    for period in periods:
        df = add_ema_column(df, group_by_column, period)
    return df

def add_fractal_short_schemas(df, path, group_by_column, prefix):
    """
    wykrywa schematy fraktalne o stałej długości w ramcę danych,
    dodaje odpowiadające im kolumny i zwraca kopię
    :param df: ramka danych z kolumnami 'close', 'high', 'low', 'open', 'adjusted_close'
    :param path: ścieżka do folderu z funkcjami fraktalnymi
    :param group_by_column: kolumna, po której dane są grupowane
    :param prefix: przedrostek do nazw nowych kolumn
    :return: kopia wejściowej ramki danych z dodanymi kolumnami
    """
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

    res_data = pd.concat([df] + [pd.Series(value, name=key) for key, value in new_columns.items()], axis=1)
    return res_data

def remove_empty_columns(df_train, df_val, df_test):
    columns_to_remove = [col for col in df_train.columns if df_train[col].nunique() == 1]
    df_train = df_train.drop(columns=columns_to_remove)
    df_val = df_val.drop(columns=columns_to_remove)
    df_test = df_test.drop(columns=columns_to_remove)
    return df_train, df_val, df_test

def standarize_training_columns(df_train, columns_to_standarize, group_by_column):
    scalers = {}
    for name, group in df_train.groupby(group_by_column):
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(group[columns_to_standarize])
        df_train.loc[df_train[group_by_column] == name, columns_to_standarize] = scaled_values
        scalers[name] = scaler
    return df_train, scalers

def standarize_columns(df, scalers, columns_to_standarize, group_by_column):
    for name, group in df.groupby(group_by_column):
        scaler = scalers[name]
        scaled_values = scaler.transform(group[columns_to_standarize])
        df.loc[df[group_by_column] == name, columns_to_standarize] = scaled_values
    return df

def save_object(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def load_object(path):
    with open(path, 'rb') as file:
        obj = pickle.load(file)
        return obj

def inverse_target_scaling(data, name, scalers):
    # name = "AAPL"
    columns_to_standarize = ["adjusted_close", "close", "high", "low", "open", "volume"]
    
    data_scaled = data[data[f'name_{name}'] == 1][columns_to_standarize]
    data_original = scalers[name].inverse_transform(data_scaled)
    close_original = data_original[:, 1]
    return close_original

def create_time_series(data, columns, group_by_column, target_column, sort_columns, n, k=1):
    """
    zmienia dane cenowe na szeregi czasowe
    :param data: ramka danych z kolumnami 'close', 'high', 'low', 'open', 'adjusted_close'
    :param columns: lista kolumn, które mają być zwielokrotnione dla każdej obserwacji
    :param group_by_column: kolumna, po której dane są grupowane
    :param target_column: zmienna celu
    :param sort_columns: lista kolumn, po których dane będą sortowane
    :param n: długość szeregu (ilość obserwacji)
    :param k: odległość w liczbie obserwacji zmiennej celu od ostatniej znanej obserwacji
    :return: ramka danych
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

def create_time_series_set(data, columns, group_by_column, target_column, sort_columns, n, max_target_horizon):
    sets = {}
    for k in range(1, max_target_horizon + 1):
        sets[k] = create_time_series(data, columns, group_by_column, target_column, sort_columns, n, k)
        sets[k] = sets[k].reset_index(drop=True)
    return sets

def drop_columns_from_sets(sets, max_target_horizon, columns):
    for k in range(1, max_target_horizon + 1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Warning)
            sets[k] = sets[k].drop(columns=columns)
    return sets

def get_target_columns(sets, max_target_horizon):
    y_sets = {}
    for k in range(1, max_target_horizon + 1):
        y_sets[k] = sets[k]["y"]
    sets = drop_columns_from_sets(sets, max_target_horizon, ["y"])
    return sets, y_sets

def save_data_sets(X_sets, y_sets, directory, x_name, y_name):
    """
    zapisuje dane z wielu ramek danych do plików, dodając informacje o indeksie w nazwie
    :param X_sets: słownik ramek danych
    :param y_sets: słownik ramek danych, o tej samej długości co X_sets
    :param directory: katalog do zapisu
    :param x_name: nazwa plików dla zbiorów z X_sets
    :param y_name: nazwa plików dla zbiorów z y_sets
    :return:
    """
    n = len(X_sets)
    for k in range(1, n + 1):
        X_path = os.path.join(directory, f"{x_name}_{k}.csv")
        y_path = os.path.join(directory, f"{y_name}_{k}.csv")
        X_sets[k].to_csv(X_path, index=False)
        y_sets[k].to_csv(y_path, index=False)
      
#%%

def pipeline():
    df = load_data("data", "_data_1d.csv")
    df = convert_column_types(df)
    df = remove_name_values(df, "EURGBP=X")
    
    # OPTIONAL validation
    
    df = add_time_columns(df)
    df = encode_names(df)
    df_train, df_val, df_test = split_data(df)
    
    functions_path = 'Wskaźniki itp/Schemat zmienna długość'
    group_by_column = 'name'
    width = 20
    
    df_train = add_fractal_long_schemas(df_train, functions_path, group_by_column, "long_formation_", width)
    df_val = add_fractal_long_schemas(df_val, functions_path, group_by_column, "long_formation_", width)
    df_test = add_fractal_long_schemas(df_test, functions_path, group_by_column, "long_formation_", width)
        
    df_train = add_cycle_columns(df_train, group_by_column, width)
    df_val = add_cycle_columns(df_val, group_by_column, width)
    df_test = add_cycle_columns(df_test, group_by_column, width)
        
    ema_periods = [10, 20, 50, 100, 200]
    
    df_train = add_ema_columns(df_train, group_by_column, ema_periods)
    df_val = add_ema_columns(df_val, group_by_column, ema_periods)
    df_test = add_ema_columns(df_test, group_by_column, ema_periods)
    
    functions_path = 'Wskaźniki itp/Schemat stała długość'
    
    df_train = add_fractal_short_schemas(df_train, functions_path, group_by_column, "short_formation_")
    df_val = add_fractal_short_schemas(df_val, functions_path, group_by_column, "short_formation_")
    df_test = add_fractal_short_schemas(df_test, functions_path, group_by_column, "short_formation_")
    
    df_train, df_val, df_test = remove_empty_columns(df_train, df_val, df_test)
    
    columns_to_standarize = ["adjusted_close", "close", "high", "low", "open", "volume"]
    
    df_train, scalers = standarize_training_columns(df_train, columns_to_standarize, group_by_column)
    df_val = standarize_columns(df_val, scalers, columns_to_standarize, group_by_column)
    df_test = standarize_columns(df_test, scalers, columns_to_standarize, group_by_column)
    
    scalers_path = 'scalers/scalers.pkl'
    
    save_object(scalers, scalers_path)
    
    columns = [col for col in df_train.columns if not col.startswith("name")]
    target_column = "close"
    sort_columns = ["date_year", "date_month", "date_day_of_month"]
    n = 20
    max_target_horizon = 5
    
    train_sets = create_time_series_set(df_train, columns, group_by_column, target_column, sort_columns, n, max_target_horizon)
    val_sets = create_time_series_set(df_val, columns, group_by_column, target_column, sort_columns, n, max_target_horizon)
    test_sets = create_time_series_set(df_test, columns, group_by_column, target_column, sort_columns, n, max_target_horizon)
    
    train_sets, y_train_sets = get_target_columns(train_sets, max_target_horizon)
    val_sets, y_val_sets = get_target_columns(val_sets, max_target_horizon)
    test_sets, y_test_sets = get_target_columns(test_sets, max_target_horizon)
    
    columns_to_drop = ["name"]
    
    train_sets = drop_columns_from_sets(train_sets, max_target_horizon, columns_to_drop)
    val_sets = drop_columns_from_sets(val_sets, max_target_horizon, columns_to_drop)
    test_sets = drop_columns_from_sets(test_sets, max_target_horizon, columns_to_drop)
    
    save_directory = "datasets"
    
    save_data_sets(train_sets, y_train_sets, save_directory, "df_train", "y_train")
    save_data_sets(val_sets, y_val_sets, save_directory, "df_val", "y_val")
    save_data_sets(test_sets, y_test_sets, save_directory, "df_test", "y_test")
    
    