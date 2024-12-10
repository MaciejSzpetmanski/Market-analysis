#%% imports

import os
import numpy as np
import pandas as pd
from feature_engine.datetime import DatetimeFeatures
import warnings
import importlib.util
from indicators.cykl import wykryj_typ_cyklu
from indicators.EMA import ema
from sklearn.preprocessing import StandardScaler
import pickle

#%% global variables

DATA_DIRECTORY = "data"
FILE_SUFFIX = "_data_1d.csv"
NAME_TO_REMOVE = "EURGBP=X"
GROUP_BY_COLUMN = "name"
LONG_SCHEMA_PATH = "indicators/Schemat zmienna długość"
LONG_SCHEMA_PREFIX = "long_formation_"
SCHEMA_WIDTH = 5 #3 #10 #5 #20
EMA_PERIODS = [10, 20, 50, 100, 200]
SHORT_SCHEMA_PATH = "indicators/Schemat stała długość"
SHORT_SCHEMA_PREFIX = "short_formation_"
TRAIN_COLUMNS_PATH = "scalers/train_columns.pkl"
COLUMNS_TO_STANDARDIZE = ["adjusted_close", "close", "high", "low", "open", "volume"]
SCALERS_PATH = "scalers/scalers.pkl"
TARGET_COLUMN = "close"
SORT_COLUMNS = ["date_year", "date_month", "date_day_of_month"]
TIME_SERIES_LENGTH = 5 #3 #10 #5 #20
MAX_TARGET_HORIZON = 5
DATA_OUTPUT_DIRECTORY = "datasets"
X_TRAIN_OUTPUT_NAME = "df_train"
X_VAL_OUTPUT_NAME = "df_val"
X_TEST_OUTPUT_NAME = "df_test"
Y_TRAIN_OUTPUT_NAME = "y_train"
Y_VAL_OUTPUT_NAME = "y_val"
Y_TEST_OUTPUT_NAME = "y_test"

#%% loading data

def load_data_from_file(path):
    df = None
    if os.path.isfile(path) and path.endswith(".csv"):
        df = pd.read_csv(path)
    return df

def load_data(directory_name, suffix=""):
    df = None
    for file_name in os.listdir(directory_name):
        if not file_name.endswith(suffix):
            continue
        file_path = os.path.join(directory_name, file_name)
        df_input = load_data_from_file(file_path)
        if df_input is None:
            continue
        # parsing filename
        file_name += "_"
        file_info = file_name.replace(".csv", "").split("_")
        # adding source information
        df_input["name"] = file_info[0]
        if df is None and df_input is None:
            continue
        df = pd.concat([df, df_input], ignore_index=True)
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

def validate_data(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Not a DataFrame")
    expected_columns = {'date', 'adjusted_close', 'close', 'high', 'low', 'open', 'volume', 'name'}
    if set(df.columns) != expected_columns:
        raise ValueError("Wrong column names")
    try:
        convert_column_types(df)
    except:
        raise ValueError("Wrong types")

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
    
    # sorting by date
    df_val_test = df[df['date'] > split_date].sort_values(by=['date_year', 'date_month', 'date_day_of_month'])
    
    val_test_len = len(df_val_test)
    val_len = int(val_test_len * (1 - test_size))
    
    df_val = df_val_test.iloc[:val_len]
    df_test = df_val_test.iloc[val_len:]
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Warning)
        # removing date column
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
    def _apply_function(df, function, width):
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
                new_columns[column_name] = np.concatenate([_apply_function(group, function, width) for _, group in df.groupby(group_by_column)]).astype(int)
    res_data = pd.concat([df] + [pd.Series(value, name=key) for key, value in new_columns.items()], axis=1)
    return res_data

def add_cycle_columns(df, group_by_column, width=10):
    """
    dodaje do ramki danych kolumny z informacją o cyklu w danych cenowych
    :param df: ramka danych z kolumnami 'close', 'high', 'low', 'open', 'adjusted_close'
    :param group_by_column: kolumna, po której dane są grupowane
    :param width: ilość obserwacji, na których będzie wykrywany cykl
    :return: ramka danych z dodanymi kolumnami
    """
    def _apply_cycle_function(df, width=10):
        n = len(df)
        res = np.full(n, "", dtype=object)
        for i in range(width-1, n):
            data = df[i+1-width:i+1]
            # get cycle name only
            res[i] = wykryj_typ_cyklu(data, okres=width)[1]
        return res
    
    cycle_names = {
       "Cykl Neutralny": "neutral",
       "Cykl Zbieżny": "convergent",
       "Cykl Rozbieżny": "divergent",
       "": "not_detected"
    }
    
    df["cycle"] = np.concatenate([_apply_cycle_function(group, width) for _, group in df.groupby(group_by_column)])
    df["cycle"] = df["cycle"].apply(lambda x: cycle_names[x])
    
    df = pd.get_dummies(df, columns=["cycle"], prefix="cycle", dtype=int)
    
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

def standardize_training_columns(df_train, columns_to_standardize, group_by_column):
    scalers = {}
    for name, group in df_train.groupby(group_by_column):
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(group[columns_to_standardize])
        df_train.loc[df_train[group_by_column] == name, columns_to_standardize] = scaled_values
        scalers[name] = scaler
    return df_train, scalers

def standardize_columns(df, scalers, columns_to_standardize, group_by_column):
    for name, group in df.groupby(group_by_column):
        scaler = scalers[name]
        scaled_values = scaler.transform(group[columns_to_standardize])
        df.loc[df[group_by_column] == name, columns_to_standardize] = scaled_values
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
    columns_to_standardize = ["adjusted_close", "close", "high", "low", "open", "volume"]
    
    data_scaled = data[data[f'name_{name}'] == 1][columns_to_standardize]
    data_original = scalers[name].inverse_transform(data_scaled)
    # close_original = data_original[:, 1]
    # return close_original
    return data_original

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
    print("Wczytywanie danych")
    df = load_data(DATA_DIRECTORY, FILE_SUFFIX)
    validate_data(df)
    print("Zmiana typów kolumn")
    df = convert_column_types(df)
    df = remove_name_values(df, NAME_TO_REMOVE)
    
    df = df.sort_values(by=["name", "date"])
    
    print("Dodawanie kolumn czasowych")
    df = add_time_columns(df)
    print("Kodowanie nazw")
    df = encode_names(df)
    print("Podział na zbiory")
    df_train, df_val, df_test = split_data(df)
    
    print("Standaryzacja kolumn")
    df_train.volume = df_train.volume.astype(float)
    df_val.volume = df_val.volume.astype(float)
    df_test.volume = df_test.volume.astype(float)

    df_train, scalers = standardize_training_columns(df_train, COLUMNS_TO_STANDARDIZE, GROUP_BY_COLUMN)
    df_val = standardize_columns(df_val, scalers, COLUMNS_TO_STANDARDIZE, GROUP_BY_COLUMN)
    df_test = standardize_columns(df_test, scalers, COLUMNS_TO_STANDARDIZE, GROUP_BY_COLUMN)
    
    print("Zapisywanie obiektów skalujących")
    save_object(scalers, SCALERS_PATH)
    
    print("Dodawanie cech fraktalnych o zmiennej długości")
    df_train = add_fractal_long_schemas(df_train, LONG_SCHEMA_PATH, GROUP_BY_COLUMN, LONG_SCHEMA_PREFIX, SCHEMA_WIDTH)
    df_val = add_fractal_long_schemas(df_val, LONG_SCHEMA_PATH, GROUP_BY_COLUMN, LONG_SCHEMA_PREFIX, SCHEMA_WIDTH)
    df_test = add_fractal_long_schemas(df_test, LONG_SCHEMA_PATH, GROUP_BY_COLUMN, LONG_SCHEMA_PREFIX, SCHEMA_WIDTH)

    print("Dodawanie cech cykli cenowych")
    # requires 10 records by default
    df_train = add_cycle_columns(df_train, GROUP_BY_COLUMN, SCHEMA_WIDTH)
    df_val = add_cycle_columns(df_val, GROUP_BY_COLUMN, SCHEMA_WIDTH)
    df_test = add_cycle_columns(df_test, GROUP_BY_COLUMN, SCHEMA_WIDTH)

    print("Dodawanie cech średniej kroczącej")
    df_train = add_ema_columns(df_train, GROUP_BY_COLUMN, EMA_PERIODS)
    df_val = add_ema_columns(df_val, GROUP_BY_COLUMN, EMA_PERIODS)
    df_test = add_ema_columns(df_test, GROUP_BY_COLUMN, EMA_PERIODS)

    print("Dodawanie cech fraktalnych o stałej długości")
    df_train = add_fractal_short_schemas(df_train, SHORT_SCHEMA_PATH, GROUP_BY_COLUMN, SHORT_SCHEMA_PREFIX)
    df_val = add_fractal_short_schemas(df_val, SHORT_SCHEMA_PATH, GROUP_BY_COLUMN, SHORT_SCHEMA_PREFIX)
    df_test = add_fractal_short_schemas(df_test, SHORT_SCHEMA_PATH, GROUP_BY_COLUMN, SHORT_SCHEMA_PREFIX)

    print("Usuwanie pustych kolumn")
    df_train, df_val, df_test = remove_empty_columns(df_train, df_val, df_test)
    
    print("Zapisywanie kolumn treningowych")
    save_object(df_train.columns, TRAIN_COLUMNS_PATH)
    
    columns_to_shift = [col for col in df_train.columns if not col.startswith("name")]
    
    print("Tworzenie szeregów czasowych")
    train_sets = create_time_series_set(df_train, columns_to_shift, GROUP_BY_COLUMN, TARGET_COLUMN, SORT_COLUMNS, TIME_SERIES_LENGTH, MAX_TARGET_HORIZON)
    val_sets = create_time_series_set(df_val, columns_to_shift, GROUP_BY_COLUMN, TARGET_COLUMN, SORT_COLUMNS, TIME_SERIES_LENGTH, MAX_TARGET_HORIZON)
    test_sets = create_time_series_set(df_test, columns_to_shift, GROUP_BY_COLUMN, TARGET_COLUMN, SORT_COLUMNS, TIME_SERIES_LENGTH, MAX_TARGET_HORIZON)

    print("Wyodrębnianie zmiennej celu")
    train_sets, y_train_sets = get_target_columns(train_sets, MAX_TARGET_HORIZON)
    val_sets, y_val_sets = get_target_columns(val_sets, MAX_TARGET_HORIZON)
    test_sets, y_test_sets = get_target_columns(test_sets, MAX_TARGET_HORIZON)
    
    columns_to_drop = ["name"]

    print("Usuwanie kolumny z nazwą")
    train_sets = drop_columns_from_sets(train_sets, MAX_TARGET_HORIZON, columns_to_drop)
    val_sets = drop_columns_from_sets(val_sets, MAX_TARGET_HORIZON, columns_to_drop)
    test_sets = drop_columns_from_sets(test_sets, MAX_TARGET_HORIZON, columns_to_drop)
    
    print("Zapisywanie zbiorów danych")
    save_data_sets(train_sets, y_train_sets, DATA_OUTPUT_DIRECTORY, X_TRAIN_OUTPUT_NAME, Y_TRAIN_OUTPUT_NAME)
    save_data_sets(val_sets, y_val_sets, DATA_OUTPUT_DIRECTORY, X_VAL_OUTPUT_NAME, Y_VAL_OUTPUT_NAME)
    save_data_sets(test_sets, y_test_sets, DATA_OUTPUT_DIRECTORY, X_TEST_OUTPUT_NAME, Y_TEST_OUTPUT_NAME)

    print("Zakończono przetwarzanie")
    

def main():
    pipeline()

if __name__ == "__main__":
    main()
