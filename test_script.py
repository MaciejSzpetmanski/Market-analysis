import pytest
import pandas as pd
import numpy as np
import data_pipeline as dp
from pandas.api.types import is_numeric_dtype, is_string_dtype
import os
import copy

@pytest.mark.parametrize(("path", "n_rows"), [
    ("data/^DJI_data.csv", 1696),
])
def test_load_data(path, n_rows):
    loaded_data = dp.load_data_from_file(path)
    assert isinstance(loaded_data, pd.DataFrame)
    expected_columns = {'date', 'adjusted_close', 'close', 'high', 'low', 'open', 'volume'}
    assert set(loaded_data.columns) == expected_columns
    assert len(loaded_data) == n_rows
    

@pytest.mark.parametrize("path", [
    ("data/^DJI_data_1h.csv"),
])
def test_load_empty_data(path):
    loaded_data = dp.load_data_from_file(path)
    assert isinstance(loaded_data, pd.DataFrame)
    
@pytest.mark.parametrize("path", [
    ("data_for_tests/test.txt"),
])
def test_load_noncsv_data(path):
    loaded_data = dp.load_data_from_file(path)
    assert loaded_data is None

@pytest.mark.parametrize("path", [
    ("data/no_data.csv"),
])
def test_load_nonexisting_data(path):
    loaded_data = dp.load_data_from_file(path)
    assert loaded_data is None
    
@pytest.mark.parametrize(("directory_name", "suffix", "n_rows"), [
    ("data_for_tests", "_data_1d.csv", 3424),
])
def test_merge(directory_name, suffix, n_rows):
    loaded_data = dp.load_data(directory_name, suffix)
    assert isinstance(loaded_data, pd.DataFrame)
    expected_columns = {'date', 'adjusted_close', 'close', 'high', 'low', 'open', 'volume', 'name'}
    assert set(loaded_data.columns) == expected_columns
    assert len(loaded_data) == n_rows

@pytest.mark.parametrize(("directory_name", "suffix"), [
    ("data_for_tests", "abc"),
])
def test_empty_merge(directory_name, suffix):
    loaded_data = dp.load_data(directory_name, suffix)
    assert loaded_data is None

@pytest.mark.parametrize(("df"), [
    (dp.load_data("data_for_tests", "_data_1d.csv")),
])
def test_convert_column_types(df):
    columns1 = df.columns
    df = dp.convert_column_types(df)
    columns2 = df.columns
    assert set(columns1) == set(columns2)
    for col in ['adjusted_close', 'close', 'high', 'low', 'open', 'volume']:
        assert is_numeric_dtype(df[col])
    assert is_string_dtype(df['name'])
    
@pytest.mark.parametrize(("df"), [
    (dp.load_data("data_for_tests", "_data_1d.csv")),
])
def test_validate_correct_data(df):
    try:
        dp.validate_data(df)
        assert True
    except:
        assert False
    
@pytest.mark.parametrize(("df"), [
    (dp.load_data("data_for_tests", "")),
    (pd.DataFrame({
        "adjusted_close": ["a", "b", "c"],
        "close": [1.0, 2.0, None],
        "high": [100, 200, 300],
        "low": [90, 190, "x"],
        "open": [0, 0, None],
        "volume": [1000, 2000, None],
        "name": [123, 456, 789]
    })),
    (pd.DataFrame({
        "adjusted_close": [],
        "close": [],
        "high": [],
        "low": [],
        "open": [],
        "volume": [],
        "name": []
    })),
    (pd.DataFrame({
        "adjusted_close": [1],
        "close": [1],
        "high": [1],
        "low": [1],
        "open": [None],
        "volume": [1],
        "name": ["a"]
    })),
])
def test_validate_incorrect_data(df):
    try:
        dp.validate_data(df)
        assert False
    except:
        assert True

@pytest.mark.parametrize(("df"), [
    (dp.load_data("data_for_tests", "_data_1d.csv")),
])
def test_add_time_columns(df):
    features_to_extract=[
        'month', 'quarter', 'semester', 'year', 'week',
        'day_of_week', 'day_of_month', 'day_of_year',
        'weekend', 'month_start', 'month_end',
        'quarter_start', 'quarter_end', 'year_start',
        'year_end', 'leap_year', 'days_in_month'
    ]
    df = dp.add_time_columns(df)
    for col in features_to_extract:
        column_name = "date_" + col
        assert column_name in df.columns
        
@pytest.mark.parametrize(("df", "n_names"), [
    (dp.load_data("data_for_tests", "_data_1d.csv"), 2),
])
def test_encode_names(df, n_names):
    df = dp.encode_names(df)
    name_columns = [col for col in df.columns if col.startswith("name_")]
    assert len(name_columns) == n_names
    assert len(name_columns) == len(np.unique(df.name))
    for name in name_columns:
        assert np.sum(df.name == name.lstrip("name_")) > 0

@pytest.mark.parametrize(("df"), [
    (dp.load_data("data_for_tests", "_data_1d.csv")),
])
def test_split_data(df):
    df = dp.convert_column_types(df)
    df = dp.add_time_columns(df)
    df_train, df_val, df_test = dp.split_data(df)
    date_train = df_train.date_year.astype(str) + "-" + df_train.date_month.astype(str) + "-" + df_train.date_day_of_month.astype(str)
    date_val = df_val.date_year.astype(str) + "-" + df_val.date_month.astype(str) + "-" + df_val.date_day_of_month.astype(str)
    date_test = df_test.date_year.astype(str) + "-" + df_test.date_month.astype(str) + "-" + df_test.date_day_of_month.astype(str)
    
    date_train = pd.to_datetime(date_train)
    date_val = pd.to_datetime(date_val)
    date_test = pd.to_datetime(date_test)
    
    assert np.sum(max(date_train) > date_val) == 0
    assert np.sum(max(date_val) > date_test) == 0

@pytest.mark.parametrize(("df"), [
    (dp.load_data("data_for_tests", "_data_1d.csv")[::50].reset_index(drop=True)),
])
def test_add_fractal_long_schemas(df):
    df = dp.add_fractal_long_schemas(df, dp.LONG_SCHEMA_PATH, dp.GROUP_BY_COLUMN, dp.LONG_SCHEMA_PREFIX, dp.SCHEMA_WIDTH)
    schema_columns = [col for col in df.columns if col.startswith(dp.LONG_SCHEMA_PREFIX)]
    for col in schema_columns:
        assert is_numeric_dtype(df[col])
        assert np.max(df[col]) <= 1
        assert np.min(df[col]) >= 0

@pytest.mark.parametrize(("df"), [
    (dp.load_data("data_for_tests", "_data_1d.csv")[::50].reset_index(drop=True)),
])
def test_add_cycle_columns(df):
    n_columns1 = len(df.columns)
    df = dp.add_cycle_columns(df, dp.GROUP_BY_COLUMN, dp.SCHEMA_WIDTH)
    n_columns2 = len(df.columns)
    assert n_columns2 > n_columns1

@pytest.mark.parametrize(("df"), [
    (dp.load_data("data_for_tests", "_data_1d.csv")[::50].reset_index(drop=True)),
])
def test_add_ema_columns(df):
    n_columns1 = len(df.columns)
    df = dp.add_ema_columns(df, dp.GROUP_BY_COLUMN, dp.EMA_PERIODS)
    n_columns2 = len(df.columns)
    assert n_columns2 - n_columns1 == 5
    ema_columns = [col for col in df.columns if col.startswith("ema_")]
    for col in ema_columns:
        assert is_numeric_dtype(df[col])
        assert df[col].isnull().sum() == 0
    
@pytest.mark.parametrize(("df"), [
    (dp.load_data("data_for_tests", "_data_1d.csv")[::50].reset_index(drop=True)),
])
def test_add_fractal_short_schemas(df):
    df = dp.add_fractal_short_schemas(df, dp.SHORT_SCHEMA_PATH, dp.GROUP_BY_COLUMN, dp.SHORT_SCHEMA_PREFIX)
    schema_columns = [col for col in df.columns if col.startswith(dp.LONG_SCHEMA_PREFIX)]
    for col in schema_columns:
        assert is_numeric_dtype(df[col])
        assert np.max(df[col]) <= 1
        assert np.min(df[col]) >= 0

@pytest.mark.parametrize(("df"), [
    (dp.load_data("data_for_tests", "_data_1d.csv")),
])
def test_standardize_training_columns(df):
    eps = 1e-12
    df.volume = df.volume.astype(float)
    df, scalers = dp.standardize_training_columns(df, dp.COLUMNS_TO_STANDARDIZE, dp.GROUP_BY_COLUMN)
    for name, group in df.groupby(dp.GROUP_BY_COLUMN):
        for col in dp.COLUMNS_TO_STANDARDIZE:
            assert abs(np.mean(group[col])) < eps
            assert abs(np.std(group[col]) - 1) < eps
    assert scalers != None
    assert len(scalers) == df.name.nunique()

@pytest.mark.parametrize(("obj", "path"), [
    (None, "test_output/obj"),
    ([None, 2, 1], "test_output/obj1"),
])
def test_save_object(obj, path):
    dp.save_object(obj, path)
    assert os.path.exists(path)
    
@pytest.mark.parametrize(("obj", "path"), [
    (None, "test_output/obj"),
    ([None, 2, 1], "test_output/obj1"),
])
def test_load_object(obj, path):
    loaded_obj = dp.load_object(path)
    assert loaded_obj == obj

@pytest.mark.parametrize(("df", "scalers"), [
    (dp.load_data("data_for_tests", "_data_1d.csv"), dp.load_object("scalers/scalers.pkl")),
])
def test_standardize_columns(df, scalers):
    df.volume = df.volume.astype(float)
    df = dp.standardize_columns(df, scalers, dp.COLUMNS_TO_STANDARDIZE, dp.GROUP_BY_COLUMN)
    for name, group in df.groupby(dp.GROUP_BY_COLUMN):
        for col in dp.COLUMNS_TO_STANDARDIZE:
            assert is_numeric_dtype(df[col])
            assert df[col].isnull().sum() == 0
           

@pytest.mark.parametrize(("df", "name", "scalers"), [
    (dp.load_data_from_file("data_for_tests/^DJI_data_1d.csv"), "^DJI", dp.load_object("scalers/scalers.pkl")),
    (dp.load_data_from_file("data_for_tests/^FCHI_data_1d.csv"), "^FCHI", dp.load_object("scalers/scalers.pkl")),
])
def test_inverse_target_scaling(df, name, scalers):
    df["name"] = name
    df = dp.encode_names(df)
    df_original = copy.deepcopy(df)
    df_scaled = dp.standardize_columns(df, scalers, dp.COLUMNS_TO_STANDARDIZE, dp.GROUP_BY_COLUMN)
    df_inversed = dp.inverse_target_scaling(df_scaled, name, scalers)
    
    df_original = df_original[dp.COLUMNS_TO_STANDARDIZE]
    assert df_inversed.shape[1] == len(dp.COLUMNS_TO_STANDARDIZE)
    
    eps = 1e-12
    assert np.allclose(df_original.values, df_inversed, atol=eps)
    
@pytest.mark.parametrize(("df"), [
    (dp.load_data("data_for_tests", "_data_1d.csv")),
])
def test_create_time_series(df):
    df = dp.add_time_columns(df)
    columns_to_shift = [col for col in df.columns if not col.startswith("name")]
    time_series = dp.create_time_series(df, columns_to_shift, dp.GROUP_BY_COLUMN, dp.TARGET_COLUMN, dp.SORT_COLUMNS, dp.TIME_SERIES_LENGTH)
    shifted_columns = [col for col in time_series.columns if not col.endswith("y")]
    assert len(shifted_columns) == (len(df.columns) - len(columns_to_shift)) + len(columns_to_shift) * dp.TIME_SERIES_LENGTH
