import pytest
import pandas as pd
import numpy as np
import data_pipeline as dp
from pandas.api.types import is_numeric_dtype, is_string_dtype

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
        
        
        
