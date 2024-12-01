import pandas as pd
import data_pipeline as dp

#%%

def create_time_series_vector(data, columns, n):
    shifted_columns = {}
    for i in range(1, n):
        for col in columns:
            shifted_columns[col + "_" + str(i)] = data[col].shift(-i)
    res_data = pd.concat([data] + [pd.Series(value, name=key) for key, value in shifted_columns.items()], axis=1)
    # removing rows with missing values
    res_data = res_data.head(-(n-1))
    last_row = res_data.tail(1).reset_index(drop=True)
    return last_row

# TODO check weekend, free days
def validate_y_date(date, name):
    pass

def create_date_vector(date):
    df_date = pd.DataFrame()
    df_date["date"] = [date]
    df_date["date"] = pd.to_datetime(df_date["date"])
    df_date = dp.add_time_columns(df_date)
    return df_date

def create_prediction_date_vector(date):
    y_date = create_date_vector(date)
    y_date = y_date.drop(columns=["date"])
    y_date = y_date.add_suffix('_y')
    return y_date

#%%

def prepare_data_for_prediction(path, name):
    df = dp.load_data_from_file(path)
    # TODO checking input data size
    df = dp.convert_column_types(df)
    df = dp.add_time_columns(df)
    
    df = df.sort_values(by=["date"])
    
    # TODO trimming, filtering data
    df = df.tail(dp.TIME_SERIES_LENGTH + dp.SCHEMA_WIDTH - 1).reset_index(drop=True)
    
    columns = dp.load_object(dp.TRAIN_COLUMNS_PATH)
    name_columns = [col for col in columns if col.startswith("name_")]
    for col in name_columns:
        df[col] = 0
    df[f"name_{name}"] = 1
    # artificially add name column for grouping
    df["name"] = name
    
    print("Standaryzacja kolumn")
    scalers = dp.load_object(dp.SCALERS_PATH)
    df.volume = df.volume.astype(float)
    df = dp.standardize_columns(df, scalers, dp.COLUMNS_TO_STANDARDIZE, dp.GROUP_BY_COLUMN)
    
    print("Dodawanie cech fraktalnych o zmiennej długości")
    df = dp.add_fractal_long_schemas(df, dp.LONG_SCHEMA_PATH, dp.GROUP_BY_COLUMN, dp.LONG_SCHEMA_PREFIX, dp.SCHEMA_WIDTH)
    print("Dodawanie cech cykli cenowych")
    df = dp.add_cycle_columns(df, dp.GROUP_BY_COLUMN, dp.SCHEMA_WIDTH)
    
    print("Dodawanie cech średniej kroczącej")
    df = dp.add_ema_columns(df, dp.GROUP_BY_COLUMN, dp.EMA_PERIODS)
    
    print("Dodawanie cech fraktalnych o stałej długości")
    df = dp.add_fractal_short_schemas(df, dp.SHORT_SCHEMA_PATH, dp.GROUP_BY_COLUMN, dp.SHORT_SCHEMA_PREFIX)
    
    print("Wybór kolumn")
    missing_columns = [col for col in columns if col not in df.columns]
    for col in missing_columns:
        df[col] = 0
    df = df[columns] # removing date, setting order of columns
    
    print("Tworzenie szeregów czasowych")
    columns_to_shift = [col for col in df.columns if not col.startswith("name")]
    output_vector = create_time_series_vector(df, columns_to_shift, dp.TIME_SERIES_LENGTH)
    
    columns_to_drop = ["name"]
    print("Usuwanie kolumny z nazwą")
    output_vector = output_vector.drop(columns=columns_to_drop)
    return output_vector

def merge_vector_with_pred_date(x_vector, date):
    y_date = create_prediction_date_vector(date)
    result = pd.concat([x_vector, y_date], axis=1)
    return result
    
def main():
    path = "data/^DJI_data_1d.csv" # example
    name = "^DJI" # example
    date = "2024-09-30" # example
    x = prepare_data_for_prediction(path, name)
    x = merge_vector_with_pred_date(x, date)
    
if __name__ == "__main__":
    main()
    