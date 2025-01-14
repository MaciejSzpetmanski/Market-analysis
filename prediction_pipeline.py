import pandas as pd
import data_pipeline as dp
import yfinance as yf
from datetime import datetime, timedelta
import joblib

#%% fetch data

def fetch_data(name, n_days):
    extended_days = n_days * 2
    end_date = datetime.now()
    start_date = end_date - timedelta(days=extended_days)
    data = yf.download(name, start=start_date, end=end_date, interval='1d')
    data = data.tail(n_days)
    # TODO raise an error when there is not enough data
    return data

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

def prepare_data_for_prediction(df, name):
    # artificially add name column for grouping
    df["name"] = name
    dp.validate_data(df)
    
    # adding name columns
    columns = dp.load_object(dp.TRAIN_COLUMNS_PATH)
    name_columns = [col for col in columns if col.startswith("name_")]
    for col in name_columns:
        df[col] = 0
    df[f"name_{name}"] = 1
    
    df = dp.convert_column_types(df)
    df = dp.add_time_columns(df)
    
    df = df.sort_values(by=["date"])
    
    df = df.tail(dp.TIME_SERIES_LENGTH + dp.SCHEMA_WIDTH - 1).reset_index(drop=True)
    
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

def download_and_prepare_data(name):
    n_days = dp.TIME_SERIES_LENGTH + dp.SCHEMA_WIDTH - 1
    df = fetch_data(name, n_days)
    df = df.reset_index(drop=False)
    df.columns = [c[0] for c in df.columns]
    df.columns = df.columns.str.lower()
    # TODO change to actual value
    df["adjusted_close"] = df["close"]
    df = prepare_data_for_prediction(df, name)
    
    date = datetime.today().strftime('%Y-%m-%d')
    df = merge_vector_with_pred_date(df, date)
    return df

def predict_value(df, name, pred_name="close"):
    """
    Oczekuje ramki danych w formacie odpowiednim dla modelu.
    Wylicza przewidywany przyrost na podstawie modelu.
    Następnie przekształca tę wartosć, uzyskując przewidywaną cenę.
    Na koniec przeskalowuję ją do oryginalnej skali i zwraca.
    """
    
    # TODO replace with final model function
    # loading model
    model = joblib.load(f"models/individual/{name}.pkl")
    # predicting increment
    inc_prediction = model.predict(df)
    
    # getting the last known value
    
    last_column_name = [col for col in df.columns if col.startswith(pred_name)][-1]
    last_data = df[last_column_name]
    
    # counting new value
    prediction = last_data * inc_prediction + last_data
    
    # rescaling the value
    scalers = dp.load_object(dp.SCALERS_PATH)
    
    # preparing a df for inversion
    pred_data = {f"name_{name}": 1, "adjusted_close": 0, "close": prediction, "high": 0, "low": 0, "open": 0, "volume": 0}
    pred_df = pd.DataFrame(pred_data)
    
    original_scale_values = dp.inverse_target_scaling(pred_df, name, scalers)
    y_pred = original_scale_values[:, 1]
    return y_pred
    
def main():
    path = "data/^DJI_data_1d.csv" # example
    name = "^DJI" # example
    date = "2024-09-30" # example
    df = dp.load_data_from_file(path)
    x = prepare_data_for_prediction(df, name)
    x = merge_vector_with_pred_date(x, date)
    
if __name__ == "__main__":
    main()
