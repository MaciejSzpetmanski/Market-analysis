import pandas as pd
import data_pipeline as dp
import yfinance as yf
from datetime import datetime, timedelta
import joblib
from tensorflow.keras.models import load_model as load_nn_model
import warnings
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
import numpy as np

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

def prepare_data_for_prediction(df, name, use_arima=True):
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
    
    if len(df) < dp.TIME_SERIES_LENGTH:
        raise Exception(f"Not enough data. {dp.TIME_SERIES_LENGTH} required.")
    df = df.tail(dp.TIME_SERIES_LENGTH + dp.SCHEMA_WIDTH - 1).reset_index(drop=True)
    
    # TODO
    print("Wyliczanie wykładnika Hursta")
    df = dp.add_hurst_dim_columns(df, dp.GROUP_BY_COLUMN, dp.HURST_WIDTH)
    
    print("Standaryzacja kolumn")
    scalers = dp.load_object(dp.SCALERS_PATH)
    df.volume = df.volume.astype(float)
    df = dp.standardize_columns(df, scalers, dp.COLUMNS_TO_STANDARDIZE, dp.GROUP_BY_COLUMN)
    
    # TODO add arima
    if use_arima:
        print("Wykorzystanie ARIMA")
        df = dp.add_arima_prediction(df, dp.GROUP_BY_COLUMN, width=dp.SCHEMA_WIDTH)
    
    # TODO technical analysis indicators
    print("Dodawanie wskaźników analizy techniczej")
    df = dp.add_technical_analysis_columns(df, dp.GROUP_BY_COLUMN, prefix=dp.TECHNICAL_ANALYSIS_PREFIX)
    # TODO scaling
    print("Standaryzacja kolumn analizy technicznej")
    technical_analysis_columns = [col for col in df.columns if col.startswith(dp.TECHNICAL_ANALYSIS_PREFIX)]
    technical_analysis_scalers = dp.load_object(dp.TECHNICAL_SCALERS_PATH)
    df = dp.standardize_columns(df, technical_analysis_scalers, technical_analysis_columns, dp.GROUP_BY_COLUMN)
    
    
    
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
    
    # TODO remove year columns
    year_columns = ["date_year"] + [f"date_year_{i}" for i in range(1, dp.TIME_SERIES_LENGTH)]
    columns_to_drop = ["name"] + year_columns
    print("Usuwanie kolumny z nazwą")
    output_vector = output_vector.drop(columns=columns_to_drop)
    return output_vector

def merge_vector_with_pred_date(x_vector, date, remove_year=True):
    print("Dołączanie czasu")
    y_date = create_prediction_date_vector(date)
    result = pd.concat([x_vector, y_date], axis=1)
    result = result.drop(columns=["date_year_y"])
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

def increment_to_close(df, inc, pred_name="close"):
    # getting the last known value
    last_column_name = [col for col in df.columns if col.startswith(pred_name)][-1]
    last_data = df[last_column_name]
    # counting new value
    y = last_data * inc + last_data
    return y

#%%

def predict_value_old(df, name, pred_name="close"):
    """
    Oczekuje ramki danych w formacie odpowiednim dla modelu.
    Przetwarzanie nie uwzględnia cech ARIMA.
    Wylicza przewidywany przyrost na podstawie indywidualnego modelu.
    Następnie przekształca tę wartosć, uzyskując przewidywaną cenę.
    Na koniec przeskalowuję ją do oryginalnej skali i zwraca.
    """
    
    model = joblib.load(f"models/old_individual/{name}.pkl")
    # predicting increment
    inc_prediction = model.predict(df)
    prediction = increment_to_close(df, inc_prediction, pred_name)
    # rescaling the value
    scalers = dp.load_object(dp.SCALERS_PATH)
    # preparing df for inversion
    pred_data = {f"name_{name}": 1, "adjusted_close": 0, "close": prediction, "high": 0, "low": 0, "open": 0, "volume": 0}
    pred_df = pd.DataFrame(pred_data)
    
    original_scale_values = dp.inverse_target_scaling(pred_df, name, scalers)
    y_pred = original_scale_values[:, 1]
    return y_pred

def prepare_data_for_arima(df, name, pred_name="close"):
    if len(df) < 20:
        raise Exception("Not enough data. 20 required.")
    df["name"] = name
    dp.validate_data(df)
    df = dp.convert_column_types(df)
    df = df.sort_values(by=["date"])
    # df = df.tail(20).reset_index(drop=True)
    df = df.reset_index(drop=True)
    x = df[pred_name]
    return x

def prepare_data(models, are_on_close, df, name, date, use_arima=True):
    model = models[name]
    if model is None: # ARIMA
        df = prepare_data_for_arima(df, name)
    else:
        df = prepare_data_for_prediction(df, name)
        df = merge_vector_with_pred_date(df, date)
    return df

def arima_predict(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auto_model = auto_arima(x, seasonal=False, stepwise=True, trace=False)
        model = ARIMA(x, order=auto_model.order)
        model_fit = model.fit()
    pred = model_fit.forecast(steps=1)
    return np.array(pred)

def predict_value(models, are_on_close, df, name, pred_name="close"):
    model = models[name]
    is_on_close = are_on_close[name]
    if model == None:
        prediction = arima_predict(df)
        return prediction
    prediction = model.predict(df)
    if len(prediction) > 1:
        prediction = prediction[-1]
    prediction = prediction.reshape(-1)
    if not is_on_close:
        prediction = increment_to_close(df, prediction, pred_name)
    scalers = dp.load_object(dp.SCALERS_PATH)
    # preparing df for inversion
    pred_data = {f"name_{name}": 1, "adjusted_close": 0, "close": prediction, "high": 0, "low": 0, "open": 0, "volume": 0}
    pred_df = pd.DataFrame(pred_data)
    
    original_scale_values = dp.inverse_target_scaling(pred_df, name, scalers)
    y_pred = original_scale_values[:, 1]
    
    return y_pred
    
def load_individual_model(name):
    return joblib.load(f"models/individual/{name}.pkl")


def load_models():
    class TransformerModel(nn.Module):
        def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward):
            super(TransformerModel, self).__init__()
            self.encoder = nn.Linear(input_dim, d_model)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
                num_layers=num_layers
            )
            self.dropout = nn.Dropout(p=0.2)
            self.decoder = nn.Linear(d_model, 1)
            self.batch_norm_1 = nn.BatchNorm1d(d_model)
            self.batch_norm_2 = nn.BatchNorm1d(d_model)

        def forward(self, x):
            x = self.encoder(x)
            x = self.batch_norm_1(x)
            x = nn.ReLU()(x)
            x = x.squeeze(0)
            x = self.transformer(x)
            x = self.batch_norm_2(x)
            x = self.dropout(x)
            out = self.decoder(x)
            return out
    
    class TransformerContainer:
        def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, model_path):
            # self.model = TransformerModel(input_dim=input_dim, d_model=d_model, nhead=nhead,
            #                               num_layers=num_layers, dim_feedforward=dim_feedforward)
            # self.model.load_state_dict(torch.load(model_path))
            # self.model.eval()
            
            
            self.model = TransformerModel(input_dim=465, d_model=64, nhead=8, num_layers=4, dim_feedforward=128)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        
        def predict(self, x):
            x = torch.tensor(x.values, dtype=torch.float32)
            # duplicating vector (2 needed for prediction)
            x = torch.cat([x, x], dim=0)
            y_pred = self.model(x).detach().numpy()
            return y_pred
    
    LINEAR_REGRESSION_PATH = "models/elasticnet_model.pkl"
    RANDOM_FOREST_PATH = "models/random_forest_model.pkl"
    XGBOOST_PATH = "models/xgboost_model.pkl"
    TRANSFORMER_PATH = "models/transformers/best_model.pth"
    NEURAL_NETWORK_PATH = "models/nn.keras"
    
    lr = joblib.load(LINEAR_REGRESSION_PATH)
    rf = joblib.load(RANDOM_FOREST_PATH)
    xgb = joblib.load(XGBOOST_PATH)
    tr = TransformerContainer(465, 64, 8, 4, 128, TRANSFORMER_PATH)
    nnetwork = load_nn_model(NEURAL_NETWORK_PATH)
    arima = None
    
    models = {
        '000001.SS': lr,
         'AAPL': tr,
         'AUDUSD=X': lr,
         'BA': rf,
         'BTC-USD': lr,
         'DE': load_individual_model('DE'),
         'ETH-USD': xgb,
         'EURUSD=X': lr,
         'GBPUSD=X': lr,
         'KO': arima,
         'LMT': tr,
         'MSFT': xgb,
         'NZDUSD=X': lr,
         'PFE': lr,
         'TSLA': arima,
         'USDCAD=X': lr,
         'USDCHF=X': lr,
         'USDJPY=X': load_individual_model('USDJPY=X'),
         'WMT': load_individual_model('WMT'),
         'XOM': nnetwork,
         '^DJI': tr,
         '^FCHI': xgb,
         '^FTSE': arima,
         '^GDAXI': tr,
         '^GSPC': load_individual_model('^GSPC'),
         '^HSI': arima,
         '^IXIC': load_individual_model('^IXIC'),
         '^N225': load_individual_model('^N225'),
         '^RUT': rf
    }
    
    are_on_close = {
        '000001.SS': True,
         'AAPL': False,
         'AUDUSD=X': True,
         'BA': True,
         'BTC-USD': True,
         'DE': False,
         'ETH-USD': True,
         'EURUSD=X': True,
         'GBPUSD=X': True,
         'KO': True,
         'LMT': False,
         'MSFT': True,
         'NZDUSD=X': True,
         'PFE': True,
         'TSLA': True,
         'USDCAD=X': True,
         'USDCHF=X': True,
         'USDJPY=X': False,
         'WMT': False,
         'XOM': True,
         '^DJI': False,
         '^FCHI': True,
         '^FTSE': True,
         '^GDAXI': False,
         '^GSPC': False,
         '^HSI': True,
         '^IXIC': False,
         '^N225': False,
         '^RUT': True
    }
    
    return models, are_on_close

def test():
    models, are_on_close = load_models()
    
    name = "000001.SS" # example
    name = "KO"
    name = "^HSI"
    name = "XOM"
    name = "KO"
    path = f"data/{name}_data_1d.csv" # example
    
    date = "2024-09-30" # example
    df = dp.load_data_from_file(path)
    # x = prepare_data_for_prediction(df, name)
    x = prepare_data(models, are_on_close, df, name, date, use_arima=True)
    # x = merge_vector_with_pred_date(x, date)
    # y1 = predict_value_old(x, name, pred_name="close")
    predicted_value = predict_value(models, are_on_close, x, name)[-1]
    print(predicted_value)
    
