import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from dateutil.relativedelta import relativedelta
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from sklearn.model_selection import ParameterGrid
import pickle as pk
import xgboost
import sys

model_config_gblinear = {
    'random_state': [0],
    'objective': ['reg:squarederror'], 
    'booster': ['gblinear'],
    'updater': ['coord_descent'],
    'n_estimators': [5000, 8000],
    'eta': [0.002], 
    'reg_alpha': [0], 
    'reg_lambda': [1], 
    'nthread': [1],
    'feature_selector': ['thrifty'],
    }

model_config_gbtree = {
    'random_state': [0],
    'booster': ['gbtree'],
    'n_estimators': [500, 300],
    'eta': [0.002], 
    'reg_alpha': [0], 
    'reg_lambda': [1], 
    'nthread': [1],
    }


def create_features(df: pd.DataFrame, frequency) -> pd.DataFrame:
    """
    Extracting date-related features from data
    """
    df = df.copy()
    dates = pd.Series(df.index)
    df['year'] = list(dates.dt.year)
    df['quarter'] = list(dates.dt.quarter)
    df['month'] = list(dates.dt.month)
    lags = 0
    if frequency == 'D':
        df['dayofweek'] = list(dates.dt.dayofweek)
        df['isweekend'] = [0 if x <= 4 else 1 for x in df['dayofweek']]
        df['dayofmonth'] = list(dates.dt.day)
        df['dayofyear'] = list(dates.dt.dayofyear)
        df['weekofyear'] = list(dates.dt.isocalendar().week)
        lags = 30
    elif frequency == 'M':
        lags = 12

    for lag in list(range(1, lags + 1)):
        df['lag' + str(lag)] = df['sale'].shift(lag)
    features_names = list(df.columns)
    features_names.remove('sale')
    # df = df.dropna(subset=features_names)
    df = df.fillna(0)
    return df

def convertListToSaleDataFrame(list: list, frequency: str) -> pd.DataFrame:
        dates = []
        sales = []
        for i in list:
            dates.append(i.get("date"))
            sales.append(i.get("sale"))
        df_comp = pd.DataFrame({'date': dates, 'sale': sales})
        df_comp.date = pd.to_datetime(df_comp.date, infer_datetime_format=True)
        df_comp.set_index("date", inplace=True)
        df_comp = df_comp.asfreq(frequency)
        return df_comp

def make_future_forecast(regr, data, to_be_removed, horizon, frequency, n_components):
    """
    Forecast for future period

    Forecasting for a horizon (interval) in the future

    Parameters
    ----------
    regr : object
        Model
    data : 
        Time series with lags after creating new features from data
        but not applied PCA yet

    Returns
    -------
    forecast_df[-horizon:] : array-like 
        Predicted df with  index "date". 
    """
    max_lags = max([int(x.split('lag')[-1]) for x in list(data.columns) if 'lag' in x], default=0)
    forecast_df = data[-max_lags:]['sale']
    
    for h in range(horizon):
        if frequency == 'D':
            new_date = forecast_df.index[-1] + relativedelta(days=1)
        elif frequency == 'M':
            new_date = forecast_df.index[-1] + relativedelta(months=1)
            
        forecast_df = forecast_df.reset_index()[['date', 'sale']]
        forecast_df = forecast_df.append({'date': new_date, 'sale': np.nan}, ignore_index=True).set_index('date')

        forecast = create_features(forecast_df[-max_lags-1:], frequency) # -1 because the index starts from 0
        
        X_forecast = forecast.drop(columns=to_be_removed)
        
        
        # PCA transform on new future data
        pca = pk.load(open("pca.pkl",'rb')) 
        
        X_forecast_temp = pca.transform(X_forecast)
        # TODO: remove columns name -> cause error, find the way to create these column auto
        
        #ts_X = pd.DataFrame(X_pca, columns=['comp_' + str(x) for x in list(range(X_pca.shape[1]))])
        X_forecast_pca = pd.DataFrame(X_forecast_temp, columns=['comp_' + str(x) for x in list(range(X_forecast_temp.shape[1]))])
        # end TODO

        # TODO: load model and make prediction here
        y_forecast = regr.predict(X_forecast_pca) 
        forecast_df.iloc[-1, 0] = y_forecast[0]
    return forecast_df[-horizon:]

def pca_transfomation(ts, val_size, test_size, normalization, frequency):
    """
    Using PCA to get the most principle info
    PCA works better on standardized data (in this work, we dont use this param)
    
    Parameters:
        ts: a dataframe
        val_size:
        test_size:
        normalization (boolean): True with normalization 
    """
    X = ts.drop(columns=['sale'])
    # 1.1 Transform full dimensional frame into new representation to find the cutoff point
    # 
    if normalization:
        pca = PCA() # if n_components is not set all components are kept
        pipeline = make_pipeline(StandardScaler(), pca)
        pipeline.fit(X[: -val_size-test_size])
        X_pca_full = pipeline.transform(X)

        # calculate cumsum explained variables of all components 
        # this cumsum will provide us evidence to select the optimal cutoff or n_components
        variances = np.cumsum(pipeline.named_steps['pca'].explained_variance_ratio_)
    else:
        pca = PCA()
        pca.fit(X[: -val_size-test_size])
        X_pca_full = pca.transform(X)  # because the n_components not yet set, so that all features are kept
        variances = np.cumsum(pca.explained_variance_ratio_)
    

    # 2. Using variances to find the best cutoff value
    # by using KneeLocator to auto select a number of PCAs based on their variances
    n_components = find_pca_cutoff(X_pca_full, variances=variances)
    
    # 3. Fit PCA on train and save model for future usage
    pca_dim_reduce = PCA(n_components=n_components)
    pca_dim_reduce.fit(X=X)
    pk.dump(pca_dim_reduce, open("pca.pkl","wb")) 


    # 4. Transform X with the PCA to reduce dimensions
    # and create a df ready
    X_pca = pca_dim_reduce.transform(X=X)
    ts_X = pd.DataFrame(X_pca, columns=['comp_' + str(x) for x in list(range(X_pca.shape[1]))])
    ts = pd.concat([ts['sale'].reset_index(), ts_X], axis=1).set_index('date')
    ts.index.freq = frequency
    ts.head()
    return ts

# def read_time_series(data, frequency: str) -> tuple:

#     # Read historical data
#     dates= [x['datetime'] for x in preprocessed_data['time_series']]
#     values = [x['value'] for x in preprocessed_data['time_series']]
#     # Create a dataframe of features and target
#     time_series = np.vstack((dates, values)) # merge 2 lists by putting on stack 2xN
#     time_series = time_series.T # transpose to dataframe of feature x sales

#     # convert array Nx2 to dataframe Nx2
#     time_series = pd.DataFrame(time_series, columns=['date', 'sale'])
#     time_series['date'] = time_series['date'].apply(pd.to_datetime)
#     time_series['sale'] = time_series['sale'].astype(float)

#     # set datetime feature as index colum, hence we obtain Nx1 
#     time_series = time_series.set_index('date')
#     time_series.index.freq = frequency
#     return time_series

def find_pca_cutoff(ts_pca: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """
    Identify a cutoff point for PCA. 
    
    If the number of features >=3, we will reduce features dimension. 
    Otherwise, we keep them as is.
    
    """
    if len(variances) >= 3:
        values = variances[1:]
        cutoff = KneeLocator(list(range(len(values))), values, curve="concave", direction='increasing').knee
    else:
        cutoff = 1
    if cutoff is None or cutoff == 0:
        cutoff = 1
    return cutoff + 1 # plus 1 because the index starts from 0
def fit_full_ts(X, y, model_config):
    """
    Fit model on the entire datasets

    Parameters
    ----------
    X : array-like of shape n_samples x n_features
        Features 
    y : array-like of shape n_sanples x1
    model_config : dict
        Hyperparameters
    Returns
    -------
    regs : object

    X_fitted : array-like
        Predicted value on the entire historical data
    """

    regr = xgboost.XGBRegressor(**model_config)
    regr.fit(X,y)
    X_fitted = regr.predict(X)
    return regr, X_fitted

def train_test_split(time_series: pd.DataFrame, frequency: str):
    """
    Split original time series into 3 sets <train, val, test> 
    DAILY_TRAIN_SIZE = 0.7 * time_series_size
    DAILY_VAL_SIZE = 0.15 * time_series_size
    DAILY_TEST_SIZE = 0.15 * time_series_size

    Args:
        time series: pd.DataFrame
    Return:
        val_size, test_size
    """
    DAILY_TRAIN_SIZE = 0.7
    DAILY_VAL_SIZE = 0.15

    MONTHLY_TRAIN_SIZE = 0.7
    MONTHLY_VAL_SIZE = 0.15

    ts_size = len(time_series)
    if frequency == 'D':
        train_size = int(np.ceil(ts_size * DAILY_TRAIN_SIZE))
        val_size = int(np.ceil(ts_size * DAILY_VAL_SIZE))
    elif frequency == 'M':
        train_size = int(np.ceil(ts_size * MONTHLY_TRAIN_SIZE))
        val_size = int(np.ceil(ts_size * MONTHLY_VAL_SIZE))
    
    test_size = len(time_series) - train_size - val_size
    return val_size, test_size

def fit_full_trainval(X, y, test_size, model_config):
    X_trainval, y_trainval = X[:-test_size], y[:-test_size]
    X_test, y_test = X[-test_size:], y[-test_size:]
    regr = xgboost.XGBRegressor(**model_config)
    results = regr.fit(
        X_trainval,
        y_trainval,
    )
    y_trainval_fitted = regr.predict(X_trainval)
    y_test_forecast = regr.predict(X_test)
    metric = cal_performance(y_test_forecast, y_test, metric_for_set='test')
    return regr, y_trainval_fitted, y_test_forecast, metric

def cal_performance(y_true, y_predict, metric_for_set) -> float:
    return np.mean(np.abs(y_predict-y_true)) 

def xgboost_approach(
                data,
                horizon, 
                frequency 
                ):

    list_model_config = list(ParameterGrid(model_config_gbtree)) + list(ParameterGrid(model_config_gblinear))

    best_model_config = {}

    min_mae = sys.maxsize
    for model_config in list_model_config:

      time_series = convertListToSaleDataFrame(data, frequency)

      ts= create_features(df=time_series, frequency=frequency)

      val_size, test_size = train_test_split(ts, frequency=frequency)

      ts_pca = pca_transfomation(ts, val_size, test_size, normalization=False, frequency=frequency)

      X, y = ts_pca.drop(columns=['sale']), ts_pca['sale']

      features_to_removed = ['sale']

      # model_config['best_iteration'] = regr.best_iteration + 1

      reg, y_trainval_fitted, y_test_forecast, test_metric = fit_full_trainval(
      X=X,
      y=y,
      test_size=test_size,
      model_config=model_config,
      )

      if (min_mae > test_metric):
        best_model_config = model_config
        min_mae = test_metric

    regr, X_fitted = fit_full_ts(
        X=X,
        y=y,
        model_config=best_model_config,
    )

    print(min_mae)


    future_forecasting = make_future_forecast(
                            regr=regr, 
                            data=ts, 
                            to_be_removed=features_to_removed, 
                            horizon=horizon,
                            frequency=frequency, 
                            n_components= ts_pca.shape[1]-1)
    return future_forecasting