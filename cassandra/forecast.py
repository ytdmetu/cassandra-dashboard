import datetime
from enum import Enum

from .forecaster.baselines import gaussian_noise, naive_forecast, random_walk
from .forecaster.inception_forecast import inception_forecast
from .forecaster.lstm_forecast import lstm_forecast


class ForecastStrategy(str, Enum):
    gaussian = "gaussian"
    naive_forecast = "naive_forecast"
    random_walk = "random_walk"
    univariate_lstm = "univariate_lstm"
    multivariate_datetime = "multivariate_datetime"


def forecast(strategy, stock_id, df, n_forecast=12):
    start_time = df.index[-1].to_pydatetime()
    x = [start_time + datetime.timedelta(hours=i) for i in range(1, 1 + n_forecast)]
    if strategy == ForecastStrategy.gaussian:
        y = gaussian_noise(df, x)
    elif strategy == ForecastStrategy.random_walk:
        y = random_walk(df, x)
    elif strategy == ForecastStrategy.naive_forecast:
        y = naive_forecast(df, x)
    elif strategy == ForecastStrategy.univariate_lstm:
        y = lstm_forecast(df, x)
    elif strategy == ForecastStrategy.multivariate_datetime:
        y = inception_forecast(df, x)
    else:
        raise ValueError(strategy)
    return dict(x=x, y=y)


def forecast_past(strategy, df, stock_id, look_back=60):
    # It also indicates the number of backtesting hours
    predictions_date = []
    predictions = []
    for i in range(look_back, len(df)):
        new_df = df.iloc[i - look_back : i]
        result = forecast(strategy, stock_id, new_df, n_forecast=1)
        predictions_date.append(result["x"][0])
        predictions.append(result["y"][0])
    actual = df.iloc[look_back:].price.values.tolist()
    return {"x": predictions_date, "y": predictions, "z": actual}
