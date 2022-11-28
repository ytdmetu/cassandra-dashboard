import datetime
from enum import Enum
from random import random
import numpy as np

from .lstm_stock_price_forecast import lstm_forecast


class ForecastStrategy(str, Enum):
    gaussian = "gaussian"
    naive_forecast = "naive_forecast"
    random_walk = "random_walk"
    univariate_lstm = "univariate_lstm"


def gaussian_noise(x, n):
    return (x.mean() + x.std() * 10 * np.random.rand(n)).tolist()


def random_walk(x, n):
    initial_value = x[-1]
    delta = x.std()
    result = [initial_value]
    for i in range(1, n):
        movement = -delta if random() < 0.5 else delta
        value = result[i - 1] + movement
        result.append(value)
    return result[1:]


def naive_forecast(x, n):
    return [*x[-n:][::-1]]


def forecast(stock_id, df, n_forecast=12, strategy=ForecastStrategy.random_walk):
    start_time = df.index[-1].to_pydatetime()
    x = [start_time + datetime.timedelta(hours=i) for i in range(1, 1 + n_forecast)]
    if strategy == ForecastStrategy.gaussian:
        y = gaussian_noise(df.Close.values, n_forecast)
    elif strategy == ForecastStrategy.random_walk:
        y = random_walk(df.Close.values, n_forecast)
    elif strategy == ForecastStrategy.naive_forecast:
        y = naive_forecast(df.Close.values, n_forecast)
    elif strategy == ForecastStrategy.univariate_lstm:
        y = lstm_forecast(df.Close.values, n_forecast)
    else:
        raise ValueError(strategy)
    return dict(x=x, y=y)


def forecast_past_hours(start_date, end_date, historical_df, strategy, stock):
    new_predictions_date = []
    # It also indicates the number of backtesting hours
    past_prediction_number = historical_df.shape[0] - 1
    new_predictions = []
    for i in range(past_prediction_number):
        new_df = historical_df.iloc[: -(past_prediction_number - i)]
        strategy = strategy or ForecastStrategy.naive_lstm
        pred = forecast(stock, new_df, strategy=strategy)
        new_predictions.append(pred["y"][0])
        new_predictions_date.append(pred["x"][0])
    actual = (
        historical_df.reset_index().iloc[-past_prediction_number:][["Close"]]
    ).Close.values.tolist()
    return {"x": new_predictions_date, "y": new_predictions, "z": actual}
