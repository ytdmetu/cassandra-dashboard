import datetime
import logging
import os
from enum import Enum
from functools import lru_cache
from random import random

import numpy as np
import yfinance as yf
from dash import Dash, Input, Output, dcc, html

DEBUG = os.environ.get("DASH_DEBUG_MODE") == "True"

logging.basicConfig(
    format="%(asctime)s|%(levelname)s|%(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger()


TIMEZONE = datetime.timezone.utc
HISTORY_DAYS = 7
FORECAST_INPUT_START_OFFSET = 3

# taken from https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964


def seed_everything(seed: int):
    import os
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


seed_everything(42)

# Stock price history
@lru_cache(maxsize=10)
def fetch_stock_price(stock_id, start, end, interval="1h"):
    logger.info(
        f"Fetching historical stock price data for {stock_id} between {start} and {end}"
    )
    return (
        yf.Ticker(stock_id)
        .history(start=start, end=end, interval=interval)
        .tz_convert(TIMEZONE)
    )


# Forecast


class ForecastStrategy(Enum):
    gaussian = "gaussian"
    random_walk = "random_walk"


def forecast(stock_id, df, strategy=ForecastStrategy.random_walk):
    start_time = df.index[-1].to_pydatetime()
    n_forecast = 12
    x = [start_time + datetime.timedelta(hours=i) for i in range(1, 1 + n_forecast)]
    if strategy == ForecastStrategy.gaussian.value:
        y = gaussian_noise(df.Close.values).tolist()
    elif strategy == ForecastStrategy.random_walk.value:
        y = random_walk(df.Close.values)
    else:
        raise ValueError(strategy)
    return dict(x=x, y=y)


def gaussian_noise(x, n=12):
    return x.mean() + x.std() * 10 * np.random.rand(n)


def random_walk(x, n=12):
    initial_value = x[-1]
    delta = x.std()
    result = [initial_value]
    for i in range(1, n):
        movement = -delta if random() < 0.5 else delta
        value = result[i - 1] + movement
        result.append(value)
    return result[1:]


# App
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = Dash("Cassandra", external_stylesheets=external_stylesheets)
server = app.server


app.layout = html.Div(
    [
        dcc.Dropdown(
            id="stock-dropdown",
            options=[
                {"label": "Meta", "value": "META"},
                {"label": "Tesla", "value": "TSLA"},
                {"label": "Apple", "value": "AAPL"},
            ],
            value="META",
        ),
        dcc.Dropdown(
            id="forecast-strategy",
            options=[
                {"label": "Random walk", "value": ForecastStrategy.random_walk.value},
                {"label": "Gaussian", "value": ForecastStrategy.gaussian.value},
            ],
            value=ForecastStrategy.random_walk.value,
        ),
        dcc.Graph(id="stock-price-graph"),
    ],
    style={"width": "500"},
)


@app.callback(
    Output("stock-price-graph", "figure"),
    [Input("stock-dropdown", "value"), Input("forecast-strategy", "value")],
)
def update_graph(stock_id, forecast_strategy):
    # stock price history
    end = datetime.datetime.now(tz=TIMEZONE)
    start = end - datetime.timedelta(days=HISTORY_DAYS)
    df = fetch_stock_price(stock_id, start.date().isoformat(), end.date().isoformat())
    # forecast
    input_df = df[end - datetime.timedelta(days=FORECAST_INPUT_START_OFFSET) :]
    forecast_data = forecast(stock_id, input_df, forecast_strategy)
    # representation
    history_data = {"x": df.index.tolist(), "y": df.Close.tolist(), "name": "History"}
    forecast_data["name"] = "Forecast"
    forecast_data["x"].insert(0, history_data["x"][-1])
    forecast_data["y"].insert(0, history_data["y"][-1])
    return dict(
        data=[history_data, forecast_data],
        layout=dict(
            margin={"l": 40, "r": 0, "t": 20, "b": 30},
            legend=dict(font=dict(size=14)),
        ),
    )


if __name__ == "__main__":
    app.run_server(debug=True)
