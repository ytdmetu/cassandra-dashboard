import datetime
import logging
import os
from enum import Enum
from functools import lru_cache
from random import random

import numpy as np
import yfinance as yf
from dash import Dash, Input, Output, dcc, html

import sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from cassandra.forecast import ForecastStrategy, forecast
from cassandra.utils import get_asset_filepath

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


# App
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = Dash("Cassandra", external_stylesheets=external_stylesheets)
app.title = "Cassandra Dashboard"
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
                {"label": "Naive LSTM", "value": ForecastStrategy.naive_lstm.value},
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
    forecast_data = forecast(stock_id, input_df, 12, forecast_strategy)
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
