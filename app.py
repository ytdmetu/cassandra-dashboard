import datetime
import logging
import os
import sys
import json
import requests
from dash import Dash, Input, Output, dcc, html
from enum import Enum
from config import Config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

DEBUG = os.environ.get("DASH_DEBUG_MODE") == "True"

logging.basicConfig(
    format="%(asctime)s|%(levelname)s|%(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)
logger = logging.getLogger()


TIMEZONE = datetime.timezone.utc
FORECAST_INPUT_START_OFFSET = 30

# taken from https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964

class ForecastStrategy(str, Enum):
    gaussian = "gaussian"
    naive_forecast = "naive_forecast"
    random_walk = "random_walk"
    univariate_lstm = "univariate_lstm"
    multivariate_datetime = "multivariate_datetime"


def seed_everything(seed: int):
    import os
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


seed_everything(42)

# App
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = Dash("Cassandra", external_stylesheets=external_stylesheets)
app.title = "Cassandra Dashboard"
server = app.server


app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Dropdown(
            id="stock-dropdown",
            options=[
                {"label": "Meta", "value": "META"},
                {"label": "Tesla", "value": "TSLA"},
                {"label": "Apple", "value": "AAPL"},
            ],
            value="META",
        ),
        dcc.DatePickerRange(
            id="history-date-range",
            min_date_allowed=datetime.date(2021, 1, 1),
            max_date_allowed=datetime.date.today(),
            start_date=(datetime.date.today() - datetime.timedelta(days=30)),
            end_date=datetime.date.today(),
            display_format="Y-M-D",
        ),
        dcc.Dropdown(id="forecast-strategy", options=[]),
        dcc.Graph(id="stock-price-graph"),
    ],
    style={"width": "500"},
)


@app.callback(Output("forecast-strategy", "options"), [Input("url", "pathname")])
def update_strategy_dropdown_options(pathname):
    options = [
        {
            "label": "Multivariate Datetime - Inception",
            "value": ForecastStrategy.multivariate_datetime,
        },
        {
            "label": "Univariate - LSTM",
            "value": ForecastStrategy.univariate_lstm,
        },
        {"label": "Naive Forecast", "value": ForecastStrategy.naive_forecast},
        {"label": "Random walk", "value": ForecastStrategy.random_walk},
        {"label": "Gaussian", "value": ForecastStrategy.gaussian},
    ]
    if "dev" in pathname:
        return options
    else:
        return options[:1]


@app.callback(
    Output("forecast-strategy", "value"), [Input("forecast-strategy", "options")]
)
def update_strategy_dropdown_value(options):
    return options[0]["value"]


@app.callback(
    Output("stock-price-graph", "figure"),
    [
        Input("stock-dropdown", "value"),
        Input("history-date-range", "start_date"),
        Input("history-date-range", "end_date"),
        Input("forecast-strategy", "value"),
    ],
)
def update_graph(stock_id, start_date, end_date, forecast_strategy):
    url = f"{Config.BASE_URL}/forecast"
    payload = json.dumps({
        "stock": stock_id,
        "start_date": start_date,
        "end_date": end_date,
        "interval": "1h",
        "n_forecast": 100,
        "strategy": forecast_strategy
    })
    
    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code != 200:
        print(response.content)
        raise ValueError(f"Received {response.status_code} from API")

    return dict(
        data= response.json(),
        layout=dict(
            margin={"l": 40, "r": 0, "t": 20, "b": 30},
            legend=dict(font=dict(size=14)),
        ),
    )


if __name__ == "__main__":
    app.run_server(debug=True)
