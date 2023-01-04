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
    multivariate_diff = "multivariate_diff"
    price_nlp_model = "price_nlp_model"


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
        {"label": "Baseline - Naive Forecast", "value": ForecastStrategy.naive_forecast},
        {"label": "Baseline - Random walk", "value": ForecastStrategy.random_walk},
        {"label": "Baseline - Gaussian", "value": ForecastStrategy.gaussian},
        {
            "label": "NLP Sentiment Price Change - LSTM",
            "value": ForecastStrategy.price_nlp_model,
        },
        {
            "label": "Multivariate Price Change - LSTM",
            "value": ForecastStrategy.multivariate_diff,
        },
    ]
    if "live" in pathname:
        return [options[1]]
    else:
        return options


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
    # if day difference is less than 14, use 14
    if (
        datetime.datetime.strptime(end_date, "%Y-%m-%d")
        - datetime.datetime.strptime(start_date, "%Y-%m-%d")
    ).days < FORECAST_INPUT_START_OFFSET:
        start_date = (
            datetime.datetime.strptime(end_date, "%Y-%m-%d")
            - datetime.timedelta(days=FORECAST_INPUT_START_OFFSET)
        ).strftime("%Y-%m-%d")
    payload = json.dumps(
        {
            "stock": stock_id,
            "start_date": start_date,
            "end_date": end_date,
            "interval": "1h",
            "n_forecast": 12,
            "strategy": forecast_strategy,
        }
    )

    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload, auth = requests.auth.HTTPBasicAuth(Config.API_USERNAME, Config.API_PASSWORD))
    if response.status_code != 200:
        print(response.content)
        raise ValueError(f"Received {response.status_code} from API")

    return dict(
        data=response.json(),
        layout=dict(
            margin={"l": 80, "r": 80, "t": 50, "b": 50},
            yaxis=dict(title="Stock Price (USD)"),
            xaxis=dict(title="Date"),
            title=f"Stock Price for {stock_id}",
            legend=dict(font=dict(size=14)),
        ),
    )


if __name__ == "__main__":
    app.run_server(debug=True)
