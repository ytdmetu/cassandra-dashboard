import datetime
import pickle
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ..utils import get_asset_filepath


# LSTM Model
class ForecastLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden_state=None):
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, hidden_state)

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out, (hn.detach(), cn.detach())


# Inference
class Forecaster:
    def __init__(self, preprocessor, model, look_back):
        self.preprocessor = preprocessor
        self.model = model
        self.look_back = look_back

    def __call__(self, df: pd.DataFrame, xnew: List[datetime.datetime]):
        n = len(xnew)
        x = df.iloc[-self.look_back :].price.values
        # x.shape: (N, )
        self.model.eval()
        output = []
        for i in range(n):
            xnew = self._predict_one(x)
            output.append(xnew)
            x = np.concatenate([x[1:], [xnew]])
        return output

    def _predict_one(self, x):
        # x.shape: (N,)
        x = self.preprocessor.transform(x.reshape(-1, 1))
        with torch.no_grad():
            # xnew.shape: (1, 1)
            xnew, _ = self.model(torch.from_numpy(x[None, :, :]).type(torch.float))
        return self.preprocessor.inverse_transform(xnew.detach().numpy()).item()


def build_forecaster(config, preprocessor_filepath, model_filepath, look_back):
    preprocessor = pickle.load(open(preprocessor_filepath, "rb"))
    model = ForecastLSTM(**config["model"])
    model.load_state_dict(torch.load(model_filepath))
    forecaster = Forecaster(preprocessor, model, look_back)
    return forecaster


lstm_forecaster_config = dict(
    data=dict(
        look_back=60,
    ),
    model=dict(
        input_dim=1,
        hidden_dim=32,
        num_layers=2,
        output_dim=1,
    ),
    inference=dict(),
)

lstm_forecaster = build_forecaster(
    lstm_forecaster_config,
    get_asset_filepath("univariate-lstm/preprocessor.pkl"),
    get_asset_filepath("univariate-lstm/model.pth"),
    look_back=lstm_forecaster_config["data"]["look_back"],
)


def lstm_forecast(df, xnew):
    return lstm_forecaster(df, xnew)
