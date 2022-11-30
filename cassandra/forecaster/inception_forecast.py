import datetime
import pickle
from typing import List

import numpy as np
import torch
import torch.nn as nn
from pandas.api.types import CategoricalDtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from tsai.all import *
from tsai.data.tabular import EarlyStoppingCallback

from ..datetime_features import TARGET_VAR, prepare_dataset
from ..timeseries_utils import make_ts_samples
from ..utils import get_asset_filepath

config = dict(
    data=dict(
        look_back=60,
    ),
)


# Inference
class Forecaster:
    def __init__(self, xpp, ypp, learn, look_back):
        self.xpp = xpp
        self.ypp = ypp
        self.learn = learn
        self.look_back = look_back

    def __call__(self, df: pd.DataFrame, xnew: List[datetime.datetime]):
        n = len(xnew)
        initial_length = len(df)
        for i in range(n):
            pred = self._predict_one(df)
            df = add_prediction_to_dataset(df, pred)
        return df.Close.values[initial_length:].tolist()

    def _predict_one(self, df):
        df = prepare_dataset(df)
        x = self.xpp.transform(df.iloc[-self.look_back :])
        target_idx = df.columns.tolist().index(TARGET_VAR)
        xb, _ = make_ts_samples(x, self.look_back, target_idx)
        xb = np.swapaxes(xb, 1, 2)
        _, _, y_pred = self.learn.get_X_preds(xb)
        return self.ypp.inverse_transform(np.array(y_pred).reshape(-1, 1)).item()


def add_prediction_to_dataset(df, price):
    next_datetime = df.index[-1] + pd.DateOffset(hours=1)
    new_df = pd.DataFrame(index=[next_datetime], data=dict(Close=[price]))
    return pd.concat([df, new_df], axis=0)


def build_forecaster(
    x_preprocessor_filepath, y_preprocessor_filepath, learn_filepath, look_back
):
    learn = load_learner(learn_filepath)
    xpp = pickle.load(open(x_preprocessor_filepath, "rb"))
    ypp = pickle.load(open(y_preprocessor_filepath, "rb"))
    forecaster = Forecaster(xpp, ypp, learn, look_back)
    return forecaster


forecaster = build_forecaster(
    get_asset_filepath("multivariate-inception-datetime/xpp.pkl"),
    get_asset_filepath("multivariate-inception-datetime/ypp.pkl"),
    get_asset_filepath("multivariate-inception-datetime/learn.pkl"),
    look_back=config["data"]["look_back"],
)


def inception_forecast(df, xnew):
    return forecaster(df, xnew)
