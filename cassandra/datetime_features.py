import datetime

import holidays
import pandas as pd
from pandas.api.types import CategoricalDtype


def is_us_holiday(dt):
    return dt.strftime("%Y-%m-%d") in holidays.UnitedStates()


def extract_datetime_features(ds):
    df = pd.DataFrame()
    df.index = ds
    df["year"] = ds.year
    df["month"] = ds.month
    df["day"] = ds.day
    df["hour"] = ds.hour
    df["day_of_year"] = ds.day_of_year
    df["week_of_year"] = ds.weekofyear
    df["month_name"] = ds.month_name()
    df["day_name"] = ds.day_name()
    df["is_weekend"] = (ds.day_of_week == 5) | (ds.day_of_week == 6)
    df["is_month_start"] = ds.is_month_start
    df["is_quarter_start"] = ds.is_quarter_start
    df["is_month_end"] = ds.is_month_end
    df["is_year_start"] = ds.is_year_start
    # US holidays
    df["is_holiday"] = pd.Series(ds.values).apply(is_us_holiday).values
    df["is_day_before_holiday"] = (
        pd.Series(ds + datetime.timedelta(days=1)).map(is_us_holiday).values
    )
    df["is_day_after_holiday"] = (
        pd.Series(ds - datetime.timedelta(days=1)).map(is_us_holiday).values
    )
    return df


def add_datetime_features(df):
    return pd.concat([extract_datetime_features(df.index), df], axis=1)


ORDINALS_INFO = []
ORDINALS = [feat for feat, _ in ORDINALS_INFO]

NOMINALS = [
    "hour",
    "month_name",
    "day_name",
    "is_weekend",
    "is_month_start",
    "is_quarter_start",
    "is_month_end",
    "is_year_start",
    "is_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
]

NUMERICALS = ["day_of_year", "week_of_year", "price"]

UNUSED = [ ]

TARGET_VAR = "price"


def set_col_dtypes(dataf):
    dataf = dataf.drop(columns=UNUSED, errors="ignore")

    for col in NUMERICALS:
        if col not in dataf.columns:
            continue
        dataf[col] = dataf[col].astype("float")

    for col, categories in ORDINALS_INFO:
        if col not in dataf.columns:
            continue
        dataf[col] = dataf[col].astype(
            CategoricalDtype(categories=categories, ordered=True)
        )

    for col in NOMINALS:
        if col not in dataf.columns:
            continue
        dataf[col] = dataf[col].astype("category")

    existing_cols = set(dataf.columns)
    col_order = [
        col for col in NUMERICALS + ORDINALS + NOMINALS if col in existing_cols
    ]
    return dataf[col_order]


def prepare_dataset(df):
    return (
        pd.DataFrame(index=df.index, data=dict(price=df.Close.values))
        .pipe(add_datetime_features)
        .pipe(set_col_dtypes)
    )
