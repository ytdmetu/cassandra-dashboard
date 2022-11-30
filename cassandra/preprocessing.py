import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def get_numerical_cols(dataf):
    return dataf.select_dtypes("number").columns.tolist()


def get_ordinal_cols(dataf):
    return [
        col
        for col in dataf.select_dtypes("category").columns
        if dataf[col].dtypes.ordered
    ]


def get_nominal_cols(dataf):
    return [
        col
        for col in dataf.select_dtypes("category").columns
        if not dataf[col].dtypes.ordered
    ]


def make_preprocessor(x_train: pd.DataFrame):
    numerical_cols = get_numerical_cols(x_train)

    num_transformer = Pipeline(
        [
            ("scaler", StandardScaler()),
        ]
    )

    ordinal_cols = sorted(get_ordinal_cols(x_train))
    ordinal_category_list = [
        dt.categories.tolist() for dt in x_train[ordinal_cols].dtypes
    ]
    ordinal_transformer = Pipeline(
        [
            (
                "encoder",
                OrdinalEncoder(
                    categories=ordinal_category_list,
                    handle_unknown="use_encoded_value",
                    unknown_value=np.nan,
                ),
            ),
        ]
    )

    nominal_cols = sorted(get_nominal_cols(x_train))
    nominal_transformer = Pipeline(
        [
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        ("numerical", num_transformer, numerical_cols),
                        ("ordinal", ordinal_transformer, ordinal_cols),
                        ("nominal", nominal_transformer, nominal_cols),
                    ],
                    remainder="drop",
                ),
            )
        ]
    ).fit(x_train)

    if nominal_cols:
        nominal_enc_cols = (
            preprocessor.named_steps["preprocess"]
            .transformers_[2][1]
            .named_steps["encoder"]
            .get_feature_names_out(nominal_cols)
            .tolist()
        )
    else:
        nominal_enc_cols = []

    preprocessor.feature_names_out_ = numerical_cols + ordinal_cols + nominal_enc_cols
    return preprocessor


def make_target_preprocessor(y_train):
    return StandardScaler().fit(y_train.reshape(-1, 1))
