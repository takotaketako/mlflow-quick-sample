"""
autologを利用したロギング
lightgbmのほかに、tensorflowやpytorchにも対応している
"""

import lightgbm as lgb
import pandas as pd
from mlflow import lightgbm as mlflow_lgb
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def main():

    # auto log lightgbm
    mlflow_lgb.autolog()

    # download titanic data
    data = fetch_openml(data_id=40945, as_frame=True)

    # X
    df = data["data"]
    # object to category
    col_object = list(df.select_dtypes(include=[object]).dtypes.index)
    df[col_object] = df.loc[:, col_object].astype("category")
    col_categorical = list(df.select_dtypes(include=["category"]).dtypes.index)

    # y
    target = data["target"].astype("int")

    # split data
    X_train, X_test, y_train, y_test = train_test_split(df, target,
                                                        shuffle=True,
                                                        random_state=42)

    # training
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_test, y_test, reference=lgb_train)
    lgbm_params = {
        'objective': 'binary',
        'metric': 'binary',
        'verbosity': -1,
    }
    lgb.train(lgbm_params,
              lgb_train,
              valid_sets=lgb_valid,
              num_boost_round=1000,
              early_stopping_rounds=100,
              verbose_eval=10,
              categorical_feature=col_categorical,
              )


if __name__ == "__main__":
    main()
