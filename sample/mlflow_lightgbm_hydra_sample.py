import logging
import os

import hydra
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg):
    # cfgはdictのようなオブジェクト
    logger.info(f"type(cfg): {type(cfg)}")
    logger.info(f"cfg: {cfg}")

    # hydraを利用する際はカレントディレクトリが実行ごとのディレクトリに変わってしまうので注意が必要
    logger.info(f"current directory: {os.getcwd()}")
    logger.info(
        f"original current directory: {hydra.utils.get_original_cwd()}")

    # 実験結果を同じディレクトリに格納するため、オリジナルのディレクトリを設定する
    mlflow.set_tracking_uri(
        "file:///" + hydra.utils.get_original_cwd() + "/mlruns")

    # auto log lightgbm
    mlflow.lightgbm.autolog()
    
    train_lightgbm(dict(cfg.lightgbm))


def train_lightgbm(lgbm_params):

    # download titanic data
    data = fetch_openml(data_id=40945, as_frame=True)

    # X
    df = data["data"].copy()

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
