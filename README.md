# mlflow-quick-sample

## 概要

MLflow(Tracking)とHydraを動作させるサンプル。MLflowは別サーバ等は立てずにローカルで動作させることを想定。

パラメータやメトリクスをロギングしてMLflowのUIで確認するまでを実行する。

## Install

```bash
python -m pip install -r requirements.txt
```

## MLflow UIの起動

1. `cd sample`
1. `mlflow ui --host 0.0.0.0 --port 5000` (UIの起動)
1. `http://localhost:5000`にアクセス

## サンプルスクリプトの実行

1. `cd sample`
1. `python mlflow_sample.py` (基本的なAPIを利用したロギング)
1. `python mlflow_lightgbm_sample.py` (autologを利用したロギング)
1. `python mlflow_lightgbm_hydra_sample.py` (hydraを利用したロギング)
1. `python mlflow_lightgbm_hydra_sample.py --multirun lightgbm.num_leaves=20,30,40 lightgbm.max_depth=5,7,9` (hydraのconfigパラメータを上書きして実行する場合)

## Note

* lightgbmのデータに`category`の列が含まれていると`WARNING mlflow.lightgbm: Failed to infer model signature`が出力されるようである。
* `set_tracking_uri()`のファイルパスの記述などは、環境に応じて多少変更が必要かもしれない。（Windowsで動作確認）

## Reference

* [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html)
* [MLflow Quickstart](https://www.mlflow.org/docs/latest/quickstart.html)
* [MLflowで実験管理入門](https://future-architect.github.io/articles/20200626/)
* [Getting started | Hydra](https://hydra.cc/docs/intro)
