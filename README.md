# mlflow-quick-sample

## Overview

MLflow(Tracking)を動作させるサンプル。ローカルで動作させることを想定。

パラメータやメトリクスをロギングしてMLflowのUIで確認するまでを実行する。

## Install

```
python -m pip install -r requirements.txt
```

## Quick Sample

1. `cd sample`
1. `python mlflow_sample.py` (基本的なAPIを利用したロギング)
1. `python mlflow_sample_lightgbm.py` (autologを利用したロギング)
1. `mlflow ui --host 0.0.0.0 --port 5000` (UIの起動)
1. `http://localhost:5000`にアクセス

## Reference

* [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html)
* [MLflow Quickstart](https://www.mlflow.org/docs/latest/quickstart.html)
* [MLflowで実験管理入門](https://future-architect.github.io/articles/20200626/)
