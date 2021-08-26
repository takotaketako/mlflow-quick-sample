# mlflow-quick-sample

## Overview

MLflow(Tracking)を動作させるサンプル。ローカルで動作させることを想定。

## Install

```
python -m pip install -r requirements.txt
```

## Quick Sample

以下を実行して`http://localhost:5000`にアクセスする。

```
cd sample
python mlflow_sample.py
mlflow ui --host 0.0.0.0 --port 5000
```

## Reference

* [MLflow Tracking](https://www.mlflow.org/docs/latest/tracking.html)
* [MLflow Quickstart](https://www.mlflow.org/docs/latest/quickstart.html)
* [MLflowで実験管理入門](https://future-architect.github.io/articles/20200626/)
