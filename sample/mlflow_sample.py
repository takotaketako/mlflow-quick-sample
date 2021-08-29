"""
基本的なAPIを利用したロギング
"""

import mlflow


def main():

    # set experiment (未指定の場合はDefaultが利用される)
    experiment_name: str = "test experiment"
    try:
        print(f"create experiment: {experiment_name}")
        mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        print(f"experiment '{experiment_name}' already exists")
    mlflow.set_experiment(experiment_name)

    # start run (create run id)
    with mlflow.start_run():

        # log param
        mlflow.log_param("param_int", 5)
        mlflow.log_param("param_str", "hoge")
        mlflow.log_param("param_list", [1, 2, 3, 4, 5])
        mlflow.log_param("param_dict", {"key": "value"})
        print(f"logged parameters")

        # log metric
        for i in range(10):
            mlflow.log_metric("accuracy", i / 10, step=i+1)
        print(f"logged metrics")

        # log artifact
        mlflow.log_artifact("./artifact.txt")
        print(f"logged artifacts")

    print(f"finish!!")


if __name__ == "__main__":
    main()
