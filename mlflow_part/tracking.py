# import json, os
# from dataclasses import dataclass
# from typing import List
# import shutil
# import pandas as pd
# import mlflow
# from mlflow.tracking import MlflowClient
# from mlflow.entities.run import Run
# from mlflow.pyfunc import PyFuncModel


# def create_temporary_dir_if_not_exists(tmp_dir_path: os.PathLike = 'tmp') -> None:
#     """creation of a temporary folder 

#     Args:
#         tmp_dir_path (os.PathLike, optional): Path of the folder. Defaults to 'tmp'.
#     """
#     if not os.path.exists(tmp_dir_path):
#         os.makedirs(tmp_dir_path)
#     return tmp_dir_path


# def clean_temporary_dir(tmp_dir_path: os.PathLike = 'tmp') -> None:
#     """delete the temporary folder

#     Args:
#         tmp_dir_path (os.PathLike, optional): Path of the folder. Defaults to 'tmp'.
#     """
#     if os.path.exists(tmp_dir_path):
#         shutil.rmtree(tmp_dir_path)
        

# def load_json(fpath):
#     # JSON file
#     with open(fpath, "r") as f:
#         # Reading from file
#         data = json.loads(f.read())
#     return data


# def cameltosnake(camel_string: str) -> str:
#     # If the input string is empty, return an empty string
#     if not camel_string:
#         return ""
#     # If the first character of the input string is uppercase,
#     # add an underscore before it and make it lowercase
#     elif camel_string[0].isupper():
#         return f"_{camel_string[0].lower()}{cameltosnake(camel_string[1:])}"
#     # If the first character of the input string is lowercase,
#     # simply return it and call the function recursively on the remaining string
#     else:
#         return f"{camel_string[0]}{cameltosnake(camel_string[1:])}"


# def camel_to_snake(s: str) -> str:
#     if len(s) <= 1:
#         return s.lower()
#     # Changing the first character of the input string to lowercase
#     # and calling the recursive function on the modified string
#     return cameltosnake(s[0].lower()+s[1:])

# @dataclass
# class Experiment:
#     tracking_server_uri: str
#     name: str


# def load_model_from_run(tracking_server_uri: str, run: Run):
    
#     mlflow.set_tracking_uri(tracking_server_uri)
#     client = MlflowClient(tracking_uri=tracking_server_uri)
#     artifact_path = json.loads(
#         run.data.tags['mlflow.log-model.history'])[0]["artifact_path"]
#     tmp_dir = create_temporary_dir_if_not_exists()
#     model_path = client.download_artifacts(run.info.run_id,
#                                            artifact_path,
#                                            tmp_dir)
#     model = mlflow.pyfunc.load_model(model_path)
#     clean_temporary_dir()
#     return model


# def get_best_run(experiment: Experiment,
#                  metric: str = "valid_roc_auc", order: str = "DESC") -> Run:
#     best_runs = explore_best_runs(experiment, 1, metric, order, False)
#     return best_runs[0]


# def explore_best_runs(experiment: Experiment, n_runs: int = 5, metric: str = "valid_roc_auc",
#                       order: str = "DESC", to_dataframe: bool = True) -> List[Run] | pd.DataFrame:

#     mlflow.set_tracking_uri(experiment.tracking_server_uri)
#     client = MlflowClient(tracking_uri=experiment.tracking_server_uri)
#     # Retrieve Experiment information
#     experiment_id = mlflow.set_experiment(experiment.name).experiment_id
#     # Retrieve Runs information
#     runs = client.search_runs(
#         experiment_ids=experiment_id,
#         max_results=n_runs,
#         order_by=[f"metrics.{metric} {order}"]
#     )
#     if to_dataframe:
#         run_ids = [run.info.run_id for run in runs if metric in run.data.metrics]
#         run_metrics = [run.data.metrics[metric]
#                        for run in runs if metric in run.data.metrics]
#         run_dataframe = pd.DataFrame({"Run ID": run_ids, "Perf.": run_metrics})
#         return run_dataframe
#     return runs


