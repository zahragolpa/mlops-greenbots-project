from dataclasses import dataclass
from typing import List, Tuple
import json
import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.run import Run
from mlflow.pyfunc import PyFuncModel
from helpers.utils import create_temporary_dir_if_not_exists
from helpers.utils import clean_temporary_dir

@dataclass
class Experiment:
    """
    A dataclass used to represent an Experiment on MLflow
    Attributes
    ----------
    tracking_server_uri : str
        the URI of MLFlow experiment tracking server
    name : str
        the name of the experiment
    """
    tracking_server_uri:str
    name:str

def load_model_from_run(tracking_server_uri:str, run:Run):
    """load the model stored within a given experiment run

    Args:
        tracking_server_uri (str): mlflow tracking server URI
        run (Run): the entity of the given run

    Returns:
        model (PyFuncModel): the stored model
    """
    
    mlflow.set_tracking_uri(tracking_server_uri)
    client = MlflowClient(tracking_uri=tracking_server_uri)
    artifact_path = json.loads(run.data.tags['mlflow.log-model.history'])[0]["artifact_path"]
    tmp_dir = create_temporary_dir_if_not_exists()
    model_path = client.download_artifacts(run.info.run_id, 
                                                artifact_path, 
                                                tmp_dir)
    model = mlflow.pyfunc.load_model(model_path)
    clean_temporary_dir()
    return model

def get_best_run(experiment:Experiment, 
                 metric:str="valid_accuracy",
                 order:str="DESC",
                 filter_string:str=""):
    """Find the best experiment run entity

    Args:
        experiment (Experiment): experiment settings
        metric (str, optional): the metric for runs comparison. Defaults to "valid_accuracy".
        order (str, optional): the sorting order to find the best at first row w.r.t the metric. Defaults to "DESC".
        filter_string (str, optional): a string with which to filter the runs. Defaults to empty string, thus searching all runs.

    Returns:
        Run: the best run entity associated with the given experiment
    """
    best_runs = explore_best_runs(experiment, 1, metric, order, filter_string, False)
    return best_runs[0]

def explore_best_runs(experiment:Experiment, n_runs:int=5, metric:str="valid_accuracy", 
                      order:str="DESC", filter_string:str="", to_dataframe:bool=True):
    """find the best runs from the given experiment

    Args:
        experiment (Experiment): Experiment settings
        n_runs (int, optional): the count of runs to return. Defaults to 5.
        metric (str, optional): the metric for runs comparison. Defaults to "valid_accuracy".
        order (str, optional): the sorting order w.r.t the metric to have the best at first row. Defaults to "DESC".
        filter_string (str, optional): a string with which to filter the runs. Defaults to empty string, thus searching all runs.
        to_dataframe (bool, optional): True for a derived Dataframe of Run ID / Perf. Metric. Defaults to True.

    Returns:
        List[Run] | pd.DataFrame: set of the best runs (Entity or Dataframe)
    """
    mlflow.set_tracking_uri(experiment.tracking_server_uri)
    client = MlflowClient(tracking_uri=experiment.tracking_server_uri)
    # Retrieve Experiment information
    experiment_id = mlflow.set_experiment(experiment.name).experiment_id
    # Retrieve Runs information
    runs = client.search_runs(
        experiment_ids=experiment_id,
        max_results=n_runs,
        filter_string=filter_string,
        order_by=[f"metrics.{metric} {order}"]
    )
    if to_dataframe:
        run_ids = [run.info.run_id for run in runs if metric in run.data.metrics]
        run_metrics = [run.data.metrics[metric] for run in runs if metric in run.data.metrics]
        run_dataframe = pd.DataFrame({"Run ID": run_ids, "Perf.": run_metrics})
        return run_dataframe
    return runs

def get_raw_artifacts_from_run(tracking_server_uri, run, tmp_dir_path: os.PathLike='tmp'):
    client = MlflowClient(tracking_uri=tracking_server_uri)
    # We assume that our saves will be under classifier/artifacts
    feat_eng_path = client.download_artifacts(run.info.run_id, 'classifier/artifacts/feature_eng.joblib', tmp_dir_path)
    model_path = client.download_artifacts(run.info.run_id, 'classifier/artifacts/model.joblib', tmp_dir_path)
    return feat_eng_path, model_path
