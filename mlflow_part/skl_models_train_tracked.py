# import os
# import abc
# import pandas as pd
# import numpy as np
# from typing import List
# import joblib
# import mlflow
# import mlflow.pyfunc
# from mlflow.models import infer_signature
# import joblib
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import RocCurveDisplay, roc_auc_score, average_precision_score
# from tracking import create_temporary_dir_if_not_exists, clean_temporary_dir, Experiment, camel_to_snake
# from steps.preprocess_data import preprocess_data


# def train_and_evaluate_with_tracking(
#         data_path,
#         classifers_list: dict,
#         experiment: Experiment) -> None:
#     """
#     Trains each classifier from the list on the training data and evaluates on all splits, 
#     while tracking metrics and artifacts using MLflow.

#     Args:
#     - data_path : Dataset path.
#     - classifiers_list (List[abc.ABCMeta]): List of Scikit-learn classifier classes.
#     - experiment (Experiment): Experiment settings for MLflow tracking
#     """
#     # Set up MLflow tracking
#     mlflow.set_tracking_uri(experiment.tracking_server_uri)
#     experiment_id = mlflow.set_experiment(experiment.name).experiment_id
#     # Create a temporary directory for storing artifacts
#     tmp_dir = create_temporary_dir_if_not_exists()
#     def tmp_fpath(fpath): return os.path.join(tmp_dir, fpath)
#     # Transform the data for training, validation, and test sets
#     x_train, x_val, y_train, y_val = preprocess_data(data_path)
    
#     # Loop through each classifier in the list
#     for cls_name, classifier in classifers_list.items():
#         # Set up run-specific details
#         classifier_shortname = cls_name
#         with mlflow.start_run(experiment_id=experiment_id,
#                               run_name=f"run_{classifier_shortname}"):
#             mlflow.set_tag("sklearn_model", classifier_shortname)
#             # Instantiate and train the classifier
#             model = classifier
#             model.fit(x_train, y_train)
#             # joblib.dump(model, tmp_fpath('model.joblib'))
            
#             # Evaluate accuracy on different datasets
#             preds = model.predict(x_val)
#             ap = average_precision_score(y_val, preds)
#             roc = roc_auc_score(y_val, preds)
            
#             # Track accuracy metrics
#             mlflow.log_metric("valid_roc_auc", roc)
#             mlflow.log_metric("valid_avg_precision", ap)
            
#             # Generate an example input and a model signature
#             # sample = data.train_x.sample(3)
#             signature = infer_signature(x_train,
#                                         model.predict(x_train))
#             # Log the trained model as an MLflow artifact
#             mlflow_model_path = 'classifier'
#             mlflow.sklearn.log_model(
#                 sk_model=model,
#                 artifact_path=mlflow_model_path,
#                 signature=signature,
#                 registered_model_name="sk-learn-" +classifier_shortname+ "-model",
#             )

#             # Track ROC curve plots for validation and test sets
#             display = RocCurveDisplay.from_predictions(y_val,
#                                                        model.predict(x_val))
#             mlflow.log_figure(display.figure_, 'plots/TestRocCurveDisplay.png')
#     # Clean up temporary directory
#     clean_temporary_dir()

