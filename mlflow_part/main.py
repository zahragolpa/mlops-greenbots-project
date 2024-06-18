import sys

import mlflow
from steps.train_final_model import train_model



def pipeline():
    mlflow.set_experiment("cvd_bma")
    roc, pr = train_model(data_path='/home/armand/mlflow_project/data/data_cardio.csv')
    print(f"Model is trained. \nTestset ROC AUC: {roc}\nTestset PR AUC: {pr}")


if __name__ == "__main__":
    pipeline()
