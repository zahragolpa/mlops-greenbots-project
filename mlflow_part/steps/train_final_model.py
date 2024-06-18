import click
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import mlflow
from steps.preprocess_data import preprocess_data
from sklearn.linear_model import LogisticRegression

def train_model(data_path):
    x_train, x_test, y_train, y_test = preprocess_data(data_path)
    
    
    mlflow.set_experiment("cvd_bma")
    experiment = mlflow.get_experiment_by_name("cvd_bma")
    client = mlflow.tracking.MlflowClient()
    run = client.create_run(experiment.experiment_id)
    with mlflow.start_run(run_id = run.info.run_id):
        # Log parameters (Here we use default parameters of Logistic Regression)
        mlflow.log_param("C", 1.0)
        mlflow.log_param("solver", "lbfgs")
        
        # Create and train logistic regression model
        model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=10000)
        model.fit(x_train, y_train)
        
        preds = model.predict(x_test)
        ap = average_precision_score(y_test, preds)
        roc = roc_auc_score(y_test, preds)

        # Log metrics
        mlflow.log_metric("Test ROC AUC", roc)
        mlflow.log_metric("Test PR AUC", ap)
        
        # Log model
        mlflow.sklearn.log_model(model, "logistic_regression_model")

    return roc, ap


if __name__ == "__main__":
    train_model()
