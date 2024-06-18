from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from preprocessor.preprocess import PreprocessData
import pandas as pd
from mlflow import MlflowClient
from scripts.model_validator import validate_model as dc_validate_model
from scripts.dataset_validator import validate_train_test_dataframe


import mlflow
from mlflow.data.pandas_dataset import PandasDataset
import pickle

from prefect import task, flow

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

class ModelManager:
    def __init__(self) -> None:
        pass
    
    def get_best_model(self):
        """
        Retrieve the model with the highest version.
        """
        runs = mlflow.search_runs(order_by=["artifact_version DESC"])
        print(runs)
        if not runs.empty:
            latest_model_version = runs.iloc[0]['artifact_version']
            best_model = mlflow.sklearn.load_model(f"models:/<model_name>/{latest_model_version}")
            return best_model
        else:
            print("No runs found. Make sure to log metrics and artifacts during training.")
            return None
        
    def get_latest_model_version(tracking_uri: str, model_name: str):
        """
        Retrieves the latest model version and its stage for a specified model.

        Args:
        - tracking_uri (str): The URI where the MLflow tracking server is running.
        - model_name (str): The name of the model to retrieve the latest version for.

        Returns:
        - Dict: A dictionary containing version and stage information
                for the latest version of the specified model.
                Example: {"version": "1", "stage": "Production"}
        """
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        latest_versions = client.get_latest_versions(name=model_name)
        
        if latest_versions:
            # Sort the versions based on version number in descending order
            latest_version = sorted(latest_versions, key=lambda x: int(x.version), reverse=True)[0]
            return {"version": latest_version.version, "stage": latest_version.current_stage}
        else:
            return None

    
    def stage_model(self, model, stage_name="Staging"):
        """
        Stage a model for deployment.
        
        Parameters:
        - model: The model to be staged.
        - stage_name: Name of the staging environment.
        """
        mlflow.sklearn.log_model(sk_model=model, artifact_path=stage_name)
    
    def deploy_model(self, model, deployment_name="Production"):
        """
        Deploy a model to production.
        
        Parameters:
        - model: The model to be deployed.
        - deployment_name: Name of the deployment environment.
        """
        mlflow.sklearn.log_model(sk_model=model, artifact_path=deployment_name, registered_model_name='model')
        

class TrainModel:
    def __init__(self) -> None:
        self.model_name = "untitled_model"

    def apply_label_encoder(self, df):
        """
        Apply label encoding to categorical columns in the DataFrame.

        Parameters:
        - df: DataFrame to apply label encoding.

        Returns:
        - df: DataFrame with label encoding applied.
        """
        le = LabelEncoder()
        try:
            df = df.apply(le.fit_transform)
            print("Label encoding applied successfully.")
        except Exception as e:
            print(f"An error occurred while applying label encoding: {e}")
        return df

    def split_data(self, df=None, columns_to_drop=['cardio', 'gender', 'alco', 'id'], target_column='cardio', test_size=0.20, random_state=1):
        """
        Split the DataFrame into training and testing sets.

        Parameters:
        - df: DataFrame to split.
        - columns_to_drop: List of columns to drop from features.
        - target_column: Column to predict.
        - test_size: Proportion of the dataset to include in the test split.
        - random_state: Seed for random number generation.

        Returns:
        - x_train, x_test, y_train, y_test: Split datasets.
        """
        x = df.drop(columns_to_drop, axis=1)
        y = df[target_column]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test

    def save_model(self, model, path):
        """
        Save the trained model to a file.

        Parameters:
        - model: Trained model to save.
        - path: File path to save the model.
        """
        try:
            with open(path, 'wb') as file:
                pickle.dump(model, file)
            print("Model saved successfully.")
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")

    def validate_model_with_deepchecks(self, train_x, train_y, test_x, test_y, model):
        """
        Validate the model using deepchecks.
        Args:
            train_x: Training data
            train_y: Training labels
            test_x: Test data
            test_y: Test labels
            model: A scikit-learn-compatible fitted estimator instance.

        Returns:
            passes (Boolean): True if the model passes validation, False otherwise
        """
        train_ds = train_x.join(train_y)
        test_ds = test_x.join(test_y)
        passes = dc_validate_model(train_ds, test_ds, model, self.model_name)
        return passes


class RandomForestTrainer(TrainModel):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "RandomForestClassifier"

    @task
    def perform_grid_search(self, x_train, y_train, param_grid, cv=5, scoring='accuracy', n_jobs=-1):
        """
        Perform grid search to find the best hyperparameters for RandomForestClassifier.

        Parameters:
        - x_train: Training features.
        - y_train: Training labels.
        - param_grid: Parameter grid for grid search.
        - cv: Number of cross-validation folds.
        - scoring: Scoring metric for evaluation.
        - n_jobs: Number of jobs to run in parallel.

        Returns:
        - best_estimator: Best trained RandomForestClassifier model.
        - best_params: Best hyperparameters found by grid search.
        """
        estimator = RandomForestClassifier(random_state=1)
        grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
        grid_search.fit(x_train, y_train)
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_
        return best_estimator, best_params

    @task
    def evaluate_model(self, model, x_test, y_test):
            """
            Evaluate the trained model on the test set and generate evaluation metrics.

            Parameters:
            - model: Trained model.
            - x_test: Test features.
            - y_test: Test labels.

            Returns:
            - evaluation_metrics: Dictionary containing accuracy, recall, and f1-score.
            """
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            evaluation_metrics = {
                "accuracy": accuracy,
                "recall": recall,
                "f1": f1
            }
            
            return evaluation_metrics
    
    def train_random_forest(self,data_path, model_destination_path):
        """
        Train a RandomForestClassifier, perform grid search for hyperparameter tuning,
        save the best model, and print the classification report.

        Parameters:
        - data_path: Path to the dataset.
        - model_destination_path: File path to save the trained model.
        """
        mlflow.set_experiment("Green Bots Learner")            
        with mlflow.start_run():            
           # Load data and preprocess
            PreprocessData.process_file(origin_path='../../data/raw/cardio_train.csv',
                              destination_dir='../../data/processed/')
            data = PreprocessData.load_data(path=data_path, separator=',')
            data = TrainModel.apply_label_encoder(self,df=data)
            
            # Split data into training and testing sets
            x_train, x_test, y_train, y_test = self.split_data(data, columns_to_drop=['cardio','id'], target_column='cardio')
            passes_deepchecks_train_test_validation = validate_train_test_dataframe(x_train.join(y_train),
                                                                                    x_test.join(y_test),
                                                                                    dataframe_name="cardio_dataset")

            # Define parameter grid for grid search
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None],
            }
            
            rf_best_params = {
                'n_estimators': [100],
                'max_depth': [10],
                'min_samples_split': [10],
                'min_samples_leaf': [1],
                'max_features': ['sqrt'],
            }

            # Perform grid search
            best_estimator, best_params = self.perform_grid_search(self,x_train, y_train, rf_best_params)

            # Evaluate the best model
            classification_report_dict = self.evaluate_model(self,best_estimator, x_test, y_test)

            # deepchecks
            passes_deepchecks_model_validation = self.validate_model_with_deepchecks(x_train,
                                                                                     y_train,
                                                                                     x_test,
                                                                                     y_test,
                                                                                     best_estimator)

            classification_report_dict['passes_deepchecks_model_validation'] = passes_deepchecks_model_validation
            classification_report_dict['passes_deepchecks_train_test_validation'] = passes_deepchecks_train_test_validation

            # Save the best model
            self.save_model(best_estimator, model_destination_path)
            dataset_example_path = "../data/dataset_example.csv"
            x_train.sample(5).to_csv(dataset_example_path, index=False)
            df = pd.read_csv(dataset_example_path)
            dataset: PandasDataset = mlflow.data.from_pandas(df, source=dataset_example_path)
            mlflow.log_input(dataset,"Training Sample")
            mlflow.set_tag("sklearn","Random Forest")
            mlflow.log_params(best_params)
            mlflow.log_metrics(classification_report_dict)
            mlflow.sklearn.log_model(sk_model=best_estimator, artifact_path="random_forest_model", registered_model_name='model')                
            print(classification_report_dict)


class SVMTrainer(TrainModel):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "SVC"

    def perform_grid_search(self, x_train, y_train, param_grid, cv=3, scoring='accuracy', n_jobs=2):
        """
        Perform grid search to find the best hyperparameters for SVM.

        Parameters:
        - x_train: Training features.
        - y_train: Training labels.
        - param_grid: Parameter grid for grid search.
        - cv: Number of cross-validation folds.
        - scoring: Scoring metric for evaluation.
        - n_jobs: Number of jobs to run in parallel.

        Returns:
        - best_estimator: Best trained SVM model.
        - best_params: Best hyperparameters found by grid search.
        """
        estimator = SVC(random_state=1)
        grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
        grid_search.fit(x_train, y_train)
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_
        return best_estimator, best_params

    def evaluate_model(self, model, x_test, y_test):
                """
                Evaluate the trained model on the test set and generate evaluation metrics.

                Parameters:
                - model: Trained model.
                - x_test: Test features.
                - y_test: Test labels.

                Returns:
                - evaluation_metrics: Dictionary containing accuracy, recall, and f1-score.
                """
                y_pred = model.predict(x_test)
                accuracy = accuracy_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                evaluation_metrics = {
                    "accuracy": accuracy,
                    "recall": recall,
                    "f1": f1
                }
                
                return evaluation_metrics

    def train_svm(self, data_path, model_destination_path):
        """
        Train a Support Vector Machine (SVM), perform grid search for hyperparameter tuning,
        save the best model, and print the classification report.

        Parameters:
        - data_path: Path to the dataset.
        - model_destination_path: File path to save the trained model.
        """
        # Load data and preprocess
        with mlflow.start_run():
            data = PreprocessData.load_data(self='processing',path=data_path, separator=',')
            data = self.apply_label_encoder(data)
            
            # Split data into training and testing sets
            x_train, x_test, y_train, y_test = self.split_data(data, columns_to_drop=['cardio', 'id'], target_column='cardio')
            passes_deepchecks_train_test_validation = validate_train_test_dataframe(x_train.join(y_train),
                                                                                    x_test.join(y_test),
                                                                                    dataframe_name="cardio_dataset")

            # Define parameter grid for grid search
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto'],
            }
            
            svm_best_params = {
                'C': [10],
                'kernel': ['rbf'],
                'gamma': ['auto'],
            }

            # Perform grid search
            best_estimator, best_params = self.perform_grid_search(x_train, y_train, svm_best_params)

            # Evaluate the best model
            classification_report_dict = self.evaluate_model(best_estimator, x_test, y_test)

            # deepchecks
            passes_deepchecks_model_validation = self.validate_model_with_deepchecks(x_train,
                                                                                     y_train,
                                                                                     x_test,
                                                                                     y_test,
                                                                                     best_estimator)

            classification_report_dict['passes_deepchecks_model_validation'] = passes_deepchecks_model_validation
            classification_report_dict[
                'passes_deepchecks_train_test_validation'] = passes_deepchecks_train_test_validation

            # Save the best model
            self.save_model(best_estimator, model_destination_path)

            # Print classification report
            mlflow.log_params(best_params)
            mlflow.log_metrics(classification_report_dict)
            mlflow.set_tag("Training cvd svd","greenbots")
            mlflow.sklearn.log_model(sk_model=best_estimator, artifact_path="svm", registered_model_name='cvd_svm')
            print(classification_report_dict)

class LogisticRegressionTrainer(TrainModel):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = 'LogisticRegression'

    @task
    def perform_grid_search(self, x_train, y_train, param_grid, cv=5, scoring='accuracy', n_jobs=-1):
        """
        Perform grid search to find the best hyperparameters for Logistic Regression.

        Parameters:
        - x_train: Training features.
        - y_train: Training labels.
        - param_grid: Parameter grid for grid search.
        - cv: Number of cross-validation folds.
        - scoring: Scoring metric for evaluation.
        - n_jobs: Number of jobs to run in parallel.

        Returns:
        - best_estimator: Best trained Logistic Regression model.
        - best_params: Best hyperparameters found by grid search.
        """
        estimator = LogisticRegression(random_state=1, max_iter=1000)
        grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
        grid_search.fit(x_train, y_train)
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_
        return best_estimator, best_params
    
    def evaluate_model(self, model, x_test, y_test):
        """
        Evaluate the trained model on the test set and generate evaluation metrics.

        Parameters:
        - model: Trained model.
        - x_test: Test features.
        - y_test: Test labels.

        Returns:
        - evaluation_metrics: Dictionary containing accuracy, recall, and f1-score.
        """
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        evaluation_metrics = {
            "accuracy": accuracy,
            "recall": recall,
            "f1": f1
        }
        
        return evaluation_metrics

    @task
    def train_logistic_regression(self, data_path, model_destination_path):
        """
        Train a Logistic Regression model, perform grid search for hyperparameter tuning,
        save the best model, and print the classification report.

        Parameters:
        - data_path: Path to the dataset.
        - model_destination_path: File path to save the trained model.
        """
        with mlflow.start_run():
            # Load data and preprocess
            data = PreprocessData.load_data(path=data_path, separator=',')
            data = self.apply_label_encoder(data)
            
            # Split data into training and testing sets
            x_train, x_test, y_train, y_test = self.split_data(data, columns_to_drop=['cardio', 'id'], target_column='cardio')
            passes_deepchecks_train_test_validation = validate_train_test_dataframe(x_train.join(y_train),
                                                                                    x_test.join(y_test),
                                                                                    dataframe_name="cardio_dataset")

            # Define parameter grid for grid search
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
            
            lr_best_params = {
                'C': [3],
                'penalty': ['l1'],
                'solver': ['liblinear']
            }

            # Perform grid search
            best_estimator, best_params = self.perform_grid_search(x_train, y_train, lr_best_params)

            # Evaluate the best model
            classification_report_dict = self.evaluate_model(best_estimator, x_test, y_test)

            #            # deepchecks
            passes_deepchecks_model_validation = self.validate_model_with_deepchecks(x_train,
                                                                                     y_train,
                                                                                     x_test,
                                                                                     y_test,
                                                                                     best_estimator)

            classification_report_dict['passes_deepchecks_model_validation'] = passes_deepchecks_model_validation
            classification_report_dict['passes_deepchecks_train_test_validation'] = passes_deepchecks_train_test_validation


            # Save the best model
            self.save_model(best_estimator, model_destination_path)
        
            mlflow.log_params(best_params)
            mlflow.log_metrics(classification_report_dict)
            mlflow.set_tag("Training cvd","greenbots")
            mlflow.sklearn.log_model(sk_model=best_estimator, artifact_path="logistic_regression_model", registered_model_name='cvd_lr')

            print(classification_report_dict)
