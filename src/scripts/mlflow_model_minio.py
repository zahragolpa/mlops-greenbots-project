from minio import Minio
import tempfile
import os
import cloudpickle
from mlflows.registry import load_production_model 


ACCESS_KEY = "minio7777"
SECRET_KEY = "minio8858"
MINIO_API_HOST = "localhost:31975"
MINIO_BUCKET_NAME = "model"  # Bucket where you want to store the model in MinIO

MINIO_CLIENT = Minio(MINIO_API_HOST, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False)

def upload_model_to_minio(model, object_name):
    # Create a temporary directory to save the model file
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Serialize the model using cloudpickle and save it to a file in the temporary directory
        model_path = os.path.join(tmpdirname, "model.pkl")
        with open(model_path, "wb") as f:
            cloudpickle.dump(model, f)

        # Upload the model file to MinIO
        MINIO_CLIENT.fput_object(MINIO_BUCKET_NAME, object_name, model_path)

    print(f"Model uploaded to MinIO at '{MINIO_BUCKET_NAME}/{object_name}'")

if __name__ == '__main__':
    # Load the model from MLflow
    model_name = 'model'  # Replace 'your_model_name' with the actual name of your MLflow model
    version = '9'   # Replace 'your_model_version' with the desired version of your MLflow model
    model = load_production_model('http://127.0.0.1:5000',model_name)

    # Specify the object name in MinIO where you want to store the model
    object_name = 'random_forest_model.pkl'

    # Upload the model to MinIO
    upload_model_to_minio(model, object_name)
