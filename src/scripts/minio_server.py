import os
from minio import Minio
from io import StringIO
import pandas as pd

ACCESS_KEY = "minio7777"
SECRET_KEY = "minio8858"
MINIO_API_HOST = "localhost:31975"
MINIO_CLIENT = Minio(MINIO_API_HOST, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False)

# Ensure buckets exist or create them
for bucket_name in ["cardiodata", "model"]:
    if not MINIO_CLIENT.bucket_exists(bucket_name):
        MINIO_CLIENT.make_bucket(bucket_name)
    else:
        print(f"Bucket '{bucket_name}' already exists")


MINIO_CLIENT.fput_object("cardiodata", 'data/processed/validation_data.csv', '../../data/processed/validation_data.csv')

print(f"Sample data uploaded to MinIO at 'cardiodata/data/processed/validation_data.csv'")
