import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def train_val_test_split(x, y, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test


def preprocess_data(dset_path, test_size=0.2):
    df = pd.read_csv(dset_path)
    x = df.drop(['cardio'], axis=1)
    y = df['cardio']
    # Train/test split 
    x_train, x_test, y_train, y_test = train_val_test_split(x, y, test_size=test_size)
    # Standardize features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test
