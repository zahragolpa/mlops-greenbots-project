import pandas as pd
import joblib
from prefect import flow

@flow(name='Green Bots ML pipeline Predict')
def predict(path_pickle, data_path):
    # Load data
    data = pd.read_csv(data_path)
    data = data.drop(['cardio', 'id'], axis=1)

    # Load model
    model = joblib.load(path_pickle)

    # Predict
    predictions = model.predict(data)
    print(predictions)

if __name__ == '__main__':
    predict('../models/random_forest.pkl', '../../data/processed/prediction_data.csv')
    predict('../models/svm.pkl', '../../data/processed/prediction_data.csv')
