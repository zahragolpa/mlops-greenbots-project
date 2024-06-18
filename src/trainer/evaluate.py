import pandas as pd
from sklearn.metrics import classification_report
import joblib

def evaluate(model_path, test_data_path):
    # Load test data
    test_data = pd.read_csv(test_data_path)

    # Extract features and target
    x_test = test_data.drop(['cardio', 'id'], axis=1)
    y_test = test_data['cardio']

    # Load model
    model = joblib.load(model_path)

    # Evaluate model
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(report)

if __name__ == '__main__':
    evaluate('../models/random_forest.pkl', '../../data/processed/validation_data.csv')
    evaluate('../models/svm.pkl', '../../data/processed/validation_data.csv')
