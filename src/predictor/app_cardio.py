import os.path

import joblib
import pandas as pd

from flask import Flask, request, render_template, jsonify
from io import StringIO

from scripts.dataset_validator import validate_single_dataframe

app = Flask(__name__)

@app.route("/", methods=['GET'])
def upload():
    return render_template('mlops-project-upload.html')

@app.route("/predict", methods=['POST'])
def predict():
    rf_model = joblib.load('models/random_forest.pkl')
    # Check if file is uploaded
    if 'file' in request.files:
        uploaded_file = request.files.get('file')
        csv_data = StringIO(uploaded_file.stream.read().decode('UTF8'), newline=None)
        test_data = pd.read_csv(csv_data)
        test_data = test_data.drop(['cardio', 'id'], axis=1)
    else:
        # If file is not uploaded, read values from form
        test_data = pd.DataFrame({
            'gender': [int(request.form['gender'])],            
            'cholesterol': [int(request.form['cholesterol'])],
            'gluc': [int(request.form['gluc'])],
            'smoke': [int(request.form['smoke'])],
            'alco': [int(request.form['alco'])],
            'active': [int(request.form['active'])],
            'age_group': [int(request.form['age_group'])],
            'bmi': [int(request.form['bmi'])],
            'map': [int(request.form['map'])]
        })

    passes_deepchecks_data_integrity_validation = validate_single_dataframe(test_data, dataframe_name='cardio_dataset')

    predictions = rf_model.predict(test_data)
    input_data = test_data.values.tolist()
    result = list(zip(input_data, predictions))
    return render_template('prediction_result.html', predictions=result)

if __name__ == '__main__':
    print('inside app_cardio')
    app.run(host='0.0.0.0', port='6060', debug=True)
