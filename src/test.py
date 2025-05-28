import joblib
import pandas as pd

CarFeatures = {
    "manufacturer": "Toyota",
    "model": "Camry",
    "condition": "excellent",
    "cylinders": "4 cylinders",
    "fuel": "gas",
    "odometer": 75234.0,
    "title_status": "clean",
    "transmission": "automatic",
    "drive": "fwd",
    "paint_color": "silver",
    "type": "sedan",
    "car_age": 5,
    "region": "Los Angeles",
    "state": "CA"
}

def transform_raw_data(raw_data: pd.DataFrame):
    preprocessor_path = '../airflow/models/preprocessor.joblib'
    try:
        preprocessor = joblib.load(preprocessor_path)
    except FileNotFoundError:
        raise FileNotFoundError("Preprocessor not found. Please ensure it's trained and saved.")

    if 'price' in raw_data.columns:
        raw_data = raw_data.drop('price', axis=1)

    transformed_data = preprocessor.transform(raw_data)
    
    return transformed_data

print(transform_raw_data(pd.DataFrame([CarFeatures])))