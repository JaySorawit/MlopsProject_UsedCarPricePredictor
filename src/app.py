import pickle
import gdown
import joblib
import pandas as pd
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Initialize app
app = FastAPI(title="UsedCarPricePredictor")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model with error handling
MODEL_PATH = Path("mlflow/model.pkl")
GDRIVE_FILE_ID = "13tDSWMYMyUvUpLnG_vZIQRUoCRio2m5e"

def download_model():
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading model from Google Drive...")
    gdown.download(url, str(MODEL_PATH), quiet=False)

model = None

try:
    if not MODEL_PATH.exists():
        download_model()

    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
        print("Model loaded successfully.")

except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Input schema
class CarFeatures(BaseModel):
    manufacturer: str
    model: str
    condition: str
    cylinders: str
    fuel: str
    odometer: float
    title_status: str
    transmission: str
    drive: str
    paint_color: str
    type: str
    car_age: int
    region: str
    state: str

# Transform function
def transform_raw_data(raw_data: pd.DataFrame):
    preprocessor_path = 'airflow/models/preprocessor.joblib'
    try:
        preprocessor = joblib.load(preprocessor_path)
    except FileNotFoundError:
        raise FileNotFoundError("Preprocessor not found. Please ensure it's trained and saved.")

    if 'price' in raw_data.columns:
        raw_data = raw_data.drop('price', axis=1)

    transformed_data = preprocessor.transform(raw_data)
    return transformed_data

# Routes
@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Hello from MLOpsProject UsedCarPricePredictor"}

@app.get("/ping", tags=["Health"])
def ping():
    return {"status": "OK"}

@app.post("/predict", tags=["Prediction"])
def predict_price(car: CarFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Convert input to DataFrame
    data = pd.DataFrame([car.dict()])

    try:
        # Transform and predict
        processed = transform_raw_data(data)
        prediction = model.predict(processed)
        return {"predicted_price": round(float(prediction[0]), 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
