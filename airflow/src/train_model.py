import pandas as pd
from sklearn.linear_model import LinearRegression
import mlflow
import mlflow.sklearn
import joblib
import os

def train_model():
    data_path = '/opt/airflow/data/processed.csv'
    model_path = '/opt/airflow/data/model.pkl'

    df = pd.read_csv(data_path)
    X = df.drop(columns=['price'], errors='ignore')
    y = df['price'] if 'price' in df.columns else df.iloc[:, -1]

    model = LinearRegression()
    model.fit(X, y)

    # เชื่อมต่อกับ MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("UsedCarPricePrediction")

    with mlflow.start_run():
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("score", model.score(X, y))
        mlflow.sklearn.log_model(model, "model")

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
